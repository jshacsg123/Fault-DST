from __future__ import annotations
import math
import os
import random
from itertools import cycle
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from config import get_args
from datasets import SyntheticDataset, FieldSparseDataset, FieldUnsupervisedDataset
from model import DualStudentTeacherModel
from utils.io_utils import ensure_dir
from utils.metrics import seg_scores


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def make_loader(ds, batch_size, workers, shuffle=True, drop_last=True):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def optimizer_for(args, model, stage: str):
    lr = args.pretrain_lr if stage == 'pretrain' else args.joint_lr
    if args.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)


def set_cosine_warmup_lr(optimizer, base_lr: float, step_idx: int, total_steps: int, warmup_ratio: float):
    if total_steps <= 0:
        return base_lr
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
    cur_step = step_idx + 1
    if cur_step <= warmup_steps:
        lr = base_lr * cur_step / warmup_steps
    else:
        progress = (cur_step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def validate(model, loader, device):
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            pred = model.predict(batch['seis'])['teacher']
            values.append(seg_scores(pred, batch['label'], batch['mask']))
    model.train()
    if not values:
        return {'iou': 0.0, 'dice': 0.0, 'f1': 0.0}
    return {k: float(np.mean([v[k] for v in values])) for k in values[0]}


def save_ckpt(path, model, optimizer, stage, step, best_iou):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'stage': stage,
        'step': step,
        'best_iou': best_iou,
    }, path)


def load_ckpt(args, model, device):
    if not args.resume:
        return 'pretrain', 0, 0, 0.0, None
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    stage = ckpt.get('stage', 'pretrain')
    pstep = ckpt['step'] if stage == 'pretrain' else 0
    jstep = ckpt['step'] if stage == 'joint' else 0
    return stage, pstep, jstep, float(ckpt.get('best_iou', 0.0)), ckpt.get('optimizer', None)


def ramp(step, total):
    if total <= 0:
        return 1.0
    return min(1.0, step / total)


def main():
    args = get_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    scaler = amp.GradScaler(enabled=args.amp and device.type == 'cuda')

    syn_train = SyntheticDataset(
        args.syn_train_dir,
        args.train_size,
        max(args.pretrain_steps, args.joint_steps),
        max(args.pretrain_batch_size, args.joint_batch_size),
        True,
    )
    fld_sup = FieldSparseDataset(args.field_train_dir, args.train_size, args.joint_steps, args.joint_batch_size, args.ignore_label, True)
    fld_uns = FieldUnsupervisedDataset(
        args.field_train_dir,
        args.field_unlabeled_dir,
        args.train_size,
        args.joint_steps,
        args.joint_batch_size,
        args.min_overlap,
        args.max_overlap,
    )

    syn_pre_loader = make_loader(syn_train, args.pretrain_batch_size, args.num_workers, True, True)
    syn_joint_loader = make_loader(syn_train, args.joint_batch_size, args.num_workers, True, True)
    fld_sup_loader = make_loader(fld_sup, args.joint_batch_size, args.num_workers, True, True)
    fld_uns_loader = make_loader(fld_uns, args.joint_batch_size, args.num_workers, True, True)

    val_loader = None
    if os.path.isdir(args.syn_val_dir) and any(f.endswith(('.npz', '.npy')) for f in os.listdir(args.syn_val_dir)):
        val_loader = make_loader(SyntheticDataset(args.syn_val_dir, args.train_size, 1, 1, False), 1, 0, False, False)

    model = DualStudentTeacherModel(args).to(device)
    stage, pstart, jstart, best_iou, opt_state = load_ckpt(args, model, device)

    if stage == 'pretrain' and args.pretrain_steps > 0:
        optimizer = optimizer_for(args, model, 'pretrain')
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        syn_iter = cycle(syn_pre_loader)
        print('===== Stage 1: synthetic pretraining =====', flush=True)
        for step in range(pstart, args.pretrain_steps):
            lr = set_cosine_warmup_lr(optimizer, args.pretrain_lr, step, args.pretrain_steps, args.warmup_ratio)
            batch = to_device(next(syn_iter), device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=scaler.is_enabled()):
                losses = model.pretrain_forward(batch['seis'], batch['label'], batch['mask'])
            scaler.scale(losses['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_teacher(args.teacher_momentum)
            if step == 0 or (step + 1) % args.log_every == 0:
                print(
                    f"Pretrain [{step + 1}/{args.pretrain_steps}] lr={lr:.6e} "
                    f"total={losses['total_loss'].item():.4f} cnn={losses['sup_cnn'].item():.4f} swin={losses['sup_swin'].item():.4f}",
                    flush=True,
                )
            if (step + 1) % args.save_every == 0:
                save_ckpt(os.path.join(args.save_dir, f'pretrain_{step + 1:06d}.pth'), model, optimizer, 'pretrain', step + 1, best_iou)
        stage, opt_state = 'joint', None

    if args.joint_steps > 0:
        optimizer = optimizer_for(args, model, 'joint')
        if stage == 'joint' and opt_state is not None:
            optimizer.load_state_dict(opt_state)
        syn_iter = cycle(syn_joint_loader)
        fld_sup_iter = cycle(fld_sup_loader)
        fld_uns_iter = cycle(fld_uns_loader)
        print('===== Stage 2: joint training =====', flush=True)
        for step in range(jstart, args.joint_steps):
            lr = set_cosine_warmup_lr(optimizer, args.joint_lr, step, args.joint_steps, args.warmup_ratio)
            syn = to_device(next(syn_iter), device)
            fld = to_device(next(fld_sup_iter), device)
            uns = to_device(next(fld_uns_iter), device)
            rw = ramp(step + 1, args.unsup_ramp_steps)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=scaler.is_enabled()):
                logs = model.joint_forward(syn, fld, uns, rw)
            scaler.scale(logs['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_teacher(args.teacher_momentum)

            if step == 0 or (step + 1) % args.log_every == 0:
                print(
                    f"Joint [{step + 1}/{args.joint_steps}] lr={lr:.6e} total={logs['total_loss'].item():.4f} "
                    f"sup={logs['sup_total'].item():.4f} h={logs['loss_h'].item():.4f} "
                    f"s={logs['loss_s'].item():.4f} dcca={logs['loss_dcca'].item():.4f} "
                    f"ic={logs['loss_ic'].item():.4f} ramp={rw:.3f}",
                    flush=True,
                )
            if val_loader is not None and (step + 1) % args.val_every == 0:
                scores = validate(model, val_loader, device)
                print(
                    f"Val [{step + 1}/{args.joint_steps}] iou={scores['iou']:.4f} f1={scores['f1']:.4f} dice={scores['dice']:.4f}",
                    flush=True,
                )
                if scores['iou'] >= best_iou:
                    best_iou = scores['iou']
                    save_ckpt(os.path.join(args.save_dir, 'best_teacher.pth'), model, optimizer, 'joint', step + 1, best_iou)
            if (step + 1) % args.save_every == 0:
                save_ckpt(os.path.join(args.save_dir, f'joint_{step + 1:06d}.pth'), model, optimizer, 'joint', step + 1, best_iou)
        save_ckpt(os.path.join(args.save_dir, 'last.pth'), model, optimizer, 'joint', args.joint_steps, best_iou)


if __name__ == '__main__':
    main()
