import argparse
from typing import List


def parse_int_list(value: str) -> List[int]:
    if isinstance(value, list):
        return value
    return [int(v.strip()) for v in str(value).split(',') if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Dual-student 3D fault detection with SwinTransformer3D')
    p.add_argument('--syn_train_dir', type=str, default='data/synthetic/train')
    p.add_argument('--syn_val_dir', type=str, default='data/synthetic/val')
    p.add_argument('--field_train_dir', type=str, default='data/field/train')
    p.add_argument('--field_unlabeled_dir', type=str, default='data/field/unlabeled')
    p.add_argument('--save_dir', type=str, default='output/checkpoints')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--amp', action='store_true')

    p.add_argument('--train_size', type=int, default=128)
    p.add_argument('--ignore_label', type=float, default=-1.0)
    p.add_argument('--min_overlap', type=float, default=0.25)
    p.add_argument('--max_overlap', type=float, default=0.50)

    p.add_argument('--pretrain_steps', type=int, default=20000)
    p.add_argument('--joint_steps', type=int, default=80000)
    p.add_argument('--pretrain_batch_size', type=int, default=10)
    p.add_argument('--joint_batch_size', type=int, default=10)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--val_every', type=int, default=1000)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--unsup_ramp_steps', type=int, default=8000)

    p.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'])
    p.add_argument('--pretrain_lr', type=float, default=1e-3)
    p.add_argument('--joint_lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--teacher_momentum', type=float, default=0.99)
    p.add_argument('--warmup_ratio', type=float, default=0.05)

    p.add_argument('--cnn_base_width', type=int, default=8)

    p.add_argument('--swin_patch_size', type=int, default=4)
    p.add_argument('--swin_embed_dim', type=int, default=24)
    p.add_argument('--swin_depths', type=str, default='1,1,2,1')
    p.add_argument('--swin_num_heads', type=str, default='2,4,8,16')
    p.add_argument('--swin_window_size', type=int, default=4)
    p.add_argument('--swin_mlp_ratio', type=float, default=4.0)
    p.add_argument('--swin_drop_rate', type=float, default=0.0)
    p.add_argument('--swin_attn_drop_rate', type=float, default=0.0)
    p.add_argument('--swin_drop_path_rate', type=float, default=0.1)
    p.add_argument('--swin_use_checkpoint', action='store_true')

    p.add_argument('--smw_alpha', type=float, default=1.0)
    p.add_argument('--smw_beta', type=float, default=1.5)
    p.add_argument('--field_sup_weight', type=float, default=1.0)
    p.add_argument('--consistency_downsample', type=int, default=1)
    p.add_argument('--hard_conf_threshold', type=float, default=0.6)
    p.add_argument('--soft_temperature', type=float, default=0.1)
    p.add_argument('--lambda_kl', type=float, default=0.3)
    p.add_argument('--alpha_h', type=float, default=0.5)
    p.add_argument('--beta_s', type=float, default=0.5)
    p.add_argument('--gamma_dcca', type=float, default=0.5)
    p.add_argument('--mu_ic', type=float, default=1.0)
    return p


def get_args():
    args = build_parser().parse_args()
    args.swin_depths = parse_int_list(args.swin_depths)
    args.swin_num_heads = parse_int_list(args.swin_num_heads)
    if len(args.swin_depths) != 4 or len(args.swin_num_heads) != 4:
        raise ValueError('swin_depths and swin_num_heads must each have 4 integers, e.g. 1,1,2,1 and 2,4,8,16')
    return args
