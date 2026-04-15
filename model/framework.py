from __future__ import annotations
from copy import deepcopy
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from .hrnet import HRNet
from .swin_transformer3d import SwinTransformer3DSeg


class SMWDice(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p = prob.reshape(prob.shape[0], -1) * mask.reshape(mask.shape[0], -1)
        t = target.reshape(target.shape[0], -1) * mask.reshape(mask.shape[0], -1)
        inter = torch.sum(p * t, dim=1)
        denom = self.alpha * torch.sum(p, dim=1) + self.beta * torch.sum(t, dim=1)
        return (1.0 - (inter + self.smooth) / (denom + self.smooth)).mean()


class HardPseudoLoss(nn.Module):
    """
    Paper-aligned hard pseudo-label supervision:
    use argmax(student2) / threshold(student2) to supervise student1 only.
    For binary segmentation, argmax over [1-p, p] is equivalent to threshold at 0.5.
    """
    def _dice(self, prob: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
        p = prob.reshape(prob.shape[0], -1)
        t = pseudo.reshape(pseudo.shape[0], -1)
        inter = torch.sum(p * t, dim=1)
        denom = torch.sum(p, dim=1) + torch.sum(t, dim=1)
        return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

    def forward(self, student_prob: torch.Tensor, teacher_like_prob: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pseudo = (teacher_like_prob >= 0.5).float()
        return self._dice(student_prob, pseudo)


class SoftConsistency(nn.Module):
    """
    Paper-aligned soft consistency:
    sharpen student2 probabilities and use them as soft targets for student1.
    """
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def sharpen(self, p: torch.Tensor) -> torch.Tensor:
        t = self.tau
        num = p.pow(1.0 / t)
        den = num + (1.0 - p).pow(1.0 / t) + 1e-8
        return num / den

    def forward(self, student_prob: torch.Tensor, teacher_like_prob: torch.Tensor) -> torch.Tensor:
        soft_target = self.sharpen(teacher_like_prob.detach())
        return F.mse_loss(student_prob, soft_target)


class DCCALoss(nn.Module):
    """
    Paper-aligned DCCA for binary segmentation.
    Build a bi-classifier relevance matrix A from [background, fault] probabilities,
    then minimize the off-diagonal energy only.
    """
    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        p1f = p1.reshape(p1.shape[0], -1)
        p2f = p2.reshape(p2.shape[0], -1)
        p1b = 1.0 - p1f
        p2b = 1.0 - p2f

        a00 = torch.mean(p1b * p2b)
        a01 = torch.mean(p1b * p2f)
        a10 = torch.mean(p1f * p2b)
        a11 = torch.mean(p1f * p2f)

        relevance = torch.stack([
            torch.stack([a00, a01]),
            torch.stack([a10, a11]),
        ])
        off_diag = relevance.sum() - torch.trace(relevance)
        return off_diag


class InfoConsistency(nn.Module):
    def __init__(self, lambda_kl: float = 0.3):
        super().__init__()
        self.lambda_kl = lambda_kl

    def _slice(self, x: torch.Tensor, axis: int, s: int, e: int) -> torch.Tensor:
        idx = [slice(None)] * 5
        idx[axis + 2] = slice(s, e)
        return x[tuple(idx)]

    def _feat_bounds(self, s: int, e: int, full: int, feat: int):
        fs = int(s * feat / max(full, 1))
        fe = max(fs + 1, int((e * feat + full - 1) / max(full, 1)))
        return min(fs, feat - 1), min(fe, feat)

    def _single_sample_loss(
        self,
        sf: torch.Tensor,
        tf: torch.Tensor,
        sp: torch.Tensor,
        tp: torch.Tensor,
        overlap: Dict[str, int],
        full_size: int,
    ) -> torch.Tensor:
        axis = int(overlap['axis'])
        cs = int(overlap['cur_start'])
        ce = int(overlap['cur_end'])
        a_s = int(overlap['adj_start'])
        ae = int(overlap['adj_end'])

        sp = self._slice(sp.unsqueeze(0).clamp(1e-6, 1 - 1e-6), axis, cs, ce)
        tp = self._slice(tp.unsqueeze(0).clamp(1e-6, 1 - 1e-6), axis, a_s, ae)
        kl = F.kl_div(torch.log(sp), tp, reduction='batchmean')

        feat_axis = sf.shape[axis + 1]
        cfs, cfe = self._feat_bounds(cs, ce, full_size, feat_axis)
        afs, afe = self._feat_bounds(a_s, ae, full_size, feat_axis)
        sf = self._slice(sf.unsqueeze(0), axis, cfs, cfe).flatten(2)
        tf = self._slice(tf.unsqueeze(0), axis, afs, afe).flatten(2)
        length = min(sf.shape[-1], tf.shape[-1])
        sf = sf[..., :length]
        tf = tf[..., :length]
        cos = 1.0 - F.cosine_similarity(sf, tf, dim=1).mean()
        return cos + self.lambda_kl * kl

    def forward(
        self,
        sf: torch.Tensor,
        tf: torch.Tensor,
        sp: torch.Tensor,
        tp: torch.Tensor,
        overlap: dict,
        full_size: int,
    ) -> torch.Tensor:
        batch = sf.shape[0]
        losses = []
        for b in range(batch):
            sample_overlap = {}
            for k, v in overlap.items():
                if isinstance(v, torch.Tensor):
                    sample_overlap[k] = int(v[b].item()) if v.ndim > 0 else int(v.item())
                else:
                    sample_overlap[k] = int(v)
            losses.append(self._single_sample_loss(sf[b], tf[b], sp[b], tp[b], sample_overlap, full_size))
        return torch.stack(losses).mean()


class DualStudentTeacherModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.student1 = HRNet(args.cnn_base_width)
        self.student2 = SwinTransformer3DSeg(
            patch_size=args.swin_patch_size,
            embed_dim=args.swin_embed_dim,
            depths=args.swin_depths,
            num_heads=args.swin_num_heads,
            window_size=args.swin_window_size,
            mlp_ratio=args.swin_mlp_ratio,
            drop_rate=args.swin_drop_rate,
            attn_drop_rate=args.swin_attn_drop_rate,
            drop_path_rate=args.swin_drop_path_rate,
            use_checkpoint=getattr(args, 'swin_use_checkpoint', False),
        )
        self.teacher = deepcopy(self.student2)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.sup = SMWDice(args.smw_alpha, args.smw_beta)
        self.hard = HardPseudoLoss()
        self.soft = SoftConsistency(args.soft_temperature)
        self.dcca = DCCALoss()
        self.ic = InfoConsistency(args.lambda_kl)

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        for s, t in zip(self.student2.parameters(), self.teacher.parameters()):
            t.data.mul_(momentum).add_(s.data, alpha=1.0 - momentum)

    def _branch(self, model: nn.Module, x: torch.Tensor):
        feat, logit = model(x.contiguous())
        return feat, torch.sigmoid(logit)

    def _maybe_downsample_unsup(self, cube: torch.Tensor, factor: int):
        if factor <= 1:
            return cube
        return F.avg_pool3d(cube, kernel_size=factor, stride=factor)

    def _scaled_overlap(self, overlap: dict, factor: int):
        if factor <= 1:
            return overlap
        out = {}
        for k, v in overlap.items():
            if k == 'axis':
                out[k] = v
            else:
                out[k] = torch.div(v, factor, rounding_mode='floor')
        out['cur_end'] = torch.maximum(out['cur_end'], out['cur_start'] + 1)
        out['adj_end'] = torch.maximum(out['adj_end'], out['adj_start'] + 1)
        return out

    def supervised_batch_loss(self, batch: dict, weight: float = 1.0, tag: str = 'batch'):
        _, p1 = self._branch(self.student1, batch['seis'])
        _, p2 = self._branch(self.student2, batch['seis'])
        loss1 = self.sup(p1, batch['label'], batch['mask'])
        loss2 = self.sup(p2, batch['label'], batch['mask'])
        total = weight * (loss1 + loss2)
        return total, {f'{tag}_cnn': loss1.detach(), f'{tag}_swin': loss2.detach()}

    def unsupervised_losses(self, uns: dict, ramp_weight: float = 1.0):
        factor = max(1, int(getattr(self.args, 'consistency_downsample', 1)))
        unsup_cube = self._maybe_downsample_unsup(uns['unsup_cube'], factor)
        current_cube = self._maybe_downsample_unsup(uns['current_cube'], factor)
        adjacent_cube = self._maybe_downsample_unsup(uns['adjacent_cube'], factor)
        scaled_overlap = self._scaled_overlap(uns['overlap'], factor)

        _, up1 = self._branch(self.student1, unsup_cube)
        _, up2 = self._branch(self.student2, unsup_cube)
        loss_h = self.hard(up1, up2)
        loss_s = self.soft(up1, up2)
        loss_dcca = self.dcca(up1, up2)

        student_feat, student_prob = self._branch(self.student2, current_cube)
        with torch.no_grad():
            teacher_feat, teacher_prob = self._branch(self.teacher, adjacent_cube)
        loss_ic = self.ic(student_feat, teacher_feat, student_prob, teacher_prob, scaled_overlap, full_size=current_cube.shape[-1])

        total = ramp_weight * (
            self.args.alpha_h * loss_h +
            self.args.beta_s * loss_s +
            self.args.gamma_dcca * loss_dcca +
            self.args.mu_ic * loss_ic
        )
        return total, {
            'loss_h': loss_h.detach(),
            'loss_s': loss_s.detach(),
            'loss_dcca': loss_dcca.detach(),
            'loss_ic': loss_ic.detach(),
        }

    def pretrain_forward(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
        _, p1 = self._branch(self.student1, x)
        _, p2 = self._branch(self.student2, x)
        l1 = self.sup(p1, y, m)
        l2 = self.sup(p2, y, m)
        return {
            'total_loss': l1 + l2,
            'sup_cnn': l1.detach(),
            'sup_swin': l2.detach(),
        }

    def joint_forward(self, syn: dict, fld: dict, uns: dict, ramp_weight: float = 1.0):
        sup_syn_loss, _ = self.supervised_batch_loss(syn, weight=1.0, tag='syn')
        sup_fld_loss, _ = self.supervised_batch_loss(fld, weight=self.args.field_sup_weight, tag='field')
        unsup_total, uns_logs = self.unsupervised_losses(uns, ramp_weight)
        sup_total = sup_syn_loss + sup_fld_loss
        total = sup_total + unsup_total
        return {
            'total_loss': total,
            'sup_total': sup_total.detach(),
            'sup_syn': sup_syn_loss.detach(),
            'sup_field': sup_fld_loss.detach(),
            'loss_h': uns_logs['loss_h'],
            'loss_s': uns_logs['loss_s'],
            'loss_dcca': uns_logs['loss_dcca'],
            'loss_ic': uns_logs['loss_ic'],
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        self.eval()
        _, p1 = self._branch(self.student1, x)
        _, p2 = self._branch(self.student2, x)
        _, pt = self._branch(self.teacher, x)
        self.train()
        return {'cnn': p1, 'swin': p2, 'teacher': pt}
