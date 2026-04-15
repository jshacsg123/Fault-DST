from __future__ import annotations
import torch


def seg_scores(prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, th: float = 0.5):
    pred = (prob >= th).float() * mask
    tgt = target.float() * mask
    tp = torch.sum(pred * tgt).item()
    fp = torch.sum(pred * (1 - tgt)).item()
    fn = torch.sum((1 - pred) * tgt).item()
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        'iou': iou,
        'dice': dice,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
