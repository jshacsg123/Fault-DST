from __future__ import annotations
import random
import torch
import torch.nn.functional as F


def normalize_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    std = torch.std(x)
    if std < 1e-8:
        return torch.zeros_like(x)
    x = (x - torch.mean(x)) / std
    x = torch.clamp(x, -3.2, 3.2)
    mn, mx = torch.min(x), torch.max(x)
    if (mx - mn) < 1e-8:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)


def pair_geo(vols):
    outs = [v.clone() for v in vols]
    if random.random() < 0.5:
        outs = [torch.flip(v, (-1,)) for v in outs]
    if random.random() < 0.5:
        outs = [torch.flip(v, (-2,)) for v in outs]
    if random.random() < 0.5:
        outs = [torch.flip(v, (-3,)) for v in outs]
    if random.random() < 0.5:
        outs = [v.transpose(-1, -2).contiguous() for v in outs]
    k = random.randint(0, 3)
    if k:
        outs = [torch.rot90(v, k, dims=(-2, -1)).contiguous() for v in outs]
    return outs


def intensity_aug(seis: torch.Tensor) -> torch.Tensor:
    x = seis.clone().float()
    if random.random() < 0.3:
        x = x + torch.randn_like(x) * random.uniform(0.01, 0.05)
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.5)
        x0 = torch.min(x)
        x = torch.pow(torch.clamp(x - x0, min=0) + 1e-6, gamma)
    if random.random() < 0.2:
        x = F.avg_pool3d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    return normalize_torch(x)


def resize3d(x: torch.Tensor, size, mode='trilinear') -> torch.Tensor:
    kwargs = {'mode': mode}
    if mode in {'trilinear', 'bilinear'}:
        kwargs['align_corners'] = True
    return F.interpolate(x.unsqueeze(0), size=size, **kwargs).squeeze(0)
