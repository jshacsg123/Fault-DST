from __future__ import annotations
import random
import torch
from .augment import resize3d


def random_crop(seis, label, mask, crop_size, focus_mask=None):
    _, d, h, w = seis.shape
    if min(d, h, w) < crop_size:
        nd, nh, nw = max(d, crop_size), max(h, crop_size), max(w, crop_size)
        seis = resize3d(seis, (nd, nh, nw), 'trilinear')
        if label is not None:
            label = resize3d(label, (nd, nh, nw), 'nearest')
        if mask is not None:
            mask = resize3d(mask, (nd, nh, nw), 'nearest')
        d, h, w = nd, nh, nw
        if focus_mask is not None:
            focus_mask = resize3d(focus_mask, (nd, nh, nw), 'nearest')
    if focus_mask is not None and torch.any(focus_mask > 0):
        coords = torch.nonzero(focus_mask[0] > 0, as_tuple=False)
        center = coords[random.randint(0, coords.shape[0] - 1)]
        starts = []
        for val, size in zip(center.tolist(), (d, h, w)):
            lo = max(0, val - crop_size + 1)
            hi = min(val, size - crop_size)
            starts.append(lo if hi < lo else random.randint(lo, hi))
        sd, sh, sw = starts
    else:
        sd = 0 if d == crop_size else random.randint(0, d - crop_size)
        sh = 0 if h == crop_size else random.randint(0, h - crop_size)
        sw = 0 if w == crop_size else random.randint(0, w - crop_size)
    seis = seis[:, sd:sd+crop_size, sh:sh+crop_size, sw:sw+crop_size].contiguous()
    label = None if label is None else label[:, sd:sd+crop_size, sh:sh+crop_size, sw:sw+crop_size].contiguous()
    mask = None if mask is None else mask[:, sd:sd+crop_size, sh:sh+crop_size, sw:sw+crop_size].contiguous()
    return seis, label, mask
