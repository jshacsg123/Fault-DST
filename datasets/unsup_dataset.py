from __future__ import annotations
import os
import random
import torch
from torch.utils.data import Dataset
from .augment import intensity_aug, resize3d
from utils.io_utils import list_files, load_volume, to_seis_tensor


class FieldUnsupervisedDataset(Dataset):
    def __init__(self, sparse_root, unlabeled_root, crop_size, steps, batch_size, min_overlap=0.25, max_overlap=0.50):
        if not unlabeled_root:
            raise ValueError(
                'field_unlabeled_dir must be provided explicitly to match the paper setting of a separate unlabeled field set.'
            )
        if not os.path.isdir(unlabeled_root):
            raise FileNotFoundError(unlabeled_root)
        self.files = list_files(unlabeled_root)
        if not self.files:
            raise FileNotFoundError(unlabeled_root)
        self.crop_size = crop_size
        self.steps = steps
        self.batch_size = batch_size
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap

    def __len__(self):
        return max(1, self.steps * self.batch_size)

    def _random_cube(self, seis):
        _, d, h, w = seis.shape
        if min(d, h, w) < self.crop_size:
            nd, nh, nw = max(d, self.crop_size), max(h, self.crop_size), max(w, self.crop_size)
            seis = resize3d(seis, (nd, nh, nw), 'trilinear')
            d, h, w = nd, nh, nw
        sd = 0 if d == self.crop_size else random.randint(0, d - self.crop_size)
        sh = 0 if h == self.crop_size else random.randint(0, h - self.crop_size)
        sw = 0 if w == self.crop_size else random.randint(0, w - self.crop_size)
        cube = seis[:, sd:sd+self.crop_size, sh:sh+self.crop_size, sw:sw+self.crop_size].contiguous()
        return cube, (sd, sh, sw)

    def _adjacent_cube(self, seis, start):
        _, d, h, w = seis.shape
        limits = [d - self.crop_size, h - self.crop_size, w - self.crop_size]
        axis = random.randint(0, 2)
        overlap = random.randint(max(1, int(self.crop_size * self.min_overlap)), max(1, int(self.crop_size * self.max_overlap)))
        shift = self.crop_size - overlap
        starts = list(start)
        direction = 1 if random.random() < 0.5 else -1
        new_start = starts[axis] + direction * shift
        if new_start < 0 or new_start > limits[axis]:
            direction *= -1
            new_start = starts[axis] + direction * shift
        if new_start < 0 or new_start > limits[axis]:
            new_start = starts[axis]
            direction = 0
            shift = 0
        starts_adj = starts.copy()
        starts_adj[axis] = new_start
        sd, sh, sw = starts_adj
        cube = seis[:, sd:sd+self.crop_size, sh:sh+self.crop_size, sw:sw+self.crop_size].contiguous()
        if direction >= 0:
            cur_start, cur_end = shift, self.crop_size
            adj_start, adj_end = 0, self.crop_size - shift
        else:
            cur_start, cur_end = 0, self.crop_size - shift
            adj_start, adj_end = shift, self.crop_size
        overlap_info = {
            'axis': torch.tensor(axis, dtype=torch.long),
            'cur_start': torch.tensor(cur_start, dtype=torch.long),
            'cur_end': torch.tensor(cur_end, dtype=torch.long),
            'adj_start': torch.tensor(adj_start, dtype=torch.long),
            'adj_end': torch.tensor(adj_end, dtype=torch.long),
        }
        return cube, overlap_info

    def __getitem__(self, idx):
        d = load_volume(random.choice(self.files))
        seis = to_seis_tensor(d['seis'])
        unsup_cube, _ = self._random_cube(seis)
        current_cube, start = self._random_cube(seis)
        adjacent_cube, overlap = self._adjacent_cube(seis, start)
        return {
            'unsup_cube': intensity_aug(unsup_cube).float().contiguous(),
            'current_cube': intensity_aug(current_cube).float().contiguous(),
            'adjacent_cube': intensity_aug(adjacent_cube).float().contiguous(),
            'overlap': overlap,
        }
