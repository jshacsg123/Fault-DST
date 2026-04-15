from __future__ import annotations
import random
import torch
from torch.utils.data import Dataset
from .augment import pair_geo, intensity_aug
from .common import random_crop
from utils.io_utils import list_files, load_volume, to_seis_tensor, infer_label_and_mask


class FieldSparseDataset(Dataset):
    def __init__(self, root, crop_size, steps, batch_size, ignore_label=-1.0, training=True):
        self.files = list_files(root)
        if not self.files:
            raise FileNotFoundError(root)
        self.crop_size = crop_size
        self.steps = steps
        self.batch_size = batch_size
        self.ignore_label = ignore_label
        self.training = training

    def __len__(self):
        return max(1, self.steps * self.batch_size)

    def __getitem__(self, idx):
        path = random.choice(self.files) if self.training else self.files[idx % len(self.files)]
        d = load_volume(path)
        seis = to_seis_tensor(d['seis'])
        if 'mask' in d:
            label = torch.from_numpy(d['fault'].astype('float32'))[None]
            mask = torch.from_numpy(d['mask'].astype('float32'))[None]
            label = torch.where(mask > 0, label, torch.zeros_like(label))
        else:
            lab, mask_np = infer_label_and_mask(d['fault'], self.ignore_label)
            label = torch.from_numpy(lab)[None]
            mask = torch.from_numpy(mask_np)[None]
        focus = mask if torch.any(mask > 0) else None
        seis, label, mask = random_crop(seis, label, mask, self.crop_size, focus)
        if self.training:
            seis, label, mask = pair_geo([seis, label, mask])
            seis = intensity_aug(seis)
        return {'seis': seis.float(), 'label': label.float(), 'mask': (mask > 0.5).float()}
