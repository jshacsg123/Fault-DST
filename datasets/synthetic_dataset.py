from __future__ import annotations
import random
import torch
from torch.utils.data import Dataset
from .augment import pair_geo, intensity_aug
from .common import random_crop
from utils.io_utils import list_files, load_volume, to_seis_tensor


class SyntheticDataset(Dataset):
    def __init__(self, root, crop_size, steps, batch_size, training=True):
        self.files = list_files(root)
        if not self.files:
            raise FileNotFoundError(root)
        self.crop_size = crop_size
        self.steps = steps
        self.batch_size = batch_size
        self.training = training

    def __len__(self):
        return max(1, self.steps * self.batch_size)

    def __getitem__(self, idx):
        path = random.choice(self.files) if self.training else self.files[idx % len(self.files)]
        d = load_volume(path)
        seis = to_seis_tensor(d['seis'])
        label = torch.from_numpy(d['fault'].astype('float32'))[None]
        mask = torch.ones_like(label)
        seis, label, mask = random_crop(seis, label, mask, self.crop_size)
        if self.training:
            seis, label, mask = pair_geo([seis, label, mask])
            seis = intensity_aug(seis)
        return {'seis': seis.float(), 'label': label.float(), 'mask': mask.float()}
