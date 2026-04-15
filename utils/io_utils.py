from __future__ import annotations
import os
import numpy as np
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_files(root: str):
    if not root or not os.path.isdir(root):
        return []
    return [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.lower().endswith(('.npz', '.npy'))]


def load_volume(path: str):
    if path.lower().endswith('.npz'):
        data = np.load(path, allow_pickle=False)
        out = {'seis': data['seis']}
        if 'fault' in data:
            out['fault'] = data['fault']
        if 'mask' in data:
            out['mask'] = data['mask']
        return out
    if path.lower().endswith('.npy'):
        return {'seis': np.load(path)}
    raise ValueError(f'Unsupported file: {path}')


def normalize_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    std = float(np.std(x))
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - float(np.mean(x))) / std
    x = np.clip(x, -3.2, 3.2)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def to_seis_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(normalize_np(arr))[None]


def infer_label_and_mask(fault: np.ndarray, ignore_label: float):
    fault = fault.astype(np.float32, copy=False)
    mask = (fault != ignore_label).astype(np.float32)
    label = np.where(mask > 0, fault, 0).astype(np.float32)
    return label, mask
