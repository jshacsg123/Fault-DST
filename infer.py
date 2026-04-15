from __future__ import annotations
import argparse
import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model import DualStudentTeacherModel
from utils.io_utils import load_volume, normalize_np

try:
    from cigvis import create_slices, add_mask, plot3D  # type: ignore
    _HAS_CIGVIS = True
except Exception:
    _HAS_CIGVIS = False


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def parse_optional_int_list(value: str | None) -> Optional[List[int]]:
    if value is None or value == '':
        return None
    return parse_int_list(value)


def parse_json_or_none(value: str | None):
    if value is None or value == '':
        return None
    return json.loads(value)


class ArgsProxy:
    def __init__(self, a):
        self.cnn_base_width = a.cnn_base_width
        self.swin_patch_size = a.swin_patch_size
        self.swin_embed_dim = a.swin_embed_dim
        self.swin_depths = parse_int_list(a.swin_depths)
        self.swin_num_heads = parse_int_list(a.swin_num_heads)
        self.swin_window_size = a.swin_window_size
        self.swin_mlp_ratio = a.swin_mlp_ratio
        self.swin_drop_rate = a.swin_drop_rate
        self.swin_attn_drop_rate = a.swin_attn_drop_rate
        self.swin_drop_path_rate = a.swin_drop_path_rate
        self.swin_use_checkpoint = False
        self.smw_alpha = 1.0
        self.smw_beta = 1.5
        self.hard_conf_threshold = 0.6
        self.soft_temperature = 0.1
        self.lambda_kl = 0.3
        self.alpha_h = 0.5
        self.beta_s = 0.5
        self.gamma_dcca = 0.5
        self.mu_ic = 1.0
        self.field_sup_weight = 1.0
        self.consistency_downsample = 1


class InferEngine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        self.model = DualStudentTeacherModel(ArgsProxy(args)).to(self.device)
        self.patch_size = int(args.swin_patch_size)
        self.crop_size = int(args.infer_size)
        self.halo = int(args.padding_halo)
        self.branch = args.branch
        self.amp_enabled = bool(args.amp and self.device.type == 'cuda')

    def load_weights(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state = ckpt['model']
        else:
            state = ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def forward_branch(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.amp.autocast(device_type='cuda', enabled=self.amp_enabled):
            if self.branch == 'all':
                return self.model.predict(x)
            if self.branch == 'cnn':
                return self.model._branch(self.model.student1, x)[1]
            if self.branch == 'swin':
                return self.model._branch(self.model.student2, x)[1]
            return self.model._branch(self.model.teacher, x)[1]

    def _safe_pad_volume(self, data: np.ndarray, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        d, h, w = data.shape
        td, th, tw = target_shape
        pad = (0, tw - w, 0, th - h, 0, td - d)
        tensor = torch.from_numpy(data)[None, None]
        pad_mode = 'reflect'
        if (tw - w) >= max(w, 1) or (th - h) >= max(h, 1) or (td - d) >= max(d, 1):
            pad_mode = 'replicate'
        return torch.nn.functional.pad(tensor, pad, mode=pad_mode)

    @staticmethod
    def _sliding_positions(length: int, window: int, stride: int) -> List[int]:
        if length <= window:
            return [0]
        pos = list(range(0, length - window + 1, stride))
        if pos[-1] != length - window:
            pos.append(length - window)
        return pos

    def infer_volume(self, infer_vol_dhw: np.ndarray):
        data = normalize_np(infer_vol_dhw.astype(np.float32, copy=False))
        d, h, w = data.shape
        model_stride = self.patch_size * (2 ** 3)
        if self.crop_size % model_stride != 0:
            raise ValueError(f'infer_size={self.crop_size} must be divisible by model stride {model_stride}')
        stride = self.crop_size if self.halo <= 0 else max(model_stride, self.crop_size - 2 * self.halo)
        target_shape = (max(d, self.crop_size), max(h, self.crop_size), max(w, self.crop_size))
        volume = self._safe_pad_volume(data, target_shape)
        td, th, tw = target_shape
        d_pos = self._sliding_positions(td, self.crop_size, stride)
        h_pos = self._sliding_positions(th, self.crop_size, stride)
        w_pos = self._sliding_positions(tw, self.crop_size, stride)

        if self.branch == 'all':
            accum = {k: np.zeros(target_shape, dtype=np.float32) for k in ('cnn', 'swin', 'teacher')}
            count = np.zeros(target_shape, dtype=np.float32)
        else:
            accum = np.zeros(target_shape, dtype=np.float32)
            count = np.zeros(target_shape, dtype=np.float32)

        for ds in d_pos:
            for hs in h_pos:
                for ws in w_pos:
                    patch = volume[:, :, ds:ds + self.crop_size, hs:hs + self.crop_size, ws:ws + self.crop_size]
                    patch = patch.to(self.device, non_blocking=True)
                    pred = self.forward_branch(patch)
                    if isinstance(pred, dict):
                        for key, value in pred.items():
                            accum[key][ds:ds + self.crop_size, hs:hs + self.crop_size, ws:ws + self.crop_size] += value[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                        count[ds:ds + self.crop_size, hs:hs + self.crop_size, ws:ws + self.crop_size] += 1.0
                    else:
                        accum[ds:ds + self.crop_size, hs:hs + self.crop_size, ws:ws + self.crop_size] += pred[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                        count[ds:ds + self.crop_size, hs:hs + self.crop_size, ws:ws + self.crop_size] += 1.0

        count = np.maximum(count, 1e-6)
        if isinstance(accum, dict):
            return {k: np.clip(v / count, 0.0, 1.0)[:d, :h, :w] for k, v in accum.items()}
        return np.clip(accum / count, 0.0, 1.0)[:d, :h, :w]


def load_volume_any(path: str, npz_key: Optional[str] = None) -> np.ndarray:
    if path.lower().endswith('.npz'):
        if npz_key is None:
            data = load_volume(path)
            raw = data['seis']
        else:
            z = np.load(path, allow_pickle=False)
            try:
                if npz_key not in z.files:
                    raise KeyError(f"NPZ key '{npz_key}' not found in {path}. Available keys: {z.files}")
                raw = z[npz_key]
            finally:
                z.close()
    elif path.lower().endswith('.npy'):
        raw = np.load(path)
    else:
        raise ValueError(f'Unsupported file type: {path}')
    raw = np.asarray(raw)
    if raw.ndim > 3:
        raw = np.squeeze(raw)
    if raw.ndim != 3:
        raise ValueError(f'Input volume must be 3D, got shape={raw.shape}')
    return raw.astype(np.float32, copy=False)


def normalization_display(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - mn) / (mx - mn + 1e-8)


def custom_cmap(base_cmap: str = 'jet', min_alpha: float = 0.6):
    cmap = plt.get_cmap(base_cmap)
    colors = cmap(np.arange(cmap.N))
    alphas = np.linspace(0, 1, cmap.N)
    colors[:, -1] = np.where(alphas <= min_alpha, 0, alphas)
    return mcolors.ListedColormap(colors)


def clip_slicepos_to_shape(slicepos, shape_xyz: Sequence[int]):
    xdim, ydim, zdim = shape_xyz

    def _fix(lst, hi):
        if lst is None:
            return []
        if isinstance(lst, (int, np.integer)):
            lst = [int(lst)]
        out, seen = [], set()
        for x in lst:
            xi = max(0, min(int(x), hi))
            if xi not in seen:
                seen.add(xi)
                out.append(xi)
        return out

    return [_fix(slicepos[0], xdim - 1), _fix(slicepos[1], ydim - 1), _fix(slicepos[2], zdim - 1)]


def to_sx_axes(result_dhw: np.ndarray, t_in, sx_t) -> np.ndarray:
    infer_axes = list(t_in) if t_in is not None else [0, 1, 2]
    sx_axes = list(sx_t) if sx_t is not None else [0, 1, 2]
    q = [infer_axes.index(ax) for ax in sx_axes]
    return result_dhw.transpose(tuple(q))


def save_prediction_npz(path: str, pred: np.ndarray | Dict[str, np.ndarray]):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if isinstance(pred, dict):
        np.savez_compressed(path, **pred)
    else:
        np.savez_compressed(path, prediction=pred)


def save_orthogonal_slices(vol_xyz: np.ndarray, pred_xyz: np.ndarray, slicepos, out_root: str, base_name: str,
                           thr: float = 0.5, gray_cmap_name: str = 'gray', heat_cmap_name: str = 'jet'):
    assert vol_xyz.ndim == pred_xyz.ndim == 3 and vol_xyz.shape == pred_xyz.shape
    gray_cmap = plt.get_cmap(gray_cmap_name)
    heat_cmap = plt.get_cmap(heat_cmap_name)

    def norm01(a):
        a = a.astype(np.float32, copy=False)
        mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        return (a - mn) / (mx - mn + 1e-8)

    axes_info = [('X', 0, slicepos[0]), ('Y', 1, slicepos[1]), ('Z', 2, slicepos[2])]
    for plane_name, axis, indices in axes_info:
        if not indices:
            continue
        for s in indices:
            s = int(s)
            base2d = np.take(vol_xyz, indices=s, axis=axis)
            pred2d = np.take(pred_xyz, indices=s, axis=axis)
            if plane_name in {'X', 'Y'}:
                base2d = base2d.T
                pred2d = pred2d.T
            base_rgba = gray_cmap(norm01(base2d))
            pred2d = np.clip(np.nan_to_num(pred2d.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            heat_rgba = heat_cmap(pred2d)
            out_rgba = np.where((pred2d > thr)[..., None], heat_rgba, base_rgba)
            out_path = os.path.join(out_root, f'{base_name}_{plane_name}_{s:04d}.png')
            plt.imsave(out_path, out_rgba)


def maybe_save_3d_overlay(sx_xyz: np.ndarray, pred_xyz: np.ndarray, slicepos, output_path: str, dataset_cfg: Dict[str, Any]):
    if not _HAS_CIGVIS:
        warnings.warn('cigvis/vispy not installed, skip 3D overlay export.')
        return
    nodes = create_slices(sx_xyz, pos=slicepos, cmap='gray')
    nodes = add_mask(nodes, pred_xyz, cmaps=custom_cmap('jet', min_alpha=0.6), interpolation='nearest')
    plot_kwargs = dict(
        center=dataset_cfg.get('center'),
        fov=50,
        azimuth=float(dataset_cfg.get('azimuth', 45.0)),
        elevation=float(dataset_cfg.get('elevation', 30.0)),
        zoom_factor=float(dataset_cfg.get('zoom', 1.0)),
        run_app=False,
        axis_scales=(1, 1, 1),
    )
    if dataset_cfg.get('scale_factor') is not None:
        plot_kwargs['scale_factor'] = float(dataset_cfg['scale_factor'])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot3D([nodes], ax=ax, savename=output_path, **plot_kwargs)
    ax.set_axis_off()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description='Inference for dual-student 3D fault detection project')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input', type=str, help='Single input volume (.npy or .npz)')
    mode.add_argument('--datasets_json', type=str, help='JSON file containing a list of dataset configs')

    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument('--checkpoint', type=str, help='Single checkpoint path')
    weights.add_argument('--weights_dir', type=str, help='Directory containing one or more .pth files')

    p.add_argument('--output', type=str, default='', help='Single-input NPZ output path')
    p.add_argument('--output_root', type=str, default='output/infer', help='Root output directory for multi-dataset mode')
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--branch', type=str, default='teacher', choices=['cnn', 'swin', 'teacher', 'all'])
    p.add_argument('--infer_size', type=int, default=128)
    p.add_argument('--padding_halo', type=int, default=8)

    p.add_argument('--npz_key', type=str, default='', help='Override NPZ key in single-input mode')
    p.add_argument('--t_in', type=str, default='', help='JSON list, e.g. [2,0,1], raw -> infer_vol(D,H,W)')
    p.add_argument('--sx_t', type=str, default='', help='JSON list, e.g. [2,1,0], raw -> display(X,Y,Z)')
    p.add_argument('--slicepos', type=str, default='', help='JSON list like [[32],[25],[105]]')
    p.add_argument('--slice_save', type=str, default='', help='JSON list like [[32],[25],[105]]')
    p.add_argument('--thr', type=float, default=0.5)
    p.add_argument('--save_npz', action='store_true')
    p.add_argument('--save_slices', action='store_true')
    p.add_argument('--save_3d', action='store_true')

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
    return p.parse_args()


def collect_weight_files(args) -> List[str]:
    if args.checkpoint:
        return [args.checkpoint]
    if not os.path.isdir(args.weights_dir):
        raise FileNotFoundError(args.weights_dir)
    files = [os.path.join(args.weights_dir, f) for f in sorted(os.listdir(args.weights_dir)) if f.lower().endswith('.pth')]
    if not files:
        raise FileNotFoundError(f'No .pth files found in {args.weights_dir}')
    return files


def build_single_dataset_cfg(args) -> Dict[str, Any]:
    return {
        'name': os.path.splitext(os.path.basename(args.input))[0],
        'path': args.input,
        'npz_key': args.npz_key or None,
        't_in': parse_json_or_none(args.t_in),
        'sx_t': parse_json_or_none(args.sx_t),
        'slicepos': parse_json_or_none(args.slicepos) or [[], [], []],
        'slice_save': parse_json_or_none(args.slice_save) or parse_json_or_none(args.slicepos) or [[], [], []],
        'thr': float(args.thr),
    }


def load_dataset_cfgs(args) -> List[Dict[str, Any]]:
    if args.input:
        return [build_single_dataset_cfg(args)]
    with open(args.datasets_json, 'r', encoding='utf-8') as f:
        cfgs = json.load(f)
    if not isinstance(cfgs, list) or not cfgs:
        raise ValueError('datasets_json must contain a non-empty list')
    return cfgs


def infer_one_dataset(engine: InferEngine, weight_name: str, ds_cfg: Dict[str, Any], args):
    name = ds_cfg.get('name') or os.path.splitext(os.path.basename(ds_cfg['path']))[0]
    raw = load_volume_any(ds_cfg['path'], ds_cfg.get('npz_key'))
    t_in = ds_cfg.get('t_in')
    sx_t = ds_cfg.get('sx_t')
    infer_vol = raw if t_in is None else raw.transpose(tuple(t_in))
    pred = engine.infer_volume(infer_vol)

    if args.input:
        if args.output:
            npz_path = args.output
        else:
            npz_path = os.path.join(args.output_root, f'{name}_{weight_name}_{engine.branch}.npz')
        if args.save_npz or True:
            save_prediction_npz(npz_path, pred)
        if args.save_slices or args.save_3d:
            sx = raw if sx_t is None else raw.transpose(tuple(sx_t))
            sx = normalization_display(sx)
            pred_show = pred[engine.branch] if isinstance(pred, dict) and engine.branch != 'all' else (pred['teacher'] if isinstance(pred, dict) else pred)
            pred_xyz = to_sx_axes(pred_show, t_in, sx_t)
            slicepos = clip_slicepos_to_shape(ds_cfg.get('slicepos', [[], [], []]), sx.shape)
            slice_save = clip_slicepos_to_shape(ds_cfg.get('slice_save', ds_cfg.get('slicepos', [[], [], []])), sx.shape)
            base_root = os.path.join(args.output_root, name)
            os.makedirs(base_root, exist_ok=True)
            if args.save_slices:
                for plane in ['X', 'Y', 'Z']:
                    os.makedirs(os.path.join(base_root, plane), exist_ok=True)
                save_orthogonal_slices(sx, pred_xyz, [slice_save[0], [], []], os.path.join(base_root, 'X'), f'{weight_name}_{name}', float(ds_cfg.get('thr', args.thr)))
                save_orthogonal_slices(sx, pred_xyz, [[], slice_save[1], []], os.path.join(base_root, 'Y'), f'{weight_name}_{name}', float(ds_cfg.get('thr', args.thr)))
                save_orthogonal_slices(sx, pred_xyz, [[], [], slice_save[2]], os.path.join(base_root, 'Z'), f'{weight_name}_{name}', float(ds_cfg.get('thr', args.thr)))
            if args.save_3d:
                os.makedirs(os.path.join(base_root, '3D'), exist_ok=True)
                maybe_save_3d_overlay(sx, pred_xyz, slicepos, os.path.join(base_root, '3D', f'{weight_name}_{name}_3D.png'), ds_cfg)
        print(f'Saved inference for {name} -> {npz_path}')
        return

    ds_root = os.path.join(args.output_root, name)
    os.makedirs(ds_root, exist_ok=True)
    for sub in ['NPZ', 'X', 'Y', 'Z', '3D']:
        os.makedirs(os.path.join(ds_root, sub), exist_ok=True)
    save_prediction_npz(os.path.join(ds_root, 'NPZ', f'{weight_name}_{name}_{engine.branch}.npz'), pred)

    sx = raw if sx_t is None else raw.transpose(tuple(sx_t))
    sx = normalization_display(sx)
    pred_show = pred[engine.branch] if isinstance(pred, dict) and engine.branch != 'all' else (pred['teacher'] if isinstance(pred, dict) else pred)
    pred_xyz = to_sx_axes(pred_show, t_in, sx_t)
    slicepos = clip_slicepos_to_shape(ds_cfg.get('slicepos', [[], [], []]), sx.shape)
    slice_save = clip_slicepos_to_shape(ds_cfg.get('slice_save', ds_cfg.get('slicepos', [[], [], []])), sx.shape)
    thr = float(ds_cfg.get('thr', args.thr))

    save_orthogonal_slices(sx, pred_xyz, [slice_save[0], [], []], os.path.join(ds_root, 'X'), f'{weight_name}_{name}', thr)
    save_orthogonal_slices(sx, pred_xyz, [[], slice_save[1], []], os.path.join(ds_root, 'Y'), f'{weight_name}_{name}', thr)
    save_orthogonal_slices(sx, pred_xyz, [[], [], slice_save[2]], os.path.join(ds_root, 'Z'), f'{weight_name}_{name}', thr)

    if args.save_3d:
        maybe_save_3d_overlay(sx, pred_xyz, slicepos, os.path.join(ds_root, '3D', f'{weight_name}_{name}_3D.png'), ds_cfg)

    print(f'[{name}] done with weight {weight_name}')


def main():
    args = parse_args()
    weight_files = collect_weight_files(args)
    dataset_cfgs = load_dataset_cfgs(args)

    engine = InferEngine(args)
    print(f'[Device] {engine.device}')
    print(f'[Branch] {engine.branch}')
    print(f'[Weights] {len(weight_files)} file(s)')

    for weight_path in weight_files:
        weight_name = os.path.splitext(os.path.basename(weight_path))[0]
        print(f'Loading weight: {weight_path}')
        engine.load_weights(weight_path)
        for ds_cfg in dataset_cfgs:
            infer_one_dataset(engine, weight_name, ds_cfg, args)

    print('Inference completed.')


if __name__ == '__main__':
    main()
