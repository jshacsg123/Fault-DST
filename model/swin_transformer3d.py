from __future__ import annotations
import math
from typing import List, Tuple
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size: int = 4, in_chans: int = 1, embed_dim: int = 24):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        b, c, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(b, c, d, h, w)
        return x


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]):
    b, d, h, w, c = x.shape
    wd, wh, ww = window_size
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd * wh * ww, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int, int], b: int, d: int, h: int, w: int, c: int):
    wd, wh, ww = window_size
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, c)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int, int], num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj_drop(self.proj(x))
        return x


def compute_mask_3d(d: int, h: int, w: int, window_size: Tuple[int, int, int], shift_size: Tuple[int, int, int], device):
    wd, wh, ww = window_size
    sd, sh, sw = shift_size
    dp = int(math.ceil(d / wd)) * wd
    hp = int(math.ceil(h / wh)) * wh
    wp = int(math.ceil(w / ww)) * ww
    img_mask = torch.zeros((1, dp, hp, wp, 1), device=device)
    cnt = 0
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)) if sd > 0 else (slice(0, dp),)
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)) if sh > 0 else (slice(0, hp),)
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)) if sw > 0 else (slice(0, wp),)
    for ds in d_slices:
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, ds, hs, ws, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 4, shift_size: int = 0,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0, drop_path_prob: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size, window_size)
        self.shift_size = (shift_size, shift_size, shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, self.window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor):
        b, d, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size
        dp = int(math.ceil(d / wd)) * wd
        hp = int(math.ceil(h / wh)) * wh
        wp = int(math.ceil(w / ww)) * ww
        pad_d, pad_h, pad_w = dp - d, hp - h, wp - w
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x.permute(0, 4, 1, 2, 3), (0, pad_w, 0, pad_h, 0, pad_d)).permute(0, 2, 3, 4, 1)
        attn_mask = None
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
            attn_mask = compute_mask_3d(d, h, w, self.window_size, self.shift_size, x.device)
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, b, dp, hp, wp, c)
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(sd, sh, sw), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :d, :h, :w, :].contiguous()
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 8)
        self.reduction = nn.Linear(dim * 8, dim * 2, bias=False)

    def forward(self, x: torch.Tensor):
        b, d, h, w, c = x.shape
        pad_input = (d % 2 == 1) or (h % 2 == 1) or (w % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 4, 1, 2, 3), (0, w % 2, 0, h % 2, 0, d % 2)).permute(0, 2, 3, 4, 1)
            d, h, w = x.shape[1:4]
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = self.reduction(self.norm(x))
        return x


class BasicLayer3D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float,
                 drop: float, attn_drop: float, drop_path_probs: List[float], downsample: bool, use_checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            self.blocks.append(
                SwinTransformerBlock3D(dim, num_heads, window_size, shift, mlp_ratio, drop, attn_drop, drop_path_probs[i])
            )
        self.downsample = PatchMerging3D(dim) if downsample else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return skip, x


class ConvRefine(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=False),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class SwinTransformer3DSeg(nn.Module):
    """Standard 3D Swin Transformer encoder with a UNet-style decoder."""
    def __init__(self, patch_size: int = 4, embed_dim: int = 24, depths: List[int] | Tuple[int, ...] = (1, 1, 2, 1),
                 num_heads: List[int] | Tuple[int, ...] = (2, 4, 8, 16), window_size: int = 4, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.1, use_checkpoint: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.layers = nn.ModuleList()
        for i in range(4):
            layer = BasicLayer3D(
                dim=dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_probs=dpr[cur:cur + depths[i]],
                downsample=(i < 3),
                use_checkpoint=use_checkpoint,
            )
            cur += depths[i]
            self.layers.append(layer)

        self.up3 = nn.ConvTranspose3d(dims[3], dims[2], kernel_size=2, stride=2)
        self.dec3 = ConvRefine(dims[2] + dims[2], dims[2])
        self.up2 = nn.ConvTranspose3d(dims[2], dims[1], kernel_size=2, stride=2)
        self.dec2 = ConvRefine(dims[1] + dims[1], dims[1])
        self.up1 = nn.ConvTranspose3d(dims[1], dims[0], kernel_size=2, stride=2)
        self.dec1 = ConvRefine(dims[0] + dims[0], dims[0])
        self.final_up = nn.ConvTranspose3d(dims[0], dims[0], kernel_size=patch_size, stride=patch_size)
        self.final_refine = ConvRefine(dims[0], dims[0])
        self.seg_head = nn.Conv3d(dims[0], 1, 1, bias=False)

    def _to_cf(self, x: torch.Tensor):
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x: torch.Tensor):
        if any(s % self.patch_size != 0 for s in x.shape[-3:]):
            raise ValueError(f'Input spatial size {tuple(x.shape[-3:])} must be divisible by patch_size={self.patch_size}')
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        skips = []
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(self._to_cf(skip))
        s1, s2, s3, s4 = skips

        x = s4
        x = self.up3(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))
        feat = self.final_refine(self.final_up(x))
        seg = self.seg_head(feat)
        return feat, seg
