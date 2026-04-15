from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(c, affine=False)
        self.conv2 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(c, affine=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(x + y)


class HRNet(nn.Module):
    """Compact 3D HRNet-style CNN branch for Student1."""
    def __init__(self, base: int = 8):
        super().__init__()
        c = base
        self.stem = nn.Sequential(
            nn.Conv3d(1, c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(c),
        )
        self.high = nn.Sequential(ResBlock(c), ResBlock(c))
        self.down = nn.Sequential(
            nn.Conv3d(c, 2 * c, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(2 * c, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.low = nn.Sequential(ResBlock(2 * c), ResBlock(2 * c))
        self.fuse = nn.Sequential(
            nn.Conv3d(3 * c, 4 * c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(4 * c, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(4 * c),
        )
        self.seg_head = nn.Sequential(
            nn.Conv3d(4 * c, 2 * c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(2 * c, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(2 * c, c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor):
        high = self.high(self.stem(x))
        low = self.low(self.down(high))
        low_up = F.interpolate(low, size=high.shape[-3:], mode='trilinear', align_corners=True)
        feat = self.fuse(torch.cat([high, low_up], dim=1))
        seg = self.seg_head(feat)
        return feat, seg
