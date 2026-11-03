"""Convolutional Block Attention Module (CBAM).

This implementation retains the module hierarchy and parameter names of the
authors' reference implementation so that compatible checkpoints can be
loaded without key remapping.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(
                out_planes, eps=1e-5, momentum=0.01, affine=True
            )
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1)


class ChannelGate(nn.Module):
    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: Sequence[str] = ("avg", "max"),
    ) -> None:
        super().__init__()
        hidden_channels = gate_channels // reduction_ratio
        if hidden_channels < 1:
            raise ValueError(
                "reduction_ratio must leave at least one hidden channel"
            )
        unsupported = set(pool_types).difference({"avg", "max", "lp", "lse"})
        if unsupported:
            raise ValueError(f"Unsupported pooling types: {sorted(unsupported)}")

        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, gate_channels),
        )
        self.pool_types = tuple(pool_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_att_sum = None
        output_size = 1
        for pool_type in self.pool_types:
            if pool_type == "avg":
                pooled = F.adaptive_avg_pool2d(x, output_size)
            elif pool_type == "max":
                pooled = F.adaptive_max_pool2d(x, output_size)
            elif pool_type == "lp":
                pooled = F.lp_pool2d(x, 2, x.shape[-2:])
            else:
                pooled = logsumexp_2d(x)
            attention = self.mlp(pooled)
            channel_att_sum = (
                attention
                if channel_att_sum is None
                else channel_att_sum + attention
            )

        if channel_att_sum is None:
            raise ValueError("pool_types cannot be empty")
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return x * scale


def logsumexp_2d(tensor: torch.Tensor) -> torch.Tensor:
    flattened = tensor.flatten(start_dim=2)
    maximum = flattened.max(dim=2, keepdim=True).values
    return maximum + (flattened - maximum).exp().sum(dim=2, keepdim=True).log()


class ChannelPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maximum = x.max(dim=1, keepdim=True).values
        average = x.mean(dim=1, keepdim=True)
        return torch.cat((maximum, average), dim=1)


class SpatialGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2,
            1,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.spatial(self.compress(x)))
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels: int,
        reduction_ratio: int = 16,
        pool_types: Sequence[str] = ("avg", "max"),
        no_spatial: bool = False,
    ) -> None:
        super().__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ChannelGate(x)
        if not self.no_spatial:
            x = self.SpatialGate(x)
        return x
