"""Building blocks used by the MFSR reconstruction module.

MFSR combines the Residual Channel Attention Network (RCAN) reconstruction
blocks with the up-projection block from Deep Back-Projection Networks
(D-DBPN).  The upstream repository imported these symbols from ``common`` but
did not publish that file.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


ConvFactory = Callable[..., nn.Conv2d]

__all__ = [
    "CALayer",
    "DenseProjection",
    "RCAB",
    "ResidualGroup",
    "default_conv",
    "projection_conv",
]


def default_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = True,
) -> nn.Conv2d:
    """Create a same-resolution convolution used by RCAN blocks."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class CALayer(nn.Module):
    """Channel-attention layer from RCAN."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if reduction < 1 or channels // reduction < 1:
            raise ValueError("reduction must leave at least one hidden channel")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv_du(self.avg_pool(x))


class RCAB(nn.Module):
    """Residual channel-attention block used in each MFSR residual group."""

    def __init__(
        self,
        conv: ConvFactory,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        bias: bool = True,
        bn: bool = False,
        act: nn.Module | None = None,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if act is None:
            act = nn.ReLU(inplace=True)

        body = []
        for index in range(2):
            body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                body.append(nn.BatchNorm2d(n_feat))
            if index == 0:
                body.append(act)
        body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x).mul(self.res_scale)


class ResidualGroup(nn.Module):
    """Eight RCABs plus a group-level short skip connection by default."""

    def __init__(
        self,
        conv: ConvFactory,
        n_feat: int,
        kernel_size: int,
        reduction: int = 16,
        act: nn.Module | None = None,
        res_scale: float = 1.0,
        n_resblocks: int = 8,
    ) -> None:
        super().__init__()
        if n_resblocks < 1:
            raise ValueError("n_resblocks must be positive")
        if act is None:
            act = nn.ReLU(inplace=True)

        body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=act,
                res_scale=res_scale,
            )
            for _ in range(n_resblocks)
        ]
        body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def projection_conv(
    in_channels: int,
    out_channels: int,
    scale: int,
    up: bool = True,
) -> nn.Module:
    """Projection convolution from D-DBPN for scales 2, 4 and 8."""
    try:
        kernel_size, stride, padding = {
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2),
        }[scale]
    except KeyError as error:
        raise ValueError("DenseProjection supports scale 2, 4 or 8") from error

    conv_type = nn.ConvTranspose2d if up else nn.Conv2d
    return conv_type(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
    )


class DenseProjection(nn.Module):
    """Error-feedback up/down-projection block from D-DBPN."""

    def __init__(
        self,
        in_channels: int,
        nr: int,
        scale: int,
        up: bool = True,
        bottleneck: bool = True,
    ) -> None:
        super().__init__()
        if bottleneck:
            self.bottleneck: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr),
            )
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr),
        )
        self.conv_2 = nn.Sequential(
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels),
        )
        self.conv_3 = nn.Sequential(
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        projected = self.conv_1(x)
        reconstructed = self.conv_2(projected)
        error = reconstructed.sub(x)
        correction = self.conv_3(error)
        return projected.add(correction)
