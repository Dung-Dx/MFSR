"""Histopathology RGB adaptation of the original tri-modal MFSR model.

Only the low-resolution RGB image is used as source information. It is
bicubically enlarged and converted into the three input views expected by the
published MFSR encoder:

* a one-channel luminance view for the first branch;
* the RGB view for ``OPT_resnet``;
* a one-channel luminance view for the depth branch.

The 224x224 ground-truth image is never passed to the model. The decoder
predicts a three-channel residual which is added to the bicubic RGB baseline.
This makes ``encoder.OPT_resnet`` suitable for later transfer to the RGB
ResNet101 backbone of WSSS-Tissue Stage 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from MFSR_DEM_Depth import EncoderBlock, Super_resolution


class HistologyMFSR(nn.Module):
    """MFSR adapted for paired RGB super-resolution at x2 or x4."""

    def __init__(
        self,
        scale: int,
        pretrained: bool = True,
        backbone: str = "ResNet101",
        num_feats: int = 32,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("HistologyMFSR supports scale=2 or scale=4")

        self.scale = scale
        self.encoder = EncoderBlock(
            pretrained=pretrained,
            backbone=backbone,
            att_type=None,
        )
        # EncoderBlock's low-level feature map is always 1/4 of the enlarged
        # image. Therefore reconstruction projection is x4 for both LR x2 and
        # LR x4 training tasks.
        self.decoder = Super_resolution(
            num_feats=num_feats,
            kernel_size=kernel_size,
            scale=4,
            output_channels=3,
        )
        # Start from the stable bicubic baseline. A randomly initialized RGB
        # residual head can otherwise move the output sharply at the first
        # Adam update and destabilize the large multiplicative fusion module.
        output_conv = self.decoder.last_conv[-1]
        nn.init.zeros_(output_conv.weight)
        if output_conv.bias is not None:
            nn.init.zeros_(output_conv.bias)
        self.register_buffer(
            "rgb_to_luminance",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, low_resolution_rgb: torch.Tensor) -> torch.Tensor:
        if low_resolution_rgb.ndim != 4:
            raise ValueError("low_resolution_rgb must be a 4-D NCHW tensor")
        if low_resolution_rgb.shape[1] != 3:
            raise ValueError("low_resolution_rgb must have three RGB channels")

        target_height = low_resolution_rgb.shape[-2] * self.scale
        target_width = low_resolution_rgb.shape[-1] * self.scale
        if target_height % 4 or target_width % 4:
            raise ValueError("The enlarged spatial dimensions must be divisible by 4")

        bicubic_rgb = F.interpolate(
            low_resolution_rgb,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        luminance = (bicubic_rgb * self.rgb_to_luminance).sum(
            dim=1, keepdim=True
        )
        fused_features = self.encoder(luminance, bicubic_rgb, luminance)
        residual = self.decoder(fused_features)
        if residual.shape != bicubic_rgb.shape:
            raise RuntimeError(
                "Decoder returned an unexpected shape: "
                f"{tuple(residual.shape)} instead of {tuple(bicubic_rgb.shape)}"
            )
        return residual + bicubic_rgb

    def wsss_backbone_state_dict(self):
        """Return the directly transferable RGB ResNet101 state dictionary."""
        return self.encoder.OPT_resnet.state_dict()
