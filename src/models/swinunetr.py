"""
SwinUNETR-2D wrapper.

Uses monai.networks.nets.SwinUNETR with spatial_dims=2 (monai >= 1.4).
Input spatial dimensions MUST be divisible by 32 (2**5).
  OCTA500-6M: use image_size=384 (not 400)
  DRIVE:      use image_size=512

Dependency: monai >= 1.4 (documented in pyproject.toml)
"""

import torch
import torch.nn as nn


class SwinUNETR2D(nn.Module):
    """
    2-D SwinUNETR for binary segmentation.

    Wraps monai.networks.nets.SwinUNETR with spatial_dims=2.
    Output: logits tensor of shape (B, out_channels, H, W).

    Note: img_size must be divisible by 32 (= 2**5).
    """

    def __init__(
        self,
        img_size: int = 384,      # must be divisible by 32; 384 for OCTA, 512 for DRIVE
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as e:
            raise ImportError(
                "monai is required for SwinUNETR. "
                "Install with: uv pip install monai"
            ) from e

        # Validate divisibility
        if img_size % 32 != 0:
            raise ValueError(
                f"SwinUNETR2D requires img_size divisible by 32, got {img_size}. "
                f"Use {(img_size // 32) * 32} or {((img_size // 32) + 1) * 32}."
            )

        # monai >= 1.4 API: no img_size arg; works with arbitrary compliant sizes
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
