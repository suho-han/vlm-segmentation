"""
TransUNet 2D segmentation model (Backbone Expansion B4).

Architecture
============
Encoder: timm ViT-B/16 (`vit_base_patch16_384`, dynamic_img_size=True)
  - get_intermediate_layers(n=[2,5,8,11], reshape=True) → 4 feature maps
    all at the same (B, 768, H/16, W/16) spatial resolution
  - Uses the last block's output as the bottleneck

Decoder: 4-stage CNN upsampler
  - Each stage: ConvTranspose2d(stride=2) → Conv3×3+BN+ReLU → Conv3×3+BN+ReLU
  - Channel schedule: 768 → dec_chs[0] → dec_chs[1] → dec_chs[2] → dec_chs[3]
  - Default dec_chs = [256, 128, 64, 16]
  - After 4 upsampling stages the spatial resolution doubles 4× from H/16 to H

VLM injection
=============
At each of the 4 decoder stages, _apply_vlm adds zero-init projected
BiomedCLIP spatial features:

    out = block(x) + alpha * Conv2d(512→dec_ch)(upsample(vlm_14×14))

where alpha comes from ViTSegBase's _project_vlm (Conv2d is zero-init).

IMPORTANT: subclass attributes must be stored BEFORE super().__init__()
because ViTSegBase.__init__ immediately calls self._build_encoder().
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_template import ViTSegBase


class TransUNetSeg(ViTSegBase):
    """
    TransUNet 2D segmentation model with optional VLM spatial injection.

    Subclasses ViTSegBase; overrides _build_encoder, _encode, _decode,
    _apply_vlm, and forward.

    Args:
        in_channels:  Input image channels (default 1).
        out_channels: Output segmentation channels (default 1).
        img_size:     Input spatial resolution assumed square (default 384).
        timm_model:   timm model name for the ViT backbone (default "vit_base_patch16_384").
        pretrained:   Load ImageNet pretrained ViT weights (default False).
        decoder_chs:  Output channel dims for 4 decoder stages (default [256,128,64,16]).
        vlm_dim:      VLM image feature channels (default 512).
        text_dim:     VLM text embedding dim (default 512, unused currently).
    """

    def __init__(
        self,
        in_channels:  int        = 1,
        out_channels: int        = 1,
        img_size:     int        = 384,
        timm_model:   str        = "vit_base_patch16_384",
        pretrained:   bool       = False,
        decoder_chs:  list[int]  = None,
        vlm_dim:      int        = 512,
        text_dim:     int        = 512,
    ) -> None:
        # Store subclass attrs BEFORE super().__init__() because
        # ViTSegBase.__init__ immediately calls self._build_encoder().
        self.pretrained   = pretrained
        self._timm_model  = timm_model
        self._dec_chs     = decoder_chs if decoder_chs is not None else [256, 128, 64, 16]

        # Triggers _build_encoder() — encoder/decoder layers are registered here.
        super().__init__(in_channels, out_channels, img_size, vlm_dim, text_dim)

        # Cached VLM inputs set by forward() for use inside _decode().
        self._cached_text:    Optional[torch.Tensor] = None
        self._cached_img_vlm: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # ViTSegBase abstract interface
    # ------------------------------------------------------------------

    def _build_encoder(self) -> None:
        """Build ViT encoder + CNN decoder blocks."""
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required for TransUNetSeg. "
                              "Install with: uv pip install timm") from e

        # ViT encoder: dynamic_img_size allows 512px (DRIVE) with pos embed interpolation
        self.vit = timm.create_model(
            self._timm_model,
            pretrained=self.pretrained,
            num_classes=0,
            in_chans=self.in_channels,
            dynamic_img_size=True,
        )
        vit_dim = 768  # ViT-B hidden dim

        # CNN decoder: 4 upsampling stages doubling H,W each step
        self.decoder_blocks = nn.ModuleList()
        in_ch = vit_dim
        for out_ch in self._dec_chs:
            block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.decoder_blocks.append(block)
            in_ch = out_ch

        self.seg_head = nn.Conv2d(self._dec_chs[-1], self.out_channels, kernel_size=1)

        # VLM projection layers: Conv2d(vlm_dim → dec_ch) per stage, zero-init
        self._register_vlm_projs(self._dec_chs)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run ViT encoder; return bottleneck as single-element coarsest-first list.

        Returns:
            [bottleneck]: (B, 768, H/16, W/16)
        """
        # get_intermediate_layers with reshape=True returns (B, C, H', W') tensors
        inter = self.vit.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True)
        # Use the last block output as the sole bottleneck feature
        return [inter[-1]]  # [(B, 768, H/16, W/16)]

    def _decode(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        CNN decoder with optional VLM injection at each stage.

        Args:
            features: [bottleneck] — (B, 768, H/16, W/16).
        Returns:
            (B, out_channels, H, W) logit map.
        """
        x = features[0]  # (B, 768, H/16, W/16)

        for i, block in enumerate(self.decoder_blocks):
            x = block(x)   # ConvTranspose2d doubles spatial resolution
            x = self._apply_vlm(x, i, self._cached_text, self._cached_img_vlm)

        # Final upsample to exact input size in case of rounding differences
        if (x.shape[2], x.shape[3]) != self._input_hw:
            x = F.interpolate(x, size=self._input_hw,
                              mode="bilinear", align_corners=False)

        return self.seg_head(x)

    # ------------------------------------------------------------------
    # VLM injection hook
    # ------------------------------------------------------------------

    def _apply_vlm(
        self,
        feat:           torch.Tensor,
        level:          int,
        text_embed:     Optional[torch.Tensor],
        image_vlm_feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Add zero-init projected VLM spatial features to each decoder stage.

        No-op when image_vlm_feat is None.
        """
        if image_vlm_feat is None:
            return feat
        h, w = feat.shape[2], feat.shape[3]
        return feat + self._project_vlm(image_vlm_feat, h, w, level)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:              torch.Tensor,
        text_embed:     Optional[torch.Tensor] = None,
        image_vlm_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, in_channels, H, W) input image.
            text_embed:     (1, text_dim) or None — not used in current decoder.
            image_vlm_feat: (B, vlm_dim, 14, 14) or None — spatial VLM prior.
        Returns:
            (B, out_channels, H, W) logit map.
        """
        # Cache VLM inputs for use inside _decode → _apply_vlm
        self._cached_text    = text_embed
        self._cached_img_vlm = image_vlm_feat
        # Cache input size for final size-check in _decode
        self._input_hw       = (x.shape[2], x.shape[3])
        return self._decode(self._encode(x))
