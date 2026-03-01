"""
SegFormer-B2 2D segmentation model (Backbone Expansion B3).

Architecture
============
Encoder: PVT-v2-B2 (timm `pvt_v2_b2`, features_only=True)
  - Stage 1: (B,  64, H/4,  W/4)
  - Stage 2: (B, 128, H/8,  W/8)
  - Stage 3: (B, 320, H/16, W/16)
  - Stage 4: (B, 512, H/32, W/32)  ← closest to VLM 14×14

Decoder: All-MLP (SegFormer-style)
  - Project each stage to `decoder_dim` (default 256) via 1×1 Conv
  - Upsample all to finest stage (H/4) via bilinear interpolation
  - Concat (4 × decoder_dim) → 1×1 Conv + BN + ReLU → Dropout2d
  - Final upsample ×4 to full resolution → 1×1 Conv head

VLM injection
=============
After each per-stage projection (step 1 of decoder), `_apply_vlm` adds
zero-init spatial features from BiomedCLIP:

    out = proj_feat + alpha * Conv2d(512→decoder_dim)(upsample(vlm_14x14))

Zero-init projection weight + ViTSegBase's `_project_vlm` helper →
identity at init; injection activates as proj.weight learns.

IMPORTANT: subclass attributes must be stored BEFORE super().__init__()
because ViTSegBase.__init__ immediately calls self._build_encoder().
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_template import ViTSegBase


class SegFormerB2Seg(ViTSegBase):
    """
    SegFormer-B2 2D segmentation model with optional VLM spatial injection.

    Subclasses ViTSegBase; overrides _build_encoder, _encode, _decode,
    _apply_vlm, and forward.

    Args:
        in_channels:  Input image channels (default 1).
        out_channels: Output segmentation channels (default 1).
        img_size:     Input spatial resolution assumed square (default 384).
        pretrained:   Load ImageNet pretrained PVT-v2-B2 weights (default False).
        decoder_dim:  Unified channel dim in All-MLP decoder (default 256).
        dropout:      Dropout2d probability before seg head (default 0.1).
        vlm_dim:      VLM image feature channels (default 512).
        text_dim:     VLM text embedding dim (default 512, unused currently).
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        out_channels: int   = 1,
        img_size:     int   = 384,
        pretrained:   bool  = False,
        decoder_dim:  int   = 256,
        dropout:      float = 0.1,
        vlm_dim:      int   = 512,
        text_dim:     int   = 512,
    ) -> None:
        # Store subclass attrs BEFORE super().__init__() because
        # ViTSegBase.__init__ immediately calls self._build_encoder().
        self.pretrained   = pretrained
        self._decoder_dim = decoder_dim
        self._dropout_p   = dropout

        # Triggers _build_encoder() — encoder/decoder layers are registered here.
        super().__init__(in_channels, out_channels, img_size, vlm_dim, text_dim)

        # Cached VLM inputs set by forward() for use inside _decode().
        self._cached_text:    Optional[torch.Tensor] = None
        self._cached_img_vlm: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # ViTSegBase abstract interface
    # ------------------------------------------------------------------

    def _build_encoder(self) -> None:
        """Build PVT-v2-B2 encoder + All-MLP decoder layers."""
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required for SegFormerB2Seg. "
                              "Install with: uv pip install timm") from e

        # Encoder: PVT-v2-B2 with 4-scale feature output
        self.encoder = timm.create_model(
            "pvt_v2_b2",
            pretrained=self.pretrained,
            features_only=True,
            in_chans=self.in_channels,
        )
        # Channel dims from the encoder, finest to coarsest: [64, 128, 320, 512]
        # Reversed to coarsest-first for ViTSegBase convention:  [512, 320, 128, 64]
        self._enc_dims_fine2coarse = [64, 128, 320, 512]
        self._feat_dims = list(reversed(self._enc_dims_fine2coarse))  # [512, 320, 128, 64]

        dec_dim = self._decoder_dim

        # Per-stage 1×1 projection Conv: enc_dim → dec_dim (coarsest-first order)
        self.decode_projs = nn.ModuleList(
            [nn.Conv2d(d, dec_dim, kernel_size=1) for d in self._feat_dims]
        )

        # Fusion: (4 × dec_dim) → dec_dim
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(4 * dec_dim, dec_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout  = nn.Dropout2d(p=self._dropout_p)
        self.seg_head = nn.Conv2d(dec_dim, self.out_channels, kernel_size=1)

        # VLM projection layers: Conv2d(vlm_dim → dec_dim) per stage, zero-init
        self._register_vlm_projs([dec_dim] * 4)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run PVT-v2-B2 encoder.

        Returns:
            Coarsest-first list [stage4, stage3, stage2, stage1].
            - stage4: (B, 512, H/32, W/32)
            - stage3: (B, 320, H/16, W/16)
            - stage2: (B, 128, H/8,  W/8)
            - stage1: (B,  64, H/4,  W/4)
        """
        # timm features_only returns finest-first: [s1, s2, s3, s4]
        stage_feats = self.encoder(x)
        return list(reversed(stage_feats))  # coarsest-first

    def _decode(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        All-MLP decoder with optional VLM injection.

        Args:
            features: Coarsest-first [stage4, stage3, stage2, stage1].
        Returns:
            (B, out_channels, H, W) logit map.
        """
        # Upsample target = finest stage spatial size (H/4, W/4)
        target_hw = features[-1].shape[2:]

        projected: list[torch.Tensor] = []
        for i, feat in enumerate(features):
            p = self.decode_projs[i](feat)              # (B, dec_dim, H_i, W_i)
            p = self._apply_vlm(p, i,
                                self._cached_text,
                                self._cached_img_vlm)  # optional VLM add
            p = F.interpolate(p, size=target_hw,
                              mode="bilinear", align_corners=False)
            projected.append(p)

        fused = self.fuse_conv(torch.cat(projected, dim=1))  # (B, dec_dim, H/4, W/4)
        fused = self.dropout(fused)

        # Upsample to full input resolution
        out = F.interpolate(fused, size=self._input_hw,
                            mode="bilinear", align_corners=False)
        return self.seg_head(out)                            # (B, out_ch, H, W)

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
        Add zero-init projected VLM spatial features to decoder projection output.

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
        # Cache input size for final upsample in _decode
        self._input_hw       = (x.shape[2], x.shape[3])
        return self._decode(self._encode(x))
