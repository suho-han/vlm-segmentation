"""
Abstract base class for ViT-based 2D segmentation models (Phase B).

Design goals:
  - Uniform forward signature compatible with VLM injection infrastructure:
      forward(x, text_embed=None, image_vlm_feat=None)
  - Standardised encoder/decoder interface so concrete sub-classes only need
    to implement _build_encoder / _encode / _decode.
  - Multi-scale VLM injection hooks: _apply_vlm(feat, level, text_embed, image_vlm_feat)
    defaults to identity; subclasses override to activate injection.
  - Multi-scale projection interface: _project_vlm(image_vlm_feat, feat_size)
    bilinearly upsamples the (B, D, 14, 14) VLM patch grid to match the
    decoder feature map, then applies a 1×1 Conv projection.

Intended usage:
  1. Subclass ViTSegBase.
  2. Implement _build_encoder() — build & store encoder backbone.
  3. Implement _encode(x) → list[Tensor] of multi-scale features.
  4. Implement _decode(features) → Tensor (B, out_ch, H, W).
  5. Optionally override _apply_vlm() for VLM injection.

Architecture contract (for concrete sub-classes):
  _encode(x) must return a list ordered from **coarsest to finest**:
    features[0] = coarsest (bottleneck / deepest)
    features[-1] = finest  (shallowest / closest to input resolution)
  _decode(features) consumes this list and returns the logit map.

Multi-scale projection helpers:
  _project_vlm(image_vlm_feat, feat_h, feat_w) → projected VLM features.
  If self.vlm_proj[level] is set (a Conv2d), projects vlm_dim → feat_dim.

VLM injection points:
  _apply_vlm is called once per decoder level during forward().
  `level` 0 = coarsest decoder stage, increasing toward finest.
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTSegBase(nn.Module, abc.ABC):
    """
    Abstract base for ViT-backed 2D segmentation models with VLM injection.

    Args:
        in_channels:  Input image channels.
        out_channels: Output channels (1 for binary segmentation).
        img_size:     Input spatial resolution (H == W assumed).
        vlm_dim:      Dimension of VLM image patch features (default 512).
        text_dim:     Dimension of VLM text embeddings (default 512).
    """

    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 1,
        img_size:     int = 384,
        vlm_dim:      int = 512,
        text_dim:     int = 512,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.img_size     = img_size
        self.vlm_dim      = vlm_dim
        self.text_dim     = text_dim

        # VLM 1×1 projection layers; populated by subclasses via _register_vlm_projs().
        # vlm_proj[i]: Conv2d(vlm_dim → decoder_feat_dim[i])  or None
        self.vlm_proj: nn.ModuleList = nn.ModuleList()

        self._build_encoder()

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by concrete subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_encoder(self) -> None:
        """Build and register the encoder backbone as self.encoder (or custom attrs)."""

    @abc.abstractmethod
    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run the encoder on input x.

        Returns:
            List of multi-scale feature tensors, **coarsest-first**.
            features[0] = bottleneck / deepest representation.
            features[-1] = finest / shallowest.
        """

    @abc.abstractmethod
    def _decode(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Run the decoder on multi-scale features.

        Args:
            features: Coarsest-first list from _encode().
        Returns:
            (B, out_channels, H, W) logit map.
        """

    # ------------------------------------------------------------------
    # VLM hook — override to activate injection
    # ------------------------------------------------------------------

    def _apply_vlm(
        self,
        feat:           torch.Tensor,
        level:          int,
        text_embed:     Optional[torch.Tensor],
        image_vlm_feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Optional VLM modulation at a decoder level.

        Default: identity pass-through.

        Args:
            feat:           (B, C, H, W) decoder feature at `level`.
            level:          Decoder level index (0 = coarsest, N-1 = finest).
            text_embed:     (1, text_dim) or None.
            image_vlm_feat: (B, vlm_dim, 14, 14) or None.
        Returns:
            (B, C, H, W) — same shape as feat.
        """
        return feat

    # ------------------------------------------------------------------
    # Multi-scale VLM projection helper
    # ------------------------------------------------------------------

    def _project_vlm(
        self,
        image_vlm_feat: torch.Tensor,
        feat_h:         int,
        feat_w:         int,
        level:          int,
    ) -> torch.Tensor:
        """
        Upsample VLM patch features to (feat_h, feat_w) and project channels.

        If self.vlm_proj[level] exists, applies the 1×1 Conv projection.
        Otherwise returns the bilinearly upsampled features unchanged.

        Args:
            image_vlm_feat: (B, vlm_dim, 14, 14)
            feat_h, feat_w: Target spatial size.
            level:          Which projection layer to use.
        Returns:
            (B, C, feat_h, feat_w) where C = projected dim or vlm_dim.
        """
        upsampled = F.interpolate(
            image_vlm_feat, size=(feat_h, feat_w),
            mode="bilinear", align_corners=False,
        )
        if level < len(self.vlm_proj) and self.vlm_proj[level] is not None:
            upsampled = self.vlm_proj[level](upsampled)
        return upsampled

    # ------------------------------------------------------------------
    # Registration helper for subclasses
    # ------------------------------------------------------------------

    def _register_vlm_projs(self, feat_dims: list[int]) -> None:
        """
        Build and register vlm_proj[i]: Conv2d(vlm_dim → feat_dims[i], 1×1).
        Zero-initialised so injection is identity at training start.

        Call this from the subclass constructor after building the decoder.

        Args:
            feat_dims: Per-level decoder feature dimensions (coarsest-first).
        """
        projs = nn.ModuleList()
        for dim in feat_dims:
            conv = nn.Conv2d(self.vlm_dim, dim, kernel_size=1, bias=False)
            nn.init.zeros_(conv.weight)
            projs.append(conv)
        self.vlm_proj = projs

    # ------------------------------------------------------------------
    # Forward — calls _encode → level-wise _decode + _apply_vlm
    # ------------------------------------------------------------------

    def forward(
        self,
        x:              torch.Tensor,
        text_embed:     Optional[torch.Tensor] = None,
        image_vlm_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, in_channels, H, W).
            text_embed:     (1, text_dim) or None.
            image_vlm_feat: (B, vlm_dim, 14, 14) or None.
        Returns:
            (B, out_channels, H, W) logit map.

        Note:
            Subclasses that need level-wise VLM injection should override
            _decode() to call self._apply_vlm() at each decoder stage,
            or implement a custom forward().
        """
        features = self._encode(x)
        logits   = self._decode(features)
        # Top-level VLM hook at final output (level = len(features))
        logits = self._apply_vlm(logits, len(features), text_embed, image_vlm_feat)
        return logits
