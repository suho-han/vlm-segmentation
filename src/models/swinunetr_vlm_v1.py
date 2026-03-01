"""
SwinUNETR-2D V1 — per-image spatial VLM feature injection.

Architecture
============
Extends V0 (text-gated channel modulation) with V1: per-image spatial
features from a frozen VLM image encoder (BiomedCLIP ViT-B/16) are
upsampled to each decoder stage's spatial resolution and fused via a
small residual add with a learnable alpha scalar.

At each decoder stage (dec3 … dec0):

    vlm_up  = bilinear_upsample(image_vlm_feat, size=decoder_feat.shape[2:])
    proj    = Conv2d(vlm_dim → feat_dim, 1×1)(vlm_up)
    out     = decoder_feat + alpha * proj

where:
    - image_vlm_feat: (B, 512, 14, 14)  — frozen BiomedCLIP patch features
    - alpha:          nn.Parameter       — learnable per-stage scalar
    - proj.weight:    zero-initialised   — identity at init (alpha * 0 = 0)

vlm_mode
--------
    "image"  (default) — only spatial image injection (V1 design)
    "text"             — only text-gated channel modulation (V0 design)
    "both"             — text gates + spatial injection (combined)

Initialization
==============
- proj.weight = 0, bias disabled  → proj(vlm_feat) = 0 at init → identity
- alpha = alpha_init (default 0.1) → small perturbation once proj learns

Decoder dims (feature_size=48):
    dec3 → 384 ch  (1/16 spatial)
    dec2 → 192 ch  (1/8  spatial)
    dec1 →  96 ch  (1/4  spatial)
    dec0 →  48 ch  (1/2  spatial)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── V0 text gate (re-used when vlm_mode in ("text", "both")) ───────────────

class _GatedFusion(nn.Module):
    """
    Channel-wise gated residual modulation (identical to V0).

    gate = tanh( Linear(text_dim → feat_dim)(text_embed) )
    out  = feat * (1 + gate)

    Zero-initialised → gate=tanh(0)=0 → out=feat (identity at init).
    """

    def __init__(self, text_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, feat_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        raw  = self.proj(text_embed)                      # (B or 1, C)
        gate = torch.tanh(raw)
        gate = gate.view(gate.shape[0], gate.shape[1], 1, 1)
        return feat * (1.0 + gate)


# ─── V1 spatial image injection ─────────────────────────────────────────────

class _SpatialVLMInjection(nn.Module):
    """
    Spatial VLM feature injection via residual add.

        vlm_up = bilinear_upsample(vlm_feat, target_size)
        proj   = Conv2d(vlm_dim → feat_dim, 1×1)(vlm_up)
        out    = feat + alpha * proj

    Zero-init projection + small alpha → identity at init; gradients flow
    through proj immediately (alpha * d_proj) and through alpha once proj ≠ 0.
    """

    def __init__(self, vlm_dim: int, feat_dim: int, alpha_init: float = 0.1):
        super().__init__()
        self.proj = nn.Conv2d(vlm_dim, feat_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, feat: torch.Tensor, vlm_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:     (B, feat_dim, H, W)  decoder feature
            vlm_feat: (B, vlm_dim,  gh, gw) VLM spatial feature (14×14)
        Returns:
            (B, feat_dim, H, W)
        """
        target = feat.shape[2:]
        if vlm_feat.shape[2:] != target:
            vlm_feat = F.interpolate(
                vlm_feat, size=target, mode="bilinear", align_corners=False
            )
        return feat + self.alpha * self.proj(vlm_feat)


# ─── Main model ─────────────────────────────────────────────────────────────

class SwinUNETR2DVLMV1(nn.Module):
    """
    2-D SwinUNETR with per-image VLM spatial feature injection (V1).

    Forward signature:
        logits = model(x, text_embed=None, image_vlm_feat=None)

    When both are None the model is equivalent to the baseline SwinUNETR.

    Args:
        img_size:     Spatial size (must be divisible by 32).
        in_channels:  Input image channels.
        out_channels: Output channels (1 for binary segmentation).
        feature_size: SwinUNETR base feature size (default 48).
        text_dim:     VLM text embedding dim (512 for BiomedCLIP/CLIP).
        vlm_dim:      VLM image feature dim  (512 for ViT-B/16 projection).
        vlm_mode:     One of "image" | "text" | "both".
        alpha_init:   Initial value of learnable injection scale.
        drop_rate:    Dropout rate.
        attn_drop_rate: Attention dropout.
        use_checkpoint: Gradient checkpointing.
    """

    def __init__(
        self,
        img_size:       int   = 384,
        in_channels:    int   = 1,
        out_channels:   int   = 1,
        feature_size:   int   = 48,
        text_dim:       int   = 512,
        vlm_dim:        int   = 512,
        vlm_mode:       str   = "image",
        alpha_init:     float = 0.1,
        drop_rate:      float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool  = False,
    ):
        super().__init__()

        if img_size % 32 != 0:
            raise ValueError(
                f"SwinUNETR2DVLMV1 requires img_size divisible by 32, got {img_size}."
            )
        if vlm_mode not in ("image", "text", "both"):
            raise ValueError(f"vlm_mode must be 'image', 'text', or 'both'; got '{vlm_mode}'")

        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as e:
            raise ImportError("monai is required for SwinUNETR2DVLMV1") from e

        self.swin = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=2,
        )
        self.vlm_mode = vlm_mode

        # Decoder channel dims (fine→coarse, matching gate index order)
        # dec0=fs, dec1=2fs, dec2=4fs, dec3=8fs
        fs = feature_size
        feat_dims = [fs, fs * 2, fs * 4, fs * 8]  # [48, 96, 192, 384]

        # ── Text gates (V0 style) ─────────────────────────────────────
        if vlm_mode in ("text", "both"):
            self.text_gates = nn.ModuleList(
                [_GatedFusion(text_dim, d) for d in feat_dims]
            )

        # ── Spatial image injectors (V1) ──────────────────────────────
        if vlm_mode in ("image", "both"):
            self.img_injectors = nn.ModuleList(
                [_SpatialVLMInjection(vlm_dim, d, alpha_init) for d in feat_dims]
            )

    # ------------------------------------------------------------------

    def forward(
        self,
        x:              torch.Tensor,
        text_embed:     torch.Tensor | None = None,
        image_vlm_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, C, H, W) input image tensor.
            text_embed:     (1, text_dim) or (B, text_dim) — text gate input.
                            Required when vlm_mode in ("text", "both").
            image_vlm_feat: (B, vlm_dim, 14, 14) — frozen VLM patch features.
                            Required when vlm_mode in ("image", "both").
        Returns:
            logits: (B, out_channels, H, W)
        """
        # Baseline fallback when no VLM inputs provided
        if text_embed is None and image_vlm_feat is None:
            return self.swin(x)

        m = self.swin  # alias

        # ── Encoder / skip connections ────────────────────────────────
        hidden = m.swinViT(x, m.normalize)
        enc0 = m.encoder1(x)           # (B,  fs,    H/2,  W/2)
        enc1 = m.encoder2(hidden[0])   # (B,  fs,    H/4,  W/4)
        enc2 = m.encoder3(hidden[1])   # (B,  2*fs,  H/8,  W/8)
        enc3 = m.encoder4(hidden[2])   # (B,  4*fs,  H/16, W/16)

        # ── Bottleneck ────────────────────────────────────────────────
        dec4 = m.encoder10(hidden[4])  # (B,  16*fs, H/32, W/32)

        # ── Decoder with optional VLM modulation ─────────────────────
        #   index [3] = coarsest (dec3), [0] = finest (dec0)

        dec3 = m.decoder5(dec4, hidden[3])          # (B, 8*fs, H/16, W/16)
        dec3 = self._apply_vlm(dec3, 3, text_embed, image_vlm_feat)

        dec2 = m.decoder4(dec3, enc3)               # (B, 4*fs, H/8,  W/8)
        dec2 = self._apply_vlm(dec2, 2, text_embed, image_vlm_feat)

        dec1 = m.decoder3(dec2, enc2)               # (B, 2*fs, H/4,  W/4)
        dec1 = self._apply_vlm(dec1, 1, text_embed, image_vlm_feat)

        dec0 = m.decoder2(dec1, enc1)               # (B, fs,   H/2,  W/2)
        dec0 = self._apply_vlm(dec0, 0, text_embed, image_vlm_feat)

        # ── Final decoder + output head ───────────────────────────────
        out    = m.decoder1(dec0, enc0)             # (B, fs,   H,    W)
        logits = m.out(out)                         # (B,  1,   H,    W)
        return logits

    def _apply_vlm(
        self,
        feat:           torch.Tensor,
        idx:            int,
        text_embed:     torch.Tensor | None,
        image_vlm_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply text gate and/or image injection at decoder level `idx`."""
        if self.vlm_mode in ("text", "both") and text_embed is not None:
            feat = self.text_gates[idx](feat, text_embed)
        if self.vlm_mode in ("image", "both") and image_vlm_feat is not None:
            feat = self.img_injectors[idx](feat, image_vlm_feat)
        return feat
