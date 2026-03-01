"""
SwinUNETR-2D with channel-wise gated text-embedding modulation.

Architecture
============
Identical to the SwinUNETR2D baseline, with one addition: after each of
the 4 decoder stages (dec3→dec0) a lightweight gating module modulates
the feature map using a frozen VLM text embedding:

    gate = sigmoid( Linear(text_dim → feat_dim)(text_embed) )  # (B, C, 1, 1)
    decoder_feat = decoder_feat * (1 + gate)                   # residual

Decoder-stage feature dimensions (feature_size=48):
    dec3 → 384 ch  (1/16 spatial)
    dec2 → 192 ch  (1/8  spatial)
    dec1 →  96 ch  (1/4  spatial)
    dec0 →  48 ch  (1/2  spatial)

Initialization
==============
Linear weights and biases are zero-initialised so that at the start of
training gate=sigmoid(0)=0.5 → (1+gate)=1.5 — wait, that's not identity.
Actually we want gate=0 → (1+gate)=1 (identity pass-through).
With zero weight AND zero bias: gate = sigmoid(Linear(0)) = sigmoid(0) = 0.5.

To achieve gate=0 at init we use *negative large bias* or simply:
  - gate_proj = Linear(text_dim → feat_dim, bias=True)
  - init weight = 0, bias = -∞... that's unstable.

Better: keep the residual formulation but define:

    raw = Linear(text_dim → feat_dim)(text_embed)   # zero-init → 0
    gate = sigmoid(raw)                              # → 0.5
    feat_out = feat * (1 + gate - 0.5)              # → feat * 1.0  ✓

Or equivalently, use tanh instead of sigmoid:
    gate = tanh(raw)  # zero-init raw → tanh(0)=0 → feat * 1.0  ✓

We use tanh so that zero-init truly gives identity behaviour.
Range of gate: (-1, 1) → decoder_feat * (1+gate) ∈ (0, 2*feat).
This is safe and expressive.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _GatedFusion(nn.Module):
    """
    Channel-wise gated residual modulation.

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
        """
        Args:
            feat:       (B, C, H, W)
            text_embed: (1, text_dim) or (B, text_dim)
        Returns:
            (B, C, H, W) — same shape as feat
        """
        raw  = self.proj(text_embed)                      # (B, C) or (1, C)
        gate = torch.tanh(raw)                            # (B, C) or (1, C)
        gate = gate.view(gate.shape[0], gate.shape[1], 1, 1)  # broadcast over H,W
        return feat * (1.0 + gate)


class SwinUNETR2DVLM(nn.Module):
    """
    2-D SwinUNETR with VLM text-embedding gated fusion.

    Forward signature:
        logits = model(x, text_embed)

    When text_embed is None the model falls back to the standard
    SwinUNETR forward (identical to the baseline).

    Args:
        img_size:     Spatial size (must be divisible by 32).
        in_channels:  Input image channels.
        out_channels: Output channels (1 for binary segmentation).
        feature_size: SwinUNETR base feature size (default 48).
        text_dim:     VLM embedding dimension (BiomedCLIP/CLIP = 512).
        drop_rate:    Dropout rate.
        attn_drop_rate: Attention dropout.
        use_checkpoint: Gradient checkpointing (saves memory).
    """

    def __init__(
        self,
        img_size: int = 384,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 48,
        text_dim: int = 512,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        if img_size % 32 != 0:
            raise ValueError(
                f"SwinUNETR2DVLM requires img_size divisible by 32, got {img_size}."
            )

        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as e:
            raise ImportError("monai is required for SwinUNETR2DVLM") from e

        self.swin = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=2,
        )

        # 4 decoder levels: dec0..dec3  (finest→coarsest)
        # dims: [fs, 2*fs, 4*fs, 8*fs] = [48, 96, 192, 384] for fs=48
        feat_dims = [
            feature_size,       # dec0 (from decoder2 output)
            feature_size * 2,   # dec1 (from decoder3 output)
            feature_size * 4,   # dec2 (from decoder4 output)
            feature_size * 8,   # dec3 (from decoder5 output)
        ]
        self.gates = nn.ModuleList(
            [_GatedFusion(text_dim, d) for d in feat_dims]
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, C, H, W) input image tensor.
            text_embed: (1, text_dim) or (B, text_dim) VLM embedding.
                        When None, falls back to unmodified SwinUNETR.
        Returns:
            logits: (B, out_channels, H, W)
        """
        if text_embed is None:
            return self.swin(x)

        m = self.swin  # alias for brevity

        # ── Encoder / skip connections ────────────────────────────────
        hidden = m.swinViT(x, m.normalize)
        # hidden[0..3]: spatial pyramid features, hidden[4]: bottleneck
        enc0 = m.encoder1(x)           # (B,  48, H/2,  W/2)
        enc1 = m.encoder2(hidden[0])   # (B,  48, H/4,  W/4)
        enc2 = m.encoder3(hidden[1])   # (B,  96, H/8,  W/8)
        enc3 = m.encoder4(hidden[2])   # (B, 192, H/16, W/16)

        # ── Bottleneck ────────────────────────────────────────────────
        dec4 = m.encoder10(hidden[4])  # (B, 768, H/32, W/32)

        # ── Decoder with gated modulation ────────────────────────────
        dec3 = m.decoder5(dec4, hidden[3])        # (B, 384, H/16, W/16)
        dec3 = self.gates[3](dec3, text_embed)    # gate[3]: 512→384

        dec2 = m.decoder4(dec3, enc3)             # (B, 192, H/8,  W/8)
        dec2 = self.gates[2](dec2, text_embed)    # gate[2]: 512→192

        dec1 = m.decoder3(dec2, enc2)             # (B,  96, H/4,  W/4)
        dec1 = self.gates[1](dec1, text_embed)    # gate[1]: 512→96

        dec0 = m.decoder2(dec1, enc1)             # (B,  48, H/2,  W/2)
        dec0 = self.gates[0](dec0, text_embed)    # gate[0]: 512→48

        # ── Final decoder + output head (no gating) ──────────────────
        out    = m.decoder1(dec0, enc0)            # (B,  48, H,    W)
        logits = m.out(out)                        # (B,   1, H,    W)
        return logits
