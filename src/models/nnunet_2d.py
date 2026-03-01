"""
2D nnU-Net-style segmentation network (pure PyTorch, no nnU-Net framework).

Design follows the core nnU-Net principles for 2D:
  - InstanceNorm2d (affine=True) instead of BatchNorm
  - LeakyReLU(0.01) activations
  - Strided convolution (k=3, s=2) for downsampling instead of MaxPool
  - ConvTranspose2d for upsampling
  - Dynamic channel scaling capped at max_channels

Architecture (n_pool=5, base_channels=32, max_channels=320):
  Encoder
    enc[0]  : DoubleConv(1  → 32)  → skip[0]   (H, W)
    down[0] : Conv(32  → 64,  s=2)
    enc[1]  : DoubleConv(64  → 64)  → skip[1]   (H/2, W/2)
    down[1] : Conv(64  → 128, s=2)
    enc[2]  : DoubleConv(128 → 128) → skip[2]   (H/4, W/4)
    down[2] : Conv(128 → 256, s=2)
    enc[3]  : DoubleConv(256 → 256) → skip[3]   (H/8, W/8)
    down[3] : Conv(256 → 320, s=2)
    enc[4]  : DoubleConv(320 → 320) → skip[4]   (H/16, W/16)
    down[4] : Conv(320 → 320, s=2)
    bottleneck: DoubleConv(320 → 320)            (H/32, W/32)

  Decoder
    up[0] : ConvT(320→320) + cat(skip[4]) → DoubleConv(640→320)
    up[1] : ConvT(320→256) + cat(skip[3]) → DoubleConv(512→256)
    up[2] : ConvT(256→128) + cat(skip[2]) → DoubleConv(256→128)
    up[3] : ConvT(128→64)  + cat(skip[1]) → DoubleConv(128→64)
    up[4] : ConvT(64→32)   + cat(skip[0]) → DoubleConv(64→32)
    out   : Conv(32→1, 1×1)

Input requirements:
  - H and W must each be divisible by 2^n_pool
  - n_pool=5 → divisible by 32 (matches OCTA=384, DRIVE=512)

VLM interface:
  forward(x, text_embed=None, image_vlm_feat=None)
  Both VLM args are accepted and silently ignored by default.
  Override _apply_vlm() to enable injection (e.g., NNUNet2DVLM subclass).

Optional deep supervision:
  When deep_supervision=True, forward() returns list[Tensor]:
    [logits_full, logits_h2, logits_h4, ...]  (full res first)
  When False (default), returns Tensor.
  NOTE: current train.py assumes single tensor output; enable only with
        a compatible training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building blocks ────────────────────────────────────────────────────────

class _ConvNormAct(nn.Sequential):
    """Conv2d → InstanceNorm2d(affine) → LeakyReLU(0.01)."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )


class _DoubleConvBlock(nn.Sequential):
    """Two consecutive _ConvNormAct blocks (in_ch → mid_ch → out_ch)."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        mid = mid_ch or out_ch
        super().__init__(
            _ConvNormAct(in_ch, mid),
            _ConvNormAct(mid, out_ch),
        )


class _StrideDownConv(nn.Sequential):
    """Strided 3×3 conv (stride=2) for downsampling — replaces MaxPool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )


# ─── Main model ─────────────────────────────────────────────────────────────

class NNUNet2D(nn.Module):
    """
    2D nnU-Net-inspired UNet (pure PyTorch, no nnU-Net framework dependency).

    Args:
        in_channels:      Input image channels (1 for grayscale).
        out_channels:     Output channels (1 for binary segmentation).
        base_channels:    Channel width at the shallowest encoder level (default 32).
        n_pool:           Number of downsampling operations (default 5).
                          Input H/W must be divisible by 2^n_pool.
        max_channels:     Channel cap to limit memory usage (default 320).
        deep_supervision: Return list of logits at multiple scales when True.
    """

    def __init__(
        self,
        in_channels:      int  = 1,
        out_channels:     int  = 1,
        base_channels:    int  = 32,
        n_pool:           int  = 5,
        max_channels:     int  = 320,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.n_pool = n_pool
        self.deep_supervision = deep_supervision

        # Channel schedule: [32, 64, 128, 256, 320, 320] for default params
        chs = [min(base_channels * (2 ** i), max_channels)
               for i in range(n_pool + 1)]
        self._chs = chs  # expose for subclasses / tests

        # ── Encoder ──────────────────────────────────────────────────
        # enc_blocks[0..n_pool-1]: produce skip connections
        # enc_blocks[n_pool]:       bottleneck (no skip)
        self.enc_blocks = nn.ModuleList()
        self.down_convs  = nn.ModuleList()

        self.enc_blocks.append(_DoubleConvBlock(in_channels, chs[0]))
        for i in range(n_pool):
            self.down_convs.append(_StrideDownConv(chs[i], chs[i + 1]))
            self.enc_blocks.append(_DoubleConvBlock(chs[i + 1], chs[i + 1]))

        # ── Decoder ───────────────────────────────────────────────────
        # up_convs[i] : upsample from chs[n_pool-i]   → chs[n_pool-i-1]
        # dec_blocks[i]: fuse concat(skip, up) → chs[n_pool-i-1]
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(n_pool):
            in_up  = chs[n_pool - i]
            out_up = chs[n_pool - i - 1]
            self.up_convs.append(
                nn.ConvTranspose2d(in_up, out_up, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(_DoubleConvBlock(out_up * 2, out_up))

        # ── Output head ───────────────────────────────────────────────
        self.out_conv = nn.Conv2d(chs[0], out_channels, kernel_size=1)

        # ── Deep supervision heads (optional) ─────────────────────────
        if deep_supervision:
            # Attach output heads to decoder levels 1..n_pool-2 (skip finest)
            self.ds_heads = nn.ModuleList(
                [nn.Conv2d(chs[n_pool - 1 - i], out_channels, kernel_size=1)
                 for i in range(1, n_pool - 1)]
            )
        else:
            self.ds_heads = None

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming uniform init for conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # VLM hook (override in VLM subclass; default = identity)
    # ------------------------------------------------------------------

    def _apply_vlm(
        self,
        dec_feat:       torch.Tensor,
        level:          int,
        text_embed:     torch.Tensor | None,
        image_vlm_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Optional VLM modulation at a decoder level.

        Override this in a VLM-aware subclass (e.g., NNUNet2DVLM).
        Default: identity pass-through.

        Args:
            dec_feat:       (B, C, H, W) decoder feature at `level`.
            level:          Decoder level index (0 = coarsest, n_pool-1 = finest).
            text_embed:     (1, D) text embedding or None.
            image_vlm_feat: (B, D, 14, 14) VLM patch features or None.
        Returns:
            (B, C, H, W) — same shape as dec_feat.
        """
        return dec_feat

    # ------------------------------------------------------------------

    def forward(
        self,
        x:              torch.Tensor,
        text_embed:     torch.Tensor | None = None,
        image_vlm_feat: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            x:              (B, C, H, W) input image.
            text_embed:     Unused in base class; passed to _apply_vlm.
            image_vlm_feat: Unused in base class; passed to _apply_vlm.
        Returns:
            Tensor (B, out_ch, H, W) when deep_supervision=False.
            list[Tensor] when deep_supervision=True:
                [logits_full, logits_h2, logits_h4, ...]
        """
        # ── Encoder ──────────────────────────────────────────────────
        skips: list[torch.Tensor] = []
        for i in range(self.n_pool):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.down_convs[i](x)
        x = self.enc_blocks[self.n_pool](x)   # bottleneck

        # ── Decoder ───────────────────────────────────────────────────
        ds_out: list[torch.Tensor] = []
        ds_idx = 0  # index into ds_heads

        for i in range(self.n_pool):
            x = self.up_convs[i](x)
            skip = skips[self.n_pool - 1 - i]  # coarsest skip first

            # Pad if there is a 1-pixel mismatch from odd input sizes
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                               0, skip.shape[-2] - x.shape[-2]])

            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)
            x = self._apply_vlm(x, i, text_embed, image_vlm_feat)

            # Deep supervision (collect all but first / finest decoder level)
            if self.ds_heads is not None and 0 < i < self.n_pool - 1:
                ds_out.append(self.ds_heads[ds_idx](x))
                ds_idx += 1

        logits = self.out_conv(x)

        if self.deep_supervision and ds_out:
            # Upsample DS outputs to match full resolution
            full_size = logits.shape[2:]
            ds_upsampled = [
                F.interpolate(d, size=full_size, mode="bilinear",
                              align_corners=False)
                for d in ds_out
            ]
            return [logits] + ds_upsampled

        return logits
