"""
Unit tests for TransUNetSeg (src/models/transunet.py).

Tests cover:
  - Forward shapes: OCTA (384px), DRIVE (512px), small (64px)
  - VLM args None → output unchanged (identity at init)
  - Zero-init vlm_proj weights → adding VLM feat = identity at init
  - vlm_proj count and zero-init verification
  - vlm_proj channel shapes (vlm_dim → decoder_chs)
  - decoder_chs configuration
  - build_model dispatch for 'transunet'
  - No-NaN weights and output
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
timm  = pytest.importorskip("timm",  reason="timm not installed")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transunet import TransUNetSeg


# ─── helpers ─────────────────────────────────────────────────────────────────

def make_model(**kwargs):
    defaults = dict(
        in_channels=1, out_channels=1, img_size=64,
        timm_model="vit_base_patch16_384", pretrained=False,
        decoder_chs=[16, 8, 4, 2],  # tiny for test speed
    )
    defaults.update(kwargs)
    return TransUNetSeg(**defaults)


# ─── forward shapes ──────────────────────────────────────────────────────────

def test_forward_octa_shape():
    """OCTA-like input (B=1, 1, 384, 384) → same spatial output."""
    model = TransUNetSeg(
        in_channels=1, out_channels=1, img_size=384,
        pretrained=False, decoder_chs=[256, 128, 64, 16],
    )
    x = torch.randn(1, 1, 384, 384)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 384, 384), f"Got {out.shape}"


def test_forward_drive_shape():
    """DRIVE-like input (B=1, 1, 512, 512) → same spatial output."""
    model = TransUNetSeg(
        in_channels=1, out_channels=1, img_size=512,
        pretrained=False, decoder_chs=[256, 128, 64, 16],
    )
    x = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 512, 512), f"Got {out.shape}"


def test_forward_small_shape():
    """Small input (B=1, 1, 64, 64) with tiny decoder."""
    model = make_model()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64), f"Got {out.shape}"


def test_forward_multichannel_out():
    """Multi-class output (out_channels=4)."""
    model = make_model(out_channels=4)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 64, 64), f"Got {out.shape}"


# ─── VLM args = None → identity ──────────────────────────────────────────────

def test_vlm_none_text_identity():
    """text_embed=None is accepted and output is unchanged."""
    model = make_model()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out_plain = model(x)
        out_text  = model(x, text_embed=None)
    assert torch.allclose(out_plain, out_text)


def test_vlm_none_image_identity():
    """image_vlm_feat=None is accepted and output is unchanged."""
    model = make_model()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out_plain  = model(x)
        out_no_vlm = model(x, image_vlm_feat=None)
    assert torch.allclose(out_plain, out_no_vlm)


# ─── zero-init vlm_proj → injection is identity at init ──────────────────────

def test_zero_init_vlm_proj_identity():
    """At init all vlm_proj weights = 0 → adding VLM feat adds nothing."""
    model = make_model()
    x        = torch.randn(1, 1, 64, 64)
    vlm_feat = torch.randn(1, 512, 14, 14)
    with torch.no_grad():
        out_no_vlm   = model(x)
        out_with_vlm = model(x, image_vlm_feat=vlm_feat)
    assert torch.allclose(out_no_vlm, out_with_vlm), \
        "Zero-init vlm_proj must make VLM injection a no-op at init"


# ─── vlm_proj structure ───────────────────────────────────────────────────────

def test_vlm_proj_count():
    """TransUNetSeg should register exactly 4 vlm_proj layers (one per decode stage)."""
    model = make_model()
    assert len(model.vlm_proj) == 4, f"Expected 4 vlm_proj, got {len(model.vlm_proj)}"


def test_vlm_proj_zero_init():
    """All vlm_proj Conv2d weights must be zero at init."""
    model = make_model()
    for i, proj in enumerate(model.vlm_proj):
        assert torch.all(proj.weight == 0), \
            f"vlm_proj[{i}].weight is not zero-init"


def test_vlm_proj_channel_shapes():
    """vlm_proj[i] must project vlm_dim → decoder_chs[i]."""
    dec_chs = [16, 8, 4, 2]
    model   = make_model(decoder_chs=dec_chs, vlm_dim=512)
    for i, (proj, ch) in enumerate(zip(model.vlm_proj, dec_chs)):
        assert proj.weight.shape == (ch, 512, 1, 1), \
            f"vlm_proj[{i}] shape wrong: expected ({ch}, 512, 1, 1), got {proj.weight.shape}"


# ─── decoder_chs configuration ────────────────────────────────────────────────

def test_decoder_blocks_count():
    """Number of decoder blocks matches len(decoder_chs)."""
    dec_chs = [16, 8, 4, 2]
    model   = make_model(decoder_chs=dec_chs)
    assert len(model.decoder_blocks) == len(dec_chs), \
        f"Expected {len(dec_chs)} decoder blocks, got {len(model.decoder_blocks)}"


# ─── no NaN ──────────────────────────────────────────────────────────────────

def test_no_nan_weights():
    model = make_model()
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN weight in {name}"


def test_no_nan_output():
    model = make_model()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "NaN in output"


# ─── build_model dispatch ─────────────────────────────────────────────────────

def test_build_model_dispatch():
    """build_model('transunet') returns TransUNetSeg."""
    from src.models import build_model
    cfg = {
        "model": "transunet",
        "in_channels": 1,
        "out_channels": 1,
        "image_size": 64,
        "transunet": {
            "timm_model": "vit_base_patch16_384",
            "pretrained": False,
            "decoder_chs": [16, 8, 4, 2],
        },
    }
    model = build_model(cfg)
    assert isinstance(model, TransUNetSeg)


def test_build_model_default_transunet_cfg():
    """build_model works with minimal transunet config (uses defaults)."""
    from src.models import build_model
    cfg = {
        "model": "transunet",
        "image_size": 64,
    }
    model = build_model(cfg)
    assert isinstance(model, TransUNetSeg)
