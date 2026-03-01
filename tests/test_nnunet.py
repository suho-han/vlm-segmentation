"""
Unit tests for NNUNet2D (src/models/nnunet_2d.py).

Tests cover:
  - Channel schedule
  - Default forward pass (single tensor output)
  - Output spatial shape matches input
  - Deep supervision output list
  - VLM forward signature (text_embed / image_vlm_feat ignored → identity)
  - Kaiming init (no NaN weights)
  - build_model dispatch for 'nnunet_2d' and 'nnunet'
  - Odd-size input (pad path)
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.nnunet_2d import NNUNet2D


# ─── channel schedule ────────────────────────────────────────────────────────

def test_channel_schedule_default():
    """Default n_pool=5, base=32, max=320 → [32,64,128,256,320,320]."""
    model = NNUNet2D()
    assert model._chs == [32, 64, 128, 256, 320, 320]


def test_channel_schedule_custom():
    model = NNUNet2D(base_channels=16, n_pool=4, max_channels=128)
    assert model._chs == [16, 32, 64, 128, 128]


def test_channel_schedule_capped():
    """max_channels cap is enforced."""
    model = NNUNet2D(base_channels=32, n_pool=3, max_channels=64)
    assert all(c <= 64 for c in model._chs)


# ─── forward shape ───────────────────────────────────────────────────────────

def test_forward_default_shape():
    """Standard OCTA-like input: (B=2, 1, 384, 384) → same H,W."""
    model = NNUNet2D()
    x = torch.randn(2, 1, 384, 384)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1, 384, 384)


def test_forward_drive_shape():
    """DRIVE-like: (2, 1, 512, 512)."""
    model = NNUNet2D()
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 512, 512)


def test_forward_small():
    """Small even-size input (n_pool=3 for speed)."""
    model = NNUNet2D(n_pool=3)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_forward_multichannel_in():
    """3-channel input."""
    model = NNUNet2D(in_channels=3, n_pool=3)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_forward_multichannel_out():
    """Multi-class output."""
    model = NNUNet2D(out_channels=4, n_pool=3)
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4, 64, 64)


# ─── deep supervision ─────────────────────────────────────────────────────────

def test_deep_supervision_returns_list():
    model = NNUNet2D(n_pool=3, deep_supervision=True)
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, list), "deep_supervision=True must return list"
    assert len(out) >= 2


def test_deep_supervision_all_same_spatial():
    """All deep-supervision outputs are upsampled to full resolution."""
    model = NNUNet2D(n_pool=3, deep_supervision=True)
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        outs = model(x)
    for o in outs:
        assert o.shape[2:] == (64, 64), f"Expected 64×64, got {o.shape[2:]}"


def test_deep_supervision_false_returns_tensor():
    model = NNUNet2D(n_pool=3, deep_supervision=False)
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor)


# ─── VLM signature ───────────────────────────────────────────────────────────

def test_vlm_args_ignored_text():
    """text_embed is accepted and silently ignored."""
    model = NNUNet2D(n_pool=3)
    x = torch.randn(2, 1, 64, 64)
    text = torch.randn(1, 512)
    with torch.no_grad():
        out_plain = model(x)
        out_text  = model(x, text_embed=text)
    assert torch.allclose(out_plain, out_text)


def test_vlm_args_ignored_image_feat():
    """image_vlm_feat is accepted and silently ignored."""
    model = NNUNet2D(n_pool=3)
    x = torch.randn(2, 1, 64, 64)
    img_feat = torch.randn(2, 512, 14, 14)
    with torch.no_grad():
        out_plain = model(x)
        out_vfeat = model(x, image_vlm_feat=img_feat)
    assert torch.allclose(out_plain, out_vfeat)


# ─── weight init ─────────────────────────────────────────────────────────────

def test_no_nan_weights():
    model = NNUNet2D(n_pool=3)
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"


def test_output_not_nan():
    model = NNUNet2D(n_pool=3)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()


# ─── odd-size pad path ────────────────────────────────────────────────────────

def test_odd_size_pad():
    """Input 65×65 triggers the pad branch in the decoder — should not crash."""
    model = NNUNet2D(n_pool=3)
    x = torch.randn(1, 1, 65, 65)
    with torch.no_grad():
        out = model(x)
    # output may be 65×65 or 64×64 depending on padding strategy; check not crash
    assert out.ndim == 4


# ─── build_model dispatch ─────────────────────────────────────────────────────

def test_build_model_nnunet_2d():
    from src.models import build_model
    cfg = {"model": "nnunet_2d", "nnunet": {"n_pool": 3}}
    model = build_model(cfg)
    assert isinstance(model, NNUNet2D)


def test_build_model_nnunet_alias():
    from src.models import build_model
    cfg = {"model": "nnunet", "nnunet": {"n_pool": 3}}
    model = build_model(cfg)
    assert isinstance(model, NNUNet2D)


def test_build_model_nnunet_custom_cfg():
    from src.models import build_model
    cfg = {
        "model": "nnunet_2d",
        "nnunet": {
            "base_channels": 16,
            "n_pool": 4,
            "max_channels": 128,
            "deep_supervision": False,
        },
    }
    model = build_model(cfg)
    assert model._chs[0] == 16
    assert model.n_pool == 4
