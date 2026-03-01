"""
Tests for VLM image feature extraction (vlm_prior.py V1) and
spatial injection model (swinunetr_vlm_v1.py).

Run with:
    env -u VIRTUAL_ENV uv run pytest tests/test_vlm_image_prior.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def vlm_prior(device):
    from src.models.vlm_prior import VLMPrior
    return VLMPrior(dataset="OCTA500-6M", device=device)


@pytest.fixture(scope="module")
def vlm_v1_image_model(device):
    from src.models.swinunetr_vlm_v1 import SwinUNETR2DVLMV1
    m = SwinUNETR2DVLMV1(
        img_size=64, in_channels=1, out_channels=1,
        feature_size=48, vlm_mode="image", alpha_init=0.1,
    )
    return m.to(device)


@pytest.fixture(scope="module")
def vlm_v1_both_model(device):
    from src.models.swinunetr_vlm_v1 import SwinUNETR2DVLMV1
    m = SwinUNETR2DVLMV1(
        img_size=64, in_channels=1, out_channels=1,
        feature_size=48, vlm_mode="both", alpha_init=0.1,
    )
    return m.to(device)


# ---------------------------------------------------------------------------
# VLMPrior image feature tests
# ---------------------------------------------------------------------------

class TestVLMPriorImageFeatures:

    def test_image_features_shape_grayscale(self, vlm_prior, device):
        """Image features must be (B, 512, 14, 14) for grayscale input."""
        x = torch.rand(2, 1, 384, 384, device=device)
        feats = vlm_prior.get_image_features(x)
        assert feats.shape == (2, 512, 14, 14), \
            f"Expected (2, 512, 14, 14), got {feats.shape}"

    def test_image_features_shape_rgb(self, vlm_prior, device):
        """Image features must be (B, 512, 14, 14) for RGB input."""
        x = torch.rand(2, 3, 224, 224, device=device)
        feats = vlm_prior.get_image_features(x)
        assert feats.shape == (2, 512, 14, 14), \
            f"Expected (2, 512, 14, 14), got {feats.shape}"

    def test_image_features_on_device(self, vlm_prior, device):
        """Output must live on the VLM's device."""
        x = torch.rand(1, 1, 64, 64, device=device)
        feats = vlm_prior.get_image_features(x)
        assert feats.device.type == device.type

    def test_image_features_no_grad(self, vlm_prior, device):
        """VLM features must not require grad (frozen encoder)."""
        x = torch.rand(1, 1, 64, 64, device=device)
        feats = vlm_prior.get_image_features(x)
        assert not feats.requires_grad, "VLM image features should not require grad"

    def test_image_features_batch_independent(self, vlm_prior, device):
        """Each sample in a batch must produce independent features."""
        x = torch.rand(2, 1, 64, 64, device=device)
        feats = vlm_prior.get_image_features(x)
        # Different random inputs → different features
        diff = (feats[0] - feats[1]).abs().max().item()
        assert diff > 1e-4, "Features for different inputs should differ"

    def test_image_features_vary_across_images(self, vlm_prior, device):
        """Running twice with the same image should give same result (deterministic)."""
        x = torch.rand(1, 1, 64, 64, device=device)
        f1 = vlm_prior.get_image_features(x)
        f2 = vlm_prior.get_image_features(x)
        max_diff = (f1 - f2).abs().max().item()
        assert max_diff < 1e-5, f"Same input → same output; max diff={max_diff:.2e}"

    def test_preprocess_expands_grayscale(self, vlm_prior, device):
        """_preprocess_for_vlm must expand 1-channel to 3-channel."""
        x = torch.rand(2, 1, 64, 64, device=device)
        out = vlm_prior._preprocess_for_vlm(x)
        assert out.shape == (2, 3, 224, 224), \
            f"Preprocessed shape: {out.shape}"

    def test_preprocess_resizes(self, vlm_prior, device):
        """_preprocess_for_vlm must always output 224x224."""
        for h, w in [(64, 64), (384, 384), (512, 384)]:
            x = torch.rand(1, 1, h, w, device=device)
            out = vlm_prior._preprocess_for_vlm(x)
            assert out.shape[-2:] == (224, 224), \
                f"Input {h}x{w} → output {out.shape[-2:]}"


# ---------------------------------------------------------------------------
# _SpatialVLMInjection tests
# ---------------------------------------------------------------------------

class TestSpatialVLMInjection:

    def test_injection_output_shape(self, device):
        """Output must match feat shape exactly."""
        from src.models.swinunetr_vlm_v1 import _SpatialVLMInjection
        inj = _SpatialVLMInjection(vlm_dim=512, feat_dim=192).to(device)
        feat     = torch.randn(2, 192, 48, 48, device=device)
        vlm_feat = torch.randn(2, 512, 14, 14, device=device)
        out = inj(feat, vlm_feat)
        assert out.shape == feat.shape, f"Expected {feat.shape}, got {out.shape}"

    def test_zero_init_proj_is_identity(self, device):
        """Zero-init proj → proj(vlm_feat) = 0 → out = feat (identity)."""
        from src.models.swinunetr_vlm_v1 import _SpatialVLMInjection
        inj = _SpatialVLMInjection(vlm_dim=512, feat_dim=96).to(device)

        # Verify proj is zero-initialised
        assert inj.proj.weight.abs().max().item() == 0.0, \
            "proj.weight should be zero at init"

        feat     = torch.randn(2, 96, 96, 96, device=device)
        vlm_feat = torch.randn(2, 512, 14, 14, device=device)
        out = inj(feat, vlm_feat)

        max_diff = (out - feat).abs().max().item()
        assert max_diff < 1e-5, \
            f"Zero-init proj should give identity; max diff={max_diff:.2e}"

    def test_alpha_parameter_exists(self, device):
        """Each injector must have a learnable alpha."""
        from src.models.swinunetr_vlm_v1 import _SpatialVLMInjection
        inj = _SpatialVLMInjection(vlm_dim=512, feat_dim=48, alpha_init=0.1)
        assert hasattr(inj, "alpha") and isinstance(inj.alpha, torch.nn.Parameter)
        assert abs(inj.alpha.item() - 0.1) < 1e-6

    def test_upsample_matches_feat_size(self, device):
        """Injector must upsample VLM feat (14x14) to any decoder resolution."""
        from src.models.swinunetr_vlm_v1 import _SpatialVLMInjection
        inj = _SpatialVLMInjection(vlm_dim=512, feat_dim=384).to(device)
        feat     = torch.randn(1, 384, 24, 24, device=device)   # dec3 @ img_size=384
        vlm_feat = torch.randn(1, 512, 14, 14, device=device)
        out = inj(feat, vlm_feat)
        assert out.shape == feat.shape


# ---------------------------------------------------------------------------
# SwinUNETR2DVLMV1 forward tests
# ---------------------------------------------------------------------------

class TestSwinUNETR2DVLMV1:

    def test_forward_image_only(self, vlm_v1_image_model, vlm_prior, device):
        """Model(image, vlm_mode='image') must produce correct output shape."""
        x = torch.randn(2, 1, 64, 64, device=device)
        with torch.no_grad():
            vlm_feat = vlm_prior.get_image_features(x)
            out = vlm_v1_image_model(x, image_vlm_feat=vlm_feat)
        assert out.shape == (2, 1, 64, 64), f"Bad shape: {out.shape}"

    def test_forward_none_is_baseline(self, vlm_v1_image_model, device):
        """model(x, None, None) must equal baseline SwinUNETR forward."""
        x = torch.randn(1, 1, 64, 64, device=device)
        with torch.no_grad():
            out_vlm  = vlm_v1_image_model(x, text_embed=None, image_vlm_feat=None)
            out_base = vlm_v1_image_model.swin(x)
        max_diff = (out_vlm - out_base).abs().max().item()
        assert max_diff < 1e-5, \
            f"None inputs should give same result as baseline; diff={max_diff:.2e}"

    def test_forward_both_mode(self, vlm_v1_both_model, vlm_prior, device):
        """Model with vlm_mode='both' must handle text+image simultaneously."""
        x = torch.randn(2, 1, 64, 64, device=device)
        with torch.no_grad():
            text_embed = vlm_prior.get_text_embed()
            vlm_feat   = vlm_prior.get_image_features(x)
            out = vlm_v1_both_model(x, text_embed=text_embed, image_vlm_feat=vlm_feat)
        assert out.shape == (2, 1, 64, 64)

    def test_injector_count(self, vlm_v1_image_model):
        """Must have exactly 4 injectors (one per decoder stage)."""
        assert len(vlm_v1_image_model.img_injectors) == 4

    def test_injector_dims(self, device):
        """Injector proj must have correct output channels for each stage."""
        from src.models.swinunetr_vlm_v1 import SwinUNETR2DVLMV1
        fs = 48
        m  = SwinUNETR2DVLMV1(img_size=64, in_channels=1, out_channels=1,
                               feature_size=fs, vlm_mode="image")
        expected = [fs, fs * 2, fs * 4, fs * 8]   # [48, 96, 192, 384]
        for i, (inj, exp_ch) in enumerate(zip(m.img_injectors, expected)):
            actual = inj.proj.out_channels
            assert actual == exp_ch, \
                f"injector[{i}] out_channels should be {exp_ch}, got {actual}"

    def test_no_text_gates_in_image_mode(self, vlm_v1_image_model):
        """vlm_mode='image' must NOT create text_gates attribute."""
        assert not hasattr(vlm_v1_image_model, "text_gates"), \
            "image mode should not have text_gates"

    def test_build_model_swinunetr_vlm_v1(self, device):
        """build_model should return SwinUNETR2DVLMV1 for 'swinunetr_vlm_v1'."""
        from src.models import build_model
        from src.models.swinunetr_vlm_v1 import SwinUNETR2DVLMV1
        cfg = {
            "model": "swinunetr_vlm_v1",
            "in_channels": 1,
            "out_channels": 1,
            "image_size": 64,
            "swinunetr": {"img_size": 64, "feature_size": 48},
            "vlm": {"mode": "image", "vlm_dim": 512, "alpha_init": 0.1},
        }
        model = build_model(cfg)
        assert isinstance(model, SwinUNETR2DVLMV1)

    def test_alpha_is_learnable(self, vlm_v1_image_model):
        """Each injector's alpha must be a learnable parameter."""
        for i, inj in enumerate(vlm_v1_image_model.img_injectors):
            assert isinstance(inj.alpha, torch.nn.Parameter), \
                f"injector[{i}].alpha must be nn.Parameter"
            assert inj.alpha.requires_grad, \
                f"injector[{i}].alpha must require grad"
