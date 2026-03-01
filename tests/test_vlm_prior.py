"""
Tests for VLM prior (vlm_prior.py) and gated SwinUNETR (swinunetr_vlm.py).

Run with:
    env -u VIRTUAL_ENV uv run pytest tests/test_vlm_prior.py -v
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
def vlm_prior_octa(device):
    from src.models.vlm_prior import VLMPrior
    return VLMPrior(dataset="OCTA500-6M", device=device)


@pytest.fixture(scope="module")
def vlm_prior_drive(device):
    from src.models.vlm_prior import VLMPrior
    return VLMPrior(dataset="DRIVE", device=device)


@pytest.fixture(scope="module")
def swin_vlm_model(device):
    from src.models.swinunetr_vlm import SwinUNETR2DVLM
    # Use small img_size for fast tests (must be divisible by 32)
    m = SwinUNETR2DVLM(img_size=64, in_channels=1, out_channels=1, feature_size=48)
    return m.to(device)


# ---------------------------------------------------------------------------
# VLMPrior tests
# ---------------------------------------------------------------------------

class TestVLMPrior:
    def test_embed_shape_octa(self, vlm_prior_octa):
        """Text embedding must be (1, 512)."""
        embed = vlm_prior_octa.get_text_embed()
        assert embed.shape == (1, 512), f"Expected (1, 512), got {embed.shape}"

    def test_embed_shape_drive(self, vlm_prior_drive):
        """Text embedding must be (1, 512)."""
        embed = vlm_prior_drive.get_text_embed()
        assert embed.shape == (1, 512), f"Expected (1, 512), got {embed.shape}"

    def test_embed_normalised(self, vlm_prior_octa):
        """Embedding must be L2-normalised (unit norm)."""
        embed = vlm_prior_octa.get_text_embed()
        norm = embed.norm(dim=-1).item()
        assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm:.6f}"

    def test_embed_cached(self, vlm_prior_octa):
        """Two calls must return the exact same tensor (cache hit)."""
        e1 = vlm_prior_octa.get_text_embed()
        e2 = vlm_prior_octa.get_text_embed()
        assert e1 is e2, "get_text_embed() must return the cached tensor"

    def test_backbone_name_set(self, vlm_prior_octa):
        """backbone_name must be set and non-empty."""
        assert vlm_prior_octa.backbone_name not in ("", "unknown"), \
            f"backbone_name should be set, got '{vlm_prior_octa.backbone_name}'"

    def test_frozen_weights(self, vlm_prior_octa):
        """All VLM parameters must be frozen."""
        model = vlm_prior_octa._clip_model
        for name, p in model.named_parameters():
            assert not p.requires_grad, f"Parameter {name} is not frozen"

    def test_prompts_differ(self, vlm_prior_octa, vlm_prior_drive):
        """OCTA and DRIVE prompts must differ → embeddings must differ."""
        assert vlm_prior_octa.prompt != vlm_prior_drive.prompt
        e_octa  = vlm_prior_octa.get_text_embed()
        e_drive = vlm_prior_drive.get_text_embed()
        diff = (e_octa - e_drive).abs().max().item()
        assert diff > 1e-4, "OCTA and DRIVE embeddings should differ"


# ---------------------------------------------------------------------------
# SwinUNETR2DVLM — gating tests
# ---------------------------------------------------------------------------

class TestSwinUNETR2DVLM:
    def test_forward_with_embed(self, swin_vlm_model, vlm_prior_octa, device):
        """Model must produce correct output shape with text_embed."""
        x = torch.randn(2, 1, 64, 64, device=device)
        embed = vlm_prior_octa.get_text_embed()  # (1, 512)
        with torch.no_grad():
            out = swin_vlm_model(x, embed)
        assert out.shape == (2, 1, 64, 64), f"Bad output shape: {out.shape}"

    def test_forward_without_embed(self, swin_vlm_model, device):
        """Model must fall back gracefully when text_embed=None."""
        x = torch.randn(2, 1, 64, 64, device=device)
        with torch.no_grad():
            out = swin_vlm_model(x, text_embed=None)
        assert out.shape == (2, 1, 64, 64)

    def test_zero_init_identity(self, device):
        """
        With zero-initialised gates, forward(x, embed) should equal forward(x).

        Gate proj is zeros → raw=0 → tanh(0)=0 → feat*(1+0)=feat.
        Both paths should produce identical logits (within fp32 tolerance).
        """
        from src.models.swinunetr_vlm import SwinUNETR2DVLM
        m = SwinUNETR2DVLM(img_size=64, in_channels=1, out_channels=1,
                            feature_size=48)
        m = m.to(device).eval()

        # Verify gates are still zero-init (no training step done)
        for gate in m.gates:
            assert gate.proj.weight.abs().max().item() == 0.0, \
                "Gate weights should be zero at init"
            assert gate.proj.bias.abs().max().item() == 0.0, \
                "Gate biases should be zero at init"

        x = torch.randn(1, 1, 64, 64, device=device)
        embed = torch.randn(1, 512, device=device)  # arbitrary embed

        with torch.no_grad():
            out_with = m(x, embed)
            out_without = m(x, None)

        max_diff = (out_with - out_without).abs().max().item()
        assert max_diff < 1e-5, \
            f"Zero-init gates should give identity; max diff={max_diff:.2e}"

    def test_gate_count(self, swin_vlm_model):
        """Must have exactly 4 gate modules."""
        assert len(swin_vlm_model.gates) == 4

    def test_gate_dims(self, device):
        """Gate linear layers must have correct input/output dims."""
        from src.models.swinunetr_vlm import SwinUNETR2DVLM
        fs = 48
        m = SwinUNETR2DVLM(img_size=64, in_channels=1, out_channels=1,
                            feature_size=fs, text_dim=512)
        expected = [fs, fs * 2, fs * 4, fs * 8]  # [48, 96, 192, 384]
        for i, (gate, exp_out) in enumerate(zip(m.gates, expected)):
            assert gate.proj.in_features == 512, \
                f"gate[{i}] in_features should be 512"
            assert gate.proj.out_features == exp_out, \
                f"gate[{i}] out_features should be {exp_out}, got {gate.proj.out_features}"

    def test_build_model_swinunetr_vlm(self, device):
        """build_model should return SwinUNETR2DVLM for 'swinunetr_vlm'."""
        from src.models import build_model
        from src.models.swinunetr_vlm import SwinUNETR2DVLM
        cfg = {
            "model": "swinunetr_vlm",
            "in_channels": 1,
            "out_channels": 1,
            "image_size": 64,
            "swinunetr": {"img_size": 64, "feature_size": 48},
            "vlm": {"text_dim": 512},
        }
        model = build_model(cfg)
        assert isinstance(model, SwinUNETR2DVLM)
