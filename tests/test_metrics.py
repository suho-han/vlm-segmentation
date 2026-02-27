"""
Tests for src/metrics/dice_iou.py  (and placeholder checks for hd95/topology).
Run with:  pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from src.metrics.dice_iou import compute_metrics, dice_score, iou_score


# ─── dice_score ─────────────────────────────────────────────────────────────

class TestDiceScore:
    def test_perfect_match(self):
        mask = np.ones((64, 64), dtype=bool)
        assert dice_score(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        pred   = np.zeros((64, 64), dtype=bool)
        target = np.ones((64, 64),  dtype=bool)
        # With smooth=1e-6: (0 + 1e-6) / (0 + 64*64 + 1e-6) ≈ 0
        score = dice_score(pred, target)
        assert score < 0.01

    def test_partial_overlap(self):
        pred   = np.zeros((4, 4), dtype=bool)
        target = np.zeros((4, 4), dtype=bool)
        pred[:, :2]   = True  # left half
        target[:, 1:3] = True  # centre
        d = dice_score(pred, target)
        assert 0.0 < d < 1.0

    def test_empty_both(self):
        pred   = np.zeros((32, 32), dtype=bool)
        target = np.zeros((32, 32), dtype=bool)
        # Both empty: numerator and denominator are both smooth → ~1.0
        d = dice_score(pred, target)
        assert d == pytest.approx(1.0, abs=1e-3)


# ─── iou_score ──────────────────────────────────────────────────────────────

class TestIouScore:
    def test_perfect_match(self):
        mask = np.ones((32, 32), dtype=bool)
        assert iou_score(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        pred   = np.zeros((32, 32), dtype=bool)
        target = np.ones((32, 32),  dtype=bool)
        assert iou_score(pred, target) < 0.01

    def test_half_overlap(self):
        pred   = np.zeros((4, 4), dtype=bool)
        target = np.zeros((4, 4), dtype=bool)
        pred[:, :2]   = True
        target[:, 1:3] = True
        # intersection=4, union=12 → ~0.333
        iou = iou_score(pred, target)
        assert iou == pytest.approx(4 / 12, abs=0.01)


# ─── compute_metrics ────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_keys_present(self):
        pred   = np.random.rand(64, 64).astype(np.float32)
        target = (np.random.rand(64, 64) > 0.5).astype(np.float32)
        m = compute_metrics(pred, target)
        for key in ("dice", "iou", "precision", "recall", "f1", "accuracy"):
            assert key in m, f"Missing key: {key}"

    def test_perfect_prediction(self):
        target = (np.random.rand(32, 32) > 0.5).astype(np.float32)
        m = compute_metrics(target.copy(), target)
        assert m["dice"]     == pytest.approx(1.0, abs=1e-4)
        assert m["iou"]      == pytest.approx(1.0, abs=1e-4)
        assert m["precision"] == pytest.approx(1.0, abs=1e-4)
        assert m["recall"]   == pytest.approx(1.0, abs=1e-4)
        assert m["accuracy"] == pytest.approx(1.0, abs=1e-4)

    def test_values_in_range(self):
        pred   = np.random.rand(64, 64).astype(np.float32)
        target = (np.random.rand(64, 64) > 0.5).astype(np.float32)
        m = compute_metrics(pred, target)
        for key in ("dice", "iou", "precision", "recall", "f1", "accuracy"):
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0,1]"

    def test_logit_input(self):
        """compute_metrics should handle raw logits (values > 1)."""
        logits = np.random.randn(64, 64).astype(np.float32) * 5.0
        target = (np.random.rand(64, 64) > 0.5).astype(np.float32)
        m = compute_metrics(logits, target)
        assert 0.0 <= m["dice"] <= 1.0


# ─── hd95 placeholder interface check ───────────────────────────────────────

def test_hd95_interface():
    """HD95 should return a float (or inf for empty masks)."""
    from src.metrics.hd95 import hd95
    pred   = np.zeros((32, 32), dtype=bool)
    target = np.zeros((32, 32), dtype=bool)
    result = hd95(pred, target)
    assert isinstance(result, float)
    assert result == float("inf")  # both empty → inf


# ─── topology placeholder interface check ────────────────────────────────────

def test_betti_error_interface():
    """betti_error should return a dict with expected keys."""
    from src.metrics.topology import betti_error
    pred   = np.zeros((32, 32), dtype=bool)
    target = np.zeros((32, 32), dtype=bool)
    result = betti_error(pred, target)
    for key in ("b0_pred", "b0_target", "b0_error", "b1_pred", "b1_target", "b1_error"):
        assert key in result, f"Missing key: {key}"
