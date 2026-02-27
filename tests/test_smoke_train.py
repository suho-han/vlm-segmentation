"""
Smoke test: verify that train/eval pipeline runs end-to-end with dummy data.
Does NOT require real datasets or GPU.

Run with:  pytest tests/test_smoke_train.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="torch not installed — skipping smoke tests")

# Ensure project root is on path (also handled by conftest.py)
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Dummy dataset helpers ───────────────────────────────────────────────────

def _make_dummy_split(root: Path, split: str, n: int = 4, size: int = 64, ext=".bmp"):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = (np.random.rand(size, size) * 255).astype(np.uint8)
        fname = f"{i:04d}{ext}"
        Image.fromarray(arr, mode="L").save(root / split / "images" / fname)
        Image.fromarray(arr, mode="L").save(root / split / "labels" / fname)


# ─── UNetPlusPlus smoke ──────────────────────────────────────────────────────

def test_unetpp_forward():
    """UNet++ forward pass with random input."""
    from src.models.unetpp import UNetPlusPlus
    model = UNetPlusPlus(in_channels=1, out_channels=1, base_channels=16, depth=2)
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 64, 64), f"Unexpected output shape: {out.shape}"


def test_unetpp_deep_supervision():
    """UNet++ with deep supervision returns a list."""
    from src.models.unetpp import UNetPlusPlus
    model = UNetPlusPlus(in_channels=1, out_channels=1, base_channels=16, depth=2,
                         deep_supervision=True)
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        outs = model(x)
    assert isinstance(outs, list)
    assert len(outs) == 2  # depth levels


# ─── Loss smoke ──────────────────────────────────────────────────────────────

def test_bce_dice_loss():
    from src.losses import build_loss
    cfg = {"loss": "bce_dice", "dice_smooth": 1.0, "bce_weight": 1.0, "dice_weight": 1.0}
    criterion = build_loss(cfg)
    logits = torch.randn(2, 1, 32, 32)
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss = criterion(logits, targets)
    assert loss.item() > 0


# ─── Exp utils smoke ─────────────────────────────────────────────────────────

def test_setup_run_dir(tmp_path):
    from src.utils.exp import setup_run_dir
    cfg = {
        "dataset": "OCTA500-6M",
        "model": "swinunetr",
        "outdir": str(tmp_path / "runs"),
        "exp_id": "smoke-test-001",
    }
    run_dir = setup_run_dir(cfg)
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "git_commit.txt").exists()
    assert (run_dir / "ckpt").is_dir()
    assert (run_dir / "pred_vis").is_dir()


# ─── Dataset smoke ───────────────────────────────────────────────────────────

def test_octa500_dataset_smoke(tmp_path):
    from src.datasets.octa500 import OCTA500Dataset
    root = tmp_path / "OCTA500_6M"
    for split in ("train", "val", "test"):
        _make_dummy_split(root, split, n=4, size=80)

    ds = OCTA500Dataset(str(root), split="train", image_size=64)
    assert len(ds) == 4
    img, mask = ds[0]
    assert img.shape == (1, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert mask.min() >= 0.0 and mask.max() <= 1.0


# ─── Full mini train loop ─────────────────────────────────────────────────────

def test_mini_train_loop(tmp_path):
    """2-step training loop with UNet++ and dummy OCTA-style data."""
    from torch.utils.data import DataLoader

    from src.datasets.octa500 import OCTA500Dataset
    from src.losses import build_loss
    from src.models.unetpp import UNetPlusPlus

    root = tmp_path / "OCTA500_6M"
    for split in ("train", "val", "test"):
        _make_dummy_split(root, split, n=4, size=80)

    ds = OCTA500Dataset(str(root), split="train", image_size=64)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    model = UNetPlusPlus(in_channels=1, out_channels=1, base_channels=8, depth=2)
    criterion = build_loss({"loss": "bce_dice"})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step, (imgs, masks) in enumerate(loader):
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        if step >= 1:
            break

    # Save and reload checkpoint
    ckpt_path = tmp_path / "last.pt"
    torch.save(model.state_dict(), ckpt_path)
    model2 = UNetPlusPlus(in_channels=1, out_channels=1, base_channels=8, depth=2)
    model2.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
