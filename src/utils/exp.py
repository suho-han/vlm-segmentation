"""
Experiment management: run-dir creation, config snapshot, git hash recording.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

from .io import save_yaml


def _git_hash() -> str:
    """Return short git commit hash, or 'nogit' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "nogit"


def _auto_exp_id(model: str, dataset: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = _git_hash()
    return f"{ts}_{short_hash}"


def setup_run_dir(cfg: dict) -> Path:
    """
    Create and return the run directory for this experiment.

    Directory layout:
        <outdir>/<dataset>/<model>/<exp_id>/

    Also creates:
        config.yaml     — snapshot of cfg
        git_commit.txt  — current git hash
        ckpt/           — empty checkpoint dir
        pred_vis/       — empty visualisation dir

    Args:
        cfg: Experiment config dict (must contain 'dataset', 'model', 'outdir').

    Returns:
        Path to the run directory.
    """
    dataset = cfg.get("dataset", "unknown")
    model = cfg.get("model", "unknown")
    outdir = cfg.get("outdir", "runs")
    exp_id = cfg.get("exp_id") or _auto_exp_id(model, dataset)

    # Store resolved exp_id back into cfg so callers can access it
    cfg["exp_id"] = exp_id

    run_dir = Path(outdir) / dataset / model / exp_id
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir / "pred_vis").mkdir(parents=True, exist_ok=True)

    # Config snapshot
    save_yaml(dict(cfg), run_dir / "config.yaml")

    # Git hash
    (run_dir / "git_commit.txt").write_text(_git_hash() + "\n")

    return run_dir
