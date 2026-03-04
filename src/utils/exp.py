"""
Experiment management: run-dir creation, config snapshot, git hash recording.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from .io import save_yaml


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Build optimizer with optional learning rate split.
    
    If cfg['lr_injection'] is set, parameters belonging to VLM injection modules
    (gates, img_injectors, vlm_proj) will use lr_injection, while others use lr.
    """
    base_lr = cfg.get("lr", 1e-4)
    injection_lr = cfg.get("lr_injection", base_lr * 10.0 if "vlm" in cfg.get("model", "") else base_lr)
    weight_decay = cfg.get("weight_decay", 1e-5)
    
    # Identify injection parameters
    # Common names across models: 'gates', 'text_gates', 'img_injectors', 'vlm_proj', 'alpha'
    injection_prefixes = ("gates", "text_gates", "img_injectors", "vlm_proj")
    
    injection_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in injection_prefixes) or "alpha" in name:
            injection_params.append(param)
        else:
            base_params.append(param)
            
    param_groups = [
        {"params": base_params, "lr": base_lr},
    ]
    if injection_params:
        param_groups.append({"params": injection_params, "lr": injection_lr})
        
    opt_name = cfg.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


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
