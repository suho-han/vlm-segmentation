#!/usr/bin/env python3
"""
Vessel segmentation evaluation entry point.

Usage:
    uv run python eval.py --dataset OCTA500-6M --model swinunetr \
        --exp_id OCTA6M-B0-SwinUNETR \
        --ckpt runs/OCTA500-6M/swinunetr/OCTA6M-B0-SwinUNETR/ckpt/best.pt \
        --outdir runs --data_root ./data

metrics.json always contains:
    Dice, IoU, hd95, betti_beta0, betti_beta1
    (plus precision, recall, f1, accuracy and per-sample breakdown)

Edge-case rules:
    hd95:         returns float('inf') when either pred or gt is empty
    betti_beta0:  returns 0 when mask is empty (0 connected components)
    betti_beta1:  returns 0 when mask is empty (0 holes)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))
from src.datasets import get_loaders
from src.metrics.dice_iou import compute_metrics
from src.metrics.hd95 import hd95
from src.metrics.topology import betti_error
from src.models import build_model
from src.utils import get_logger, load_yaml, save_json, setup_run_dir


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Vessel segmentation evaluator")
    p.add_argument("--config",    default="configs/defaults.yaml")
    p.add_argument("--dataset",   default=None)
    p.add_argument("--model",     default=None)
    p.add_argument("--ckpt",      required=True, help="Path to checkpoint .pt file")
    p.add_argument("--exp_id",    default=None)
    p.add_argument("--outdir",    default=None)
    p.add_argument("--data_root", default=None,
                   help="Dataset root (overrides DATA_ROOT env var and config)")
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--threshold",  type=float, default=None)
    p.add_argument("--pred_vis_n", type=int, default=None)
    p.add_argument("--no_hd95",    action="store_true",
                   help="Skip HD95 (slow on large images)")
    p.add_argument("--no_topo",    action="store_true",
                   help="Skip topology metrics")
    p.add_argument("--dummy",      action="store_true",
                   help="Use dummy (random) data")
    p.add_argument("--dummy_data", type=int, default=0,
                   help="1 → use dummy random data (same as --dummy)")
    return p.parse_args()


def merge_cfg(args) -> dict:
    cfg = load_yaml(args.config)
    # data_root priority: CLI > DATA_ROOT env var > config YAML
    data_root = args.data_root or os.environ.get("DATA_ROOT") or cfg.get("data_root", "./data")
    overrides = {
        "dataset":    args.dataset,
        "model":      args.model,
        "exp_id":     args.exp_id,
        "outdir":     args.outdir,
        "data_root":  data_root,
        "image_size": args.image_size,
        "threshold":  args.threshold,
        "pred_vis_n": args.pred_vis_n,
    }
    if args.dummy or args.dummy_data:
        overrides["dataset"] = "dummy"
        overrides["_dummy"] = True
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ─── metrics helpers ─────────────────────────────────────────────────────────

def _safe_hd95(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """hd95 with explicit edge-case: empty pred or gt → inf."""
    if not pred_bin.any() or not gt_bin.any():
        return float("inf")
    try:
        return float(hd95(pred_bin, gt_bin))
    except Exception:
        return float("inf")


def _safe_betti(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    """betti_error with explicit edge-case: empty masks → 0 components/holes."""
    try:
        return betti_error(pred_bin, gt_bin)
    except Exception:
        return {"b0_pred": 0, "b0_target": 0, "b0_error": 0,
                "b1_pred": 0, "b1_target": 0, "b1_error": 0}


def _lock_metrics_keys(raw: dict) -> dict:
    """
    Standardise metric keys for metrics.json output contract.

    Required keys: Dice, IoU, hd95, betti_beta0, betti_beta1
    betti_beta0 = mean |b0_pred - b0_target|  (connected-component count error)
    betti_beta1 = mean |b1_pred - b1_target|  (hole count error, Euler proxy)
    """
    return {
        "Dice":         raw.get("dice",    raw.get("Dice",    0.0)),
        "IoU":          raw.get("iou",     raw.get("IoU",     0.0)),
        "precision":    raw.get("precision", 0.0),
        "recall":       raw.get("recall",    0.0),
        "f1":           raw.get("f1",        0.0),
        "accuracy":     raw.get("accuracy",  0.0),
        "hd95":         raw.get("hd95",      float("inf")),
        "betti_beta0":  raw.get("b0_error",  raw.get("betti_beta0", 0.0)),
        "betti_beta1":  raw.get("b1_error",  raw.get("betti_beta1", 0.0)),
        # Keep raw betti sub-fields for debugging
        "b0_pred":      raw.get("b0_pred",   0.0),
        "b0_target":    raw.get("b0_target", 0.0),
        "b1_pred":      raw.get("b1_pred",   0.0),
        "b1_target":    raw.get("b1_target", 0.0),
    }


# ─── pred_vis ─────────────────────────────────────────────────────────────────

def save_prediction_vis(imgs, masks, preds, run_dir: Path, n: int = 20):
    """Save side-by-side visualisations: image | gt | prediction.

    Args:
        imgs, masks, preds: lists of (H, W) float32 numpy arrays.
    """
    try:
        import torchvision.utils as vutils
        vis_dir = run_dir / "pred_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for i, (img, msk, prd) in enumerate(zip(imgs, masks, preds)):
            if saved >= n:
                break
            img_t = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0)
            msk_t = torch.from_numpy(np.asarray(msk, dtype=np.float32)).unsqueeze(0)
            prd_t = torch.from_numpy((np.asarray(prd) >= 0.5).astype(np.float32)).unsqueeze(0)
            grid = torch.stack([img_t, msk_t, prd_t], dim=0)
            vutils.save_image(grid, vis_dir / f"pred_{i:04d}.png", nrow=3)
            saved += 1
    except Exception as e:
        print(f"[WARN] Could not save pred_vis: {e}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = merge_cfg(args)
    logger = get_logger("eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    run_dir = setup_run_dir(cfg)
    logger.info(f"Run dir: {run_dir}")

    _, val_loader, test_loader = get_loaders(cfg)
    model = build_model(cfg).to(device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    threshold  = cfg.get("threshold", 0.5)
    pred_vis_n = cfg.get("pred_vis_n", 20)
    use_amp    = cfg.get("amp", True) and device.type == "cuda"

    all_metrics: list[dict] = []
    all_imgs, all_masks, all_preds = [], [], []

    model.eval()
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast("cuda", enabled=use_amp):
                logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            tgts  = masks.cpu().numpy()

            for p, t in zip(probs, tgts):
                p_sq = p.squeeze()
                t_sq = t.squeeze()
                p_bin = (p_sq >= threshold)
                t_bin = t_sq.astype(bool)

                m = compute_metrics(p_sq, t_sq, threshold)

                if not args.no_hd95:
                    m["hd95"] = _safe_hd95(p_bin, t_bin)

                if not args.no_topo:
                    m.update(_safe_betti(p_bin, t_bin))

                all_metrics.append(_lock_metrics_keys(m))
                all_imgs.append(imgs.cpu().numpy())
                all_masks.append(t[np.newaxis])
                all_preds.append(p[np.newaxis])

    # ── Aggregate (mean over test set; inf-safe for hd95) ──────────────────
    agg: dict[str, float] = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics
                if m.get(key) is not None and m[key] != float("inf")]
        inf_count = sum(1 for m in all_metrics
                        if m.get(key) == float("inf"))
        if vals:
            agg[key] = float(np.mean(vals))
        elif inf_count == len(all_metrics):
            agg[key] = float("inf")

    logger.info("=== Test Results ===")
    for k, v in agg.items():
        logger.info(f"  {k}: {v:.4f}" if v != float("inf") else f"  {k}: inf")

    # Guarantee the 5 required keys are present (fill with sentinel if missing)
    for required in ("Dice", "IoU", "hd95", "betti_beta0", "betti_beta1"):
        if required not in agg:
            agg[required] = float("inf") if required == "hd95" else 0.0

    # Mark dummy runs clearly
    if cfg.get("_dummy"):
        agg["_note"] = "dummy_data"

    save_json({"aggregate": agg, "per_sample": all_metrics},
              run_dir / "metrics.json")
    logger.info(f"Metrics saved to {run_dir / 'metrics.json'}")

    # ── Visualisations ───────────────────────────────────────────────────────
    flat_imgs  = [x.squeeze() for batch in all_imgs  for x in batch]
    flat_masks = [x.squeeze() for x in all_masks]
    flat_preds = [x.squeeze() for x in all_preds]
    save_prediction_vis(flat_imgs, flat_masks, flat_preds, run_dir, n=pred_vis_n)
    logger.info(f"Pred vis saved to {run_dir / 'pred_vis'}")


if __name__ == "__main__":
    main()
