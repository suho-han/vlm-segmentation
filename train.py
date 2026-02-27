#!/usr/bin/env python3
"""
Vessel segmentation training entry point.

Usage:
    python train.py --dataset OCTA500-6M --model swinunetr \
                    --config configs/defaults.yaml \
                    --exp_id OCTA6M-B0-SwinUNETR --outdir runs

Any key in the config YAML can be overridden via CLI:
    --batch_size 8 --lr 5e-5 --epochs 50
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# ─── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.datasets import get_loaders
from src.losses import build_loss
from src.metrics.dice_iou import compute_metrics
from src.models import build_model
from src.utils import get_logger, load_yaml, save_json, set_seed, setup_run_dir


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Vessel segmentation trainer")
    p.add_argument("--config", default="configs/defaults.yaml")
    p.add_argument("--dataset", default=None, help="OCTA500-6M | DRIVE | dummy")
    p.add_argument("--model",   default=None, help="swinunetr | unetpp")
    p.add_argument("--exp_id",  default=None)
    p.add_argument("--outdir",  default=None)
    p.add_argument("--data_root", default=None,
                   help="Dataset root (overrides DATA_ROOT env var and config)")
    p.add_argument("--dummy_data", type=int, default=0,
                   help="1 → use dummy random data (no real dataset required)")
    # Common overrides
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--image_size", type=int,   default=None)
    p.add_argument("--amp",        action="store_true", default=None)
    p.add_argument("--no_amp",     action="store_true")
    p.add_argument("--dummy",      action="store_true",
                   help="Use dummy (random) data — no real dataset required")
    return p.parse_args()


def merge_cfg(args) -> dict:
    cfg = load_yaml(args.config)
    # data_root priority: CLI > DATA_ROOT env var > config YAML
    data_root = args.data_root or os.environ.get("DATA_ROOT") or cfg.get("data_root", "./data")
    overrides = {
        "dataset": args.dataset,
        "model": args.model,
        "exp_id": args.exp_id,
        "outdir": args.outdir,
        "data_root": data_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "image_size": args.image_size,
    }
    if args.no_amp:
        overrides["amp"] = False
    elif args.amp:
        overrides["amp"] = True
    if args.dummy or args.dummy_data:
        overrides["dataset"] = "dummy"
        overrides["_dummy"] = True
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ─── training loop ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg):
    model.train()
    total_loss = 0.0
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        grad_clip = cfg.get("grad_clip", 1.0)
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    all_dice, all_iou = [], []
    threshold = cfg.get("threshold", 0.5)
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)
        total_loss += loss.item()
        preds = torch.sigmoid(logits).cpu().numpy()
        tgts = masks.cpu().numpy()
        for p, t in zip(preds, tgts):
            m = compute_metrics(p.squeeze(), t.squeeze(), threshold)
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
    import numpy as np
    return {
        "val_loss": total_loss / max(len(loader), 1),
        "val_dice": float(np.mean(all_dice)),
        "val_iou":  float(np.mean(all_iou)),
    }


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = merge_cfg(args)
    logger = get_logger("train")
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    run_dir = setup_run_dir(cfg)
    logger.info(f"Run dir: {run_dir}")

    train_loader, val_loader, _ = get_loaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scaler = GradScaler("cuda", enabled=cfg.get("amp", True) and device.type == "cuda")

    # Scheduler
    epochs = cfg.get("epochs", 100)
    sched_name = cfg.get("scheduler", "cosine")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    best_dice = -1.0
    history = []
    val_every = cfg.get("val_every", 1)
    save_every = cfg.get("save_every", 10)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, cfg
        )
        if scheduler:
            scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss}

        if epoch % val_every == 0:
            val_metrics = validate(model, val_loader, criterion, device, cfg)
            row.update(val_metrics)
            logger.info(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_metrics['val_loss']:.4f}  "
                f"val_dice={val_metrics['val_dice']:.4f}  "
                f"val_iou={val_metrics['val_iou']:.4f}"
            )
            if val_metrics["val_dice"] > best_dice:
                best_dice = val_metrics["val_dice"]
                torch.save(model.state_dict(), run_dir / "ckpt" / "best.pt")
                logger.info(f"  → new best dice={best_dice:.4f}, saved best.pt")
        else:
            logger.info(f"Epoch {epoch:03d}/{epochs}  train_loss={train_loss:.4f}")

        if epoch % save_every == 0:
            torch.save(model.state_dict(), run_dir / "ckpt" / f"epoch_{epoch:04d}.pt")

        torch.save(model.state_dict(), run_dir / "ckpt" / "last.pt")
        history.append(row)

    save_json({"history": history, "best_val_dice": best_dice}, run_dir / "train_history.json")
    logger.info(f"Training complete. Best val Dice: {best_dice:.4f}")
    logger.info(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
