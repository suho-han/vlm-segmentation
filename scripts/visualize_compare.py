"""
Visualize model comparison across all available pred_vis directories.

Sample selection:
  1. Find the best model by aggregate Dice (across all datasets or per dataset).
  2. Within that best model, find the sample index with the highest per-sample Dice.
  3. Show that same index for every model that has pred_vis images.

Layout:
  Rows  = one per (dataset, model) combination
  Cols  = [Label | Input | GT | Prediction]

Usage:
  # All datasets, auto-select best sample
  env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs

  # Filter to one dataset
  env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset DRIVE

  # Fix a specific sample index
  env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --sample_idx 3

  # Custom output path
  env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --output compare.png
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ── display order for model names ────────────────────────────────────────────
MODEL_ORDER = [
    "swinunetr",
    "unetpp",
    "nnunet_2d",
    "segformer_b2",
    "transunet",
    "swinunetr_vlm",
    "swinunetr_vlm_v1",
]

MODEL_LABELS = {
    "swinunetr":        "B0  SwinUNETR",
    "unetpp":           "B1  UNet++",
    "nnunet_2d":        "B2  nnU-Net",
    "segformer_b2":     "B3  SegFormer",
    "transunet":        "B4  TransUNet",
    "swinunetr_vlm":    "V0  SwinUNETR-VLM",
    "swinunetr_vlm_v1": "V1  SwinUNETR-VLM (img)",
}

PANEL_TITLES = ["Input", "Ground Truth", "Prediction"]


# ── helpers ───────────────────────────────────────────────────────────────────

def discover_experiments(runs_dir: Path, dataset_filter: str | None):
    """Return list of dicts with keys: dataset, model, exp_id, pred_vis_dir, metrics_path."""
    experiments = []
    for dataset_dir in sorted(runs_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for exp_dir in sorted(model_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                pred_vis = exp_dir / "pred_vis"
                metrics_path = exp_dir / "metrics.json"
                if not pred_vis.is_dir():
                    continue
                images = sorted(pred_vis.glob("pred_*.png"))
                if not images:
                    continue
                experiments.append(dict(
                    dataset=dataset_dir.name,
                    model=model_dir.name,
                    exp_id=exp_dir.name,
                    pred_vis_dir=pred_vis,
                    metrics_path=metrics_path,
                    images=images,
                ))
    return experiments


def load_metrics(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def best_sample_index(experiments: list[dict]) -> int:
    """
    Find the sample index with the highest Dice score inside the
    experiment that has the highest aggregate Dice.
    """
    best_agg = -1.0
    best_exp = None

    for exp in experiments:
        m = load_metrics(exp["metrics_path"])
        if m is None:
            continue
        agg_dice = m.get("aggregate", {}).get("Dice", -1.0)
        if agg_dice > best_agg:
            best_agg = agg_dice
            best_exp = (exp, m)

    if best_exp is None:
        return 0

    exp, m = best_exp
    per_sample = m.get("per_sample", [])
    n_imgs = len(exp["images"])

    if not per_sample:
        return 0

    # Only consider indices that have a matching image file
    valid = [(i, s.get("Dice", 0.0)) for i, s in enumerate(per_sample) if i < n_imgs]
    if not valid:
        return 0

    best_idx = max(valid, key=lambda t: t[1])[0]
    print(
        f"[sample selection] best model: {exp['exp_id']} "
        f"(agg Dice={best_agg:.4f}), "
        f"best sample index: {best_idx} "
        f"(Dice={per_sample[best_idx].get('Dice', 0):.4f})"
    )
    return best_idx


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def split_panels(img: Image.Image) -> list[Image.Image]:
    """Split a 3-panel side-by-side image into [input, gt, pred]."""
    w, h = img.size
    pw = w // 3
    panels = [img.crop((i * pw, 0, (i + 1) * pw, h)) for i in range(3)]
    return panels


def fit_font(draw: ImageDraw.Draw, text: str, max_px: int, base_size: int, min_size: int = 11):
    """Return (font, rendered_text) shrinking font until text fits max_px."""
    for size in range(base_size, min_size - 1, -1):
        font = load_font(size)
        if draw.textlength(text, font=font) <= max_px:
            return font, text
    # Last resort: hard-truncate at min size
    font = load_font(min_size)
    ellipsis = "..."
    t = text
    while t and draw.textlength(t + ellipsis, font=font) > max_px:
        t = t[:-1]
    return font, t + ellipsis


def make_label_image(
    text_lines: list[str],
    width: int,
    height: int,
    bg_color=(20, 20, 20),
    font_size: int = 19,
) -> Image.Image:
    label = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(label)
    pad_x = 14
    max_text_px = width - pad_x * 2

    # Pre-compute per-line font (auto-shrink if needed)
    fitted = [fit_font(draw, line, max_text_px, font_size) for line in text_lines]

    line_h = font_size + 10  # row height based on the base size for uniform spacing
    total_h = len(text_lines) * line_h
    y = (height - total_h) // 2
    for (font, safe_line), line in zip(fitted, text_lines):
        # Colour-code metric lines slightly dimmer
        color = (255, 255, 255) if line.startswith(("B", "V")) else (200, 200, 200)
        draw.text((pad_x, y), safe_line, fill=color, font=font)
        y += line_h
    return label


def make_caption(
    text_lines: list[str],
    width: int,
    height: int,
    bg_color=(30, 30, 30),
    font_size: int = 17,
) -> Image.Image:
    """Caption bar below a panel — lines centred horizontally."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    pad_x = 8
    max_px = width - pad_x * 2
    line_h = font_size + 6
    total_h = len(text_lines) * line_h
    y = (height - total_h) // 2
    for line in text_lines:
        font, safe = fit_font(draw, line, max_px, font_size)
        tw = draw.textlength(safe, font=font)
        draw.text(((width - tw) // 2, y), safe, fill=(220, 220, 220), font=font)
        y += line_h
    return img


def make_ds_banner(text: str, width: int, height: int = 34) -> Image.Image:
    """Full-width dataset name banner between dataset groups."""
    img = Image.new("RGB", (width, height), (50, 50, 50))
    draw = ImageDraw.Draw(img)
    font = load_font(18)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=(230, 230, 230), font=font)
    return img


def build_comparison_image(
    experiments: list[dict],
    sample_idx: int,
    panel_size: tuple[int, int] = (480, 480),
    gap: int = 6,
    caption_height: int = 72,
    ds_banner_height: int = 34,
) -> Image.Image:
    """
    New layout — one strip per dataset:
      Cols : [Input | GT | Pred_M1 | Pred_M2 | ... Pred_Mk]
      Below: captions with model name + Agg Dice + Sample Dice
      Input/GT shown once; only predictions differ per model.
    Multiple datasets stacked vertically with a banner separator.
    """
    pw, ph = panel_size

    def sort_key(exp):
        rank = MODEL_ORDER.index(exp["model"]) if exp["model"] in MODEL_ORDER else 99
        return (exp["dataset"], rank)

    exps_all = sorted(experiments, key=sort_key)

    # Group by dataset (preserve sorted order)
    by_dataset: dict[str, list] = {}
    for exp in exps_all:
        by_dataset.setdefault(exp["dataset"], []).append(exp)

    max_models = max(len(v) for v in by_dataset.values())
    n_cols = 2 + max_models          # input + gt + models
    col_w = pw + gap
    total_w = n_cols * col_w - gap

    # Height: per dataset = banner + (ph + gap + caption_height) + gap_between
    block_h = ph + gap + caption_height
    total_h = (
        len(by_dataset) * (ds_banner_height + gap + block_h)
        - gap   # no trailing gap
    )

    canvas = Image.new("RGB", (total_w, total_h), (10, 10, 10))

    ds_palette = [(45, 85, 140), (130, 65, 50), (50, 120, 65), (110, 55, 130)]
    ds_color_map: dict[str, tuple] = {}

    y_cursor = 0
    for ds_i, (dataset, exps) in enumerate(by_dataset.items()):
        ds_color = ds_palette[ds_i % len(ds_palette)]
        ds_color_map[dataset] = ds_color

        # ── dataset banner ──────────────────────────────────────────────
        banner = make_ds_banner(dataset, total_w, ds_banner_height)
        canvas.paste(banner, (0, y_cursor))
        y_cursor += ds_banner_height + gap

        # ── load input & GT from the first model (they're shared) ───────
        idx = min(sample_idx, len(exps[0]["images"]) - 1)
        ref_img = Image.open(exps[0]["images"][idx]).convert("RGB")
        ref_panels = split_panels(ref_img)   # [input, gt, pred]

        # paste input
        canvas.paste(ref_panels[0].resize((pw, ph), Image.LANCZOS), (0, y_cursor))
        # paste gt
        canvas.paste(ref_panels[1].resize((pw, ph), Image.LANCZOS), (col_w, y_cursor))

        # captions for input / gt
        for col_i, cap_text in enumerate(["Input", "Ground Truth"]):
            cap = make_caption([cap_text], pw, caption_height, bg_color=(25, 25, 25))
            canvas.paste(cap, (col_i * col_w, y_cursor + ph + gap))

        # ── paste each model's prediction ────────────────────────────────
        for m_i, exp in enumerate(exps):
            x0 = (2 + m_i) * col_w
            idx_e = min(sample_idx, len(exp["images"]) - 1)
            exp_img = Image.open(exp["images"][idx_e]).convert("RGB")
            pred_panel = split_panels(exp_img)[2]   # third panel = prediction
            canvas.paste(pred_panel.resize((pw, ph), Image.LANCZOS), (x0, y_cursor))

            # metrics
            m = load_metrics(exp["metrics_path"])
            agg_dice = m["aggregate"]["Dice"] if m else float("nan")
            per_dice = float("nan")
            if m and "per_sample" in m and idx_e < len(m["per_sample"]):
                per_dice = m["per_sample"][idx_e].get("Dice", float("nan"))

            model_label = MODEL_LABELS.get(exp["model"], exp["model"])
            cap_lines = [
                model_label,
                f"Agg  {agg_dice:.4f}",
                f"Dice {per_dice:.4f}",
            ]
            cap = make_caption(cap_lines, pw, caption_height, bg_color=ds_color)
            canvas.paste(cap, (x0, y_cursor + ph + gap))

        y_cursor += block_h + gap

    return canvas


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize model comparison from pred_vis directories.")
    parser.add_argument("--runs_dir", default="runs", help="Root runs directory")
    parser.add_argument("--dataset", default=None, help="Filter to one dataset (e.g. DRIVE, OCTA500-6M)")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Fixed sample index to show (default: auto best)")
    parser.add_argument("--output", default=None, help="Output PNG path (default: auto in runs_dir)")
    parser.add_argument("--panel_size", type=int, default=480, help="Panel width & height in pixels")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        sys.exit(f"ERROR: runs_dir '{runs_dir}' does not exist.")

    print(f"Scanning {runs_dir} ...")
    experiments = discover_experiments(runs_dir, args.dataset)

    if not experiments:
        sys.exit("No pred_vis images found. Run eval.py first to generate visualizations.")

    print(f"Found {len(experiments)} experiment(s) with pred_vis images:")
    for exp in experiments:
        print(f"  [{exp['dataset']}] {exp['model']} / {exp['exp_id']}  "
              f"({len(exp['images'])} images)")

    # Determine sample index
    if args.sample_idx is not None:
        sample_idx = args.sample_idx
        print(f"Using fixed sample index: {sample_idx}")
    else:
        sample_idx = best_sample_index(experiments)

    # Build image
    canvas = build_comparison_image(
        experiments,
        sample_idx=sample_idx,
        panel_size=(args.panel_size, args.panel_size),
    )

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        ds_tag = args.dataset.replace("/", "-") if args.dataset else "all"
        out_path = runs_dir / f"comparison_{ds_tag}_sample{sample_idx:04d}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"\nSaved: {out_path}  ({canvas.size[0]}x{canvas.size[1]} px)")


if __name__ == "__main__":
    main()
