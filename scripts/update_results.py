"""
Scan all runs/*/model/exp_id/metrics.json and regenerate project_context/results.md.
Usage:
    env -u VIRTUAL_ENV uv run python scripts/update_results.py --runs_dir runs
"""

import argparse
import json
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

DATASET_ORDER = ["OCTA500-6M", "DRIVE", "MoNuSeg", "ISIC2018"]

DATASET_NOTES = {
    "OCTA500-6M": "배치=4, 입력=384px, in_channels=1",
    "DRIVE":      "배치=2, 입력=512px, in_channels=3. †B4-TransUNet: pos_embed 불일치로 수렴 불량",
    "MoNuSeg":    "37 train → 80/20 split (29 train / 8 val), 14 test. RGB 1000×1000 → 256×256.",
    "ISIC2018":   "배치=8, 입력=256px, in_channels=3. 2594 train / 100 val / 1000 test.",
}

# Model display order within a dataset
MODEL_PRIORITY = [
    "B0", "B1", "B2", "B3", "B3-VLM", "B4", "B4-VLM",
    "V0", "V1", "V1-Topology", "V3",
]


def _priority(exp_id: str) -> int:
    for i, key in enumerate(MODEL_PRIORITY):
        if key.lower() in exp_id.lower().replace("-swinunetr", "").replace("-unetpp", ""):
            return i
    return 99


def collect_metrics(runs_dir: Path) -> dict:
    """Returns {dataset: [row_dict, ...]} sorted by model priority."""
    results: dict[str, list] = {}

    for metrics_path in sorted(runs_dir.rglob("metrics.json")):
        # path: runs/<dataset>/<model>/<exp_id>/metrics.json
        parts = metrics_path.parts
        if len(parts) < 5:
            continue
        dataset = parts[1]
        exp_id  = parts[3]

        with open(metrics_path) as f:
            data = json.load(f)

        agg = data.get("aggregate", {})
        row = {
            "exp_id":      exp_id,
            "Dice":        agg.get("Dice",         float("nan")),
            "IoU":         agg.get("IoU",          float("nan")),
            "hd95":        agg.get("hd95",         float("nan")),
            "betti_beta0": agg.get("betti_beta0",  float("nan")),
            "betti_beta1": agg.get("betti_beta1",  float("nan")),
        }
        # read epochs / best_val from config.yaml in same dir
        cfg_path = metrics_path.parent / "config.yaml"
        row["epochs"]   = "—"
        row["best_val"] = "—"
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            row["best_val"] = f"{cfg.get('_best_val', '—')}"

        # read epochs from ckpt/best.pt if possible
        ckpt_last = metrics_path.parent / "ckpt" / "last.pt"
        if ckpt_last.exists():
            try:
                import torch
                ck = torch.load(ckpt_last, map_location="cpu", weights_only=False)
                row["epochs"]   = ck.get("epoch", "—")
                row["best_val"] = f"{ck.get('best_val', '—'):.4f}" if isinstance(ck.get("best_val"), float) else ck.get("best_val", "—")
            except Exception:
                pass

        results.setdefault(dataset, []).append(row)

    # sort each dataset by model priority
    for ds in results:
        results[ds].sort(key=lambda r: _priority(r["exp_id"]))

    return results


def _fmt(v, decimals=4):
    if isinstance(v, float):
        if v != v:  # nan
            return "—"
        return f"{v:.{decimals}f}"
    return str(v)


def _bold_best(rows: list, col: str, higher_better: bool = True) -> set:
    vals = [r[col] for r in rows if isinstance(r[col], float) and r[col] == r[col]]
    if not vals:
        return set()
    best = max(vals) if higher_better else min(vals)
    return {r["exp_id"] for r in rows if r[col] == best}


def render_table(rows: list) -> str:
    if not rows:
        return "_No results yet._\n"

    bold_dice  = _bold_best(rows, "Dice",        higher_better=True)
    bold_hd95  = _bold_best(rows, "hd95",        higher_better=False)
    bold_b0    = _bold_best(rows, "betti_beta0",  higher_better=False)
    bold_b1    = _bold_best(rows, "betti_beta1",  higher_better=False)

    header = ("| exp_id | Dice | IoU | hd95 | β0 err | β1 err | epochs | best_val |\n"
              "| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = [header]
    for r in rows:
        def b(v, fmt, is_bold):
            s = fmt(v)
            return f"**{s}**" if is_bold else s

        lines.append(
            f"| {r['exp_id']} "
            f"| {b(r['Dice'],        lambda v: _fmt(v,4), r['exp_id'] in bold_dice)} "
            f"| {_fmt(r['IoU'],4)} "
            f"| {b(r['hd95'],        lambda v: _fmt(v,3), r['exp_id'] in bold_hd95)} "
            f"| {b(r['betti_beta0'], lambda v: _fmt(v,2), r['exp_id'] in bold_b0)} "
            f"| {b(r['betti_beta1'], lambda v: _fmt(v,2), r['exp_id'] in bold_b1)} "
            f"| {r['epochs']} "
            f"| {r['best_val']} |"
        )
    return "\n".join(lines) + "\n"


def render_summary_table(results: dict) -> str:
    """Cross-dataset Dice summary."""
    all_exp_ids: set[str] = set()
    for rows in results.values():
        for r in rows:
            all_exp_ids.add(r["exp_id"])

    # Build a label → row mapping per dataset
    def dice_for(ds, exp):
        for r in results.get(ds, []):
            if r["exp_id"] == exp:
                return _fmt(r["Dice"], 4)
        return "—"

    # Representative exp_ids across all datasets
    all_sorted = sorted(all_exp_ids, key=_priority)
    datasets = [d for d in DATASET_ORDER if d in results or d == "ISIC2018"]

    header_cols = " | ".join(datasets)
    sep_cols    = " | ".join(["---"] * len(datasets))
    lines = [
        f"| 모델 | {header_cols} |",
        f"| --- | {sep_cols} |",
    ]
    for exp in all_sorted:
        cols = " | ".join(dice_for(ds, exp) for ds in datasets)
        lines.append(f"| {exp} | {cols} |")
    return "\n".join(lines) + "\n"


def write_results_md(results: dict, out_path: Path):
    lines = [
        "# [All Results]",
        "",
        "**학습 설정:** epochs=1000 (max), early_stopping=val_loss (patience=50), AdamW lr=1e-4, Dice+BCE loss, seed=42  ",
        "**배치 크기:** OCTA=4, DRIVE=2, MoNuSeg=8, ISIC2018=8  ",
        "**입력 크기:** OCTA=384px, DRIVE=512px, MoNuSeg=256px, ISIC2018=256px",
        "",
        "---",
        "",
    ]

    for ds in DATASET_ORDER:
        rows = results.get(ds, [])
        lines.append(f"## {ds}")
        lines.append("")
        if ds in DATASET_NOTES:
            lines.append(DATASET_NOTES[ds])
            lines.append("")
        lines.append(render_table(rows))
        lines.append("")
        lines.append("---")
        lines.append("")

    lines += [
        "## 종합 비교 (Dice)",
        "",
        render_summary_table(results),
        "",
        "---",
        "",
        "## 완료된 주요 과제 (Phase C Milestone)",
        "",
        "- **[C1]** SegFormer/TransUNet VLM V1: `use_vlm` flag로 spatial injection 구현 완료.",
        "- **[C2]** TransUNet 512px pos_embed 수정: `_resize_pos_embed()` 적용 완료.",
        "- **[C3]** Topology-aware loss (SoftclDice): OCTA6M/DRIVE/MoNuSeg/ISIC2018 실험 완료.",
        "- **[C4]** LR split (backbone 1e-4 / injection 1e-3): `build_optimizer` param groups 완료.",
        "",
        "---",
        "",
        "## 시각화",
        "",
        "```bash",
        "env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset OCTA500-6M",
        "env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset DRIVE",
        "env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset MoNuSeg",
        "env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset ISIC2018",
        "```",
        "",
    ]

    out_path.write_text("\n".join(lines))
    print(f"Written: {out_path}  ({len(rows)} rows last dataset)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out",      default="project_context/results.md")
    args = p.parse_args()

    results = collect_metrics(Path(args.runs_dir))
    print("Datasets found:", list(results.keys()))
    for ds, rows in results.items():
        print(f"  {ds}: {len(rows)} experiments")

    write_results_md(results, Path(args.out))


if __name__ == "__main__":
    main()
