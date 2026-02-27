"""
Dataset auto-discovery utilities.

Searches for dataset folders under a root directory using candidate names,
validates expected sub-structure, and writes a debug report on failure.
"""

import os
from pathlib import Path
from typing import Optional

# ─── Candidate folder names per dataset (case-insensitive search) ────────────

_CANDIDATES = {
    "OCTA500-6M":  ["OCTA500_6M", "OCTA500-6M", "OCTA5006M", "OCTA500_6m",
                    "octa500_6m", "OCTA500"],
    "OCTA500-3M":  ["OCTA500_3M", "OCTA500-3M", "OCTA5003M", "octa500_3m"],
    "DRIVE":       ["DRIVE", "drive", "Drive"],
}

# Expected sub-structure to validate a candidate (any of these must exist)
_EXPECTED_SUBDIRS = {
    "OCTA500-6M": [["train", "images"], ["train", "labels"]],
    "OCTA500-3M": [["train", "images"], ["train", "labels"]],
    "DRIVE":      [["train", "images"], ["train", "masks"],
                   ["test",  "images"], ["test",  "masks"]],
}

# Image/mask extensions to scan
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def find_dataset_root(data_root: str, dataset: str) -> Optional[Path]:
    """
    Search for the dataset folder under data_root.

    Search order:
    1. Exact path: data_root / candidate_name (case-insensitive)
    2. Recursive single-level: any subfolder matching a candidate name

    Returns the Path if found + validated, else None.
    Writes a debug report to runs/_debug/dataset_scan_{dataset}.txt on failure.
    """
    root = Path(data_root)
    candidates = _CANDIDATES.get(dataset, [dataset])
    expected = _EXPECTED_SUBDIRS.get(dataset, [])

    # Step 1: check direct children of data_root
    if root.exists():
        children = {p.name: p for p in root.iterdir() if p.is_dir()}
        children_lower = {k.lower(): v for k, v in children.items()}
        for cand in candidates:
            found = children.get(cand) or children_lower.get(cand.lower())
            if found and _validate(found, expected):
                return found

        # Step 2: one level deeper (data_root/*/<candidate>)
        for subdir in root.iterdir():
            if not subdir.is_dir():
                continue
            sub_children = {p.name: p for p in subdir.iterdir() if p.is_dir()}
            sub_lower = {k.lower(): v for k, v in sub_children.items()}
            for cand in candidates:
                found = sub_children.get(cand) or sub_lower.get(cand.lower())
                if found and _validate(found, expected):
                    return found

    # Discovery failed → write debug report
    _write_debug_report(data_root, dataset, candidates, expected)
    return None


def _validate(path: Path, expected_subpaths: list) -> bool:
    """Return True if at least one expected sub-path exists under path."""
    if not expected_subpaths:
        return path.exists()
    for subpath in expected_subpaths:
        check = path
        for part in subpath:
            check = check / part
        if check.exists():
            return True
    return False


def _write_debug_report(data_root: str, dataset: str,
                        candidates: list, expected: list):
    """Write a debug scan report so the user can diagnose discovery failures."""
    report_dir = Path("runs/_debug")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"dataset_scan_{dataset.replace('/', '_')}.txt"

    root = Path(data_root)
    lines = [
        f"Dataset:    {dataset}",
        f"Data root:  {data_root}  (exists={root.exists()})",
        f"Candidates: {candidates}",
        f"Expected sub-paths: {expected}",
        "",
    ]

    if root.exists():
        lines.append("── Top-level directories under data_root ──")
        for p in sorted(root.iterdir()):
            marker = "DIR" if p.is_dir() else "FILE"
            lines.append(f"  [{marker}] {p.name}")

        lines.append("")
        lines.append("── Sample image / mask file counts (2 levels deep) ──")
        for lvl1 in sorted(root.iterdir()):
            if not lvl1.is_dir():
                continue
            for lvl2 in sorted(lvl1.iterdir()):
                if not lvl2.is_dir():
                    continue
                imgs = [f for f in lvl2.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS]
                if imgs:
                    exts = {f.suffix.lower() for f in imgs}
                    lines.append(
                        f"  {lvl1.name}/{lvl2.name}/  "
                        f"({len(imgs)} files, exts={sorted(exts)})"
                    )
                    for fname in sorted(imgs)[:10]:
                        lines.append(f"    {fname.name}")
    else:
        lines.append(f"  data_root does not exist: {data_root}")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"[DATASET DISCOVERY] Failed to locate '{dataset}' under '{data_root}'. "
          f"Debug report: {report_path}")


def scan_image_mask_pairs(images_dir: Path, masks_dir: Path,
                          image_exts=None, mask_exts=None) -> list:
    """
    Pair image files to mask files by stem matching.

    Pairing rules (in order):
    1. Exact stem match (same filename without extension)
    2. Mask stem = image_stem + suffix in {_mask, _gt, _label, -mask, -gt, -label}
    3. Image stem = mask_stem + suffix (reverse)

    Returns list of (image_path, mask_path) tuples (both as Path).
    """
    if image_exts is None:
        image_exts = IMAGE_EXTS
    if mask_exts is None:
        mask_exts = IMAGE_EXTS

    img_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in image_exts]
    )
    mask_map: dict[str, Path] = {}
    for mp in masks_dir.iterdir():
        if mp.suffix.lower() in mask_exts:
            mask_map[mp.stem] = mp

    MASK_SUFFIXES = ("_mask", "_gt", "_label", "-mask", "-gt", "-label")

    pairs = []
    for img in img_files:
        stem = img.stem
        # Rule 1: exact stem
        if stem in mask_map:
            pairs.append((img, mask_map[stem]))
            continue
        # Rule 2: mask = stem + suffix
        matched = False
        for suf in MASK_SUFFIXES:
            if (stem + suf) in mask_map:
                pairs.append((img, mask_map[stem + suf]))
                matched = True
                break
        if matched:
            continue
        # Rule 3: stem = mask_stem + suffix (strip suffix from image stem)
        for suf in MASK_SUFFIXES:
            if stem.endswith(suf) and stem[:-len(suf)] in mask_map:
                pairs.append((img, mask_map[stem[:-len(suf)]]))
                matched = True
                break
        # If no match found, skip silently (will be caught by caller)

    return pairs
