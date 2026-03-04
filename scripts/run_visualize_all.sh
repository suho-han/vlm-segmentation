#!/usr/bin/env bash
# Generate comparison visualizations for all datasets.
# Usage: bash scripts/run_visualize_all.sh [--runs_dir runs] [--panel_size 480]
set -e

RUNS_DIR="runs"
PANEL_SIZE="480"

while [[ $# -gt 0 ]]; do
  case $1 in
    --runs_dir)   RUNS_DIR="$2";   shift 2 ;;
    --panel_size) PANEL_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== Generating OCTA500-6M comparison ==="
env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py \
    --runs_dir "$RUNS_DIR" \
    --dataset OCTA500-6M \
    --panel_size "$PANEL_SIZE" \
    --output "$RUNS_DIR/comparison_OCTA500-6M.png"

echo ""
echo "=== Generating DRIVE comparison ==="
env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py \
    --runs_dir "$RUNS_DIR" \
    --dataset DRIVE \
    --panel_size "$PANEL_SIZE" \
    --output "$RUNS_DIR/comparison_DRIVE.png"

echo ""
echo "Done. Saved:"
echo "  $RUNS_DIR/comparison_OCTA500-6M.png"
echo "  $RUNS_DIR/comparison_DRIVE.png"
