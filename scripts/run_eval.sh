#!/usr/bin/env bash
# ============================================================
#  Quick-start evaluation script
#  Usage: bash scripts/run_eval.sh [DATASET] [MODEL] [EXP_ID] [CKPT]
#
#  Example:
#    bash scripts/run_eval.sh OCTA500-6M swinunetr OCTA6M-B0-SwinUNETR \
#         runs/OCTA500-6M/swinunetr/OCTA6M-B0-SwinUNETR/ckpt/best.pt
# ============================================================
set -euo pipefail

DATASET="${1:-OCTA500-6M}"
MODEL="${2:-swinunetr}"
EXP_ID="${3:-}"
CKPT="${4:-runs/$DATASET/$MODEL/$EXP_ID/ckpt/best.pt}"
CONFIG="configs/defaults.yaml"
OUTDIR="runs"
DATA_ROOT="/data1/suhohan/JiT/data"

cd "$(dirname "$0")/.."

CMD="python eval.py \
  --config    $CONFIG \
  --dataset   $DATASET \
  --model     $MODEL \
  --ckpt      $CKPT \
  --data_root $DATA_ROOT \
  --outdir    $OUTDIR"

if [ -n "$EXP_ID" ]; then
  CMD="$CMD --exp_id $EXP_ID"
fi

echo "[run_eval.sh] Running: $CMD"
eval $CMD
