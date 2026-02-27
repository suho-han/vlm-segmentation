#!/usr/bin/env bash
# ============================================================
#  Quick-start training script
#  Usage: bash scripts/run_train.sh [DATASET] [MODEL] [EXP_ID]
#
#  Examples:
#    bash scripts/run_train.sh OCTA500-6M swinunetr OCTA6M-B0-SwinUNETR
#    bash scripts/run_train.sh DRIVE       unetpp    DRIVE-B0-UNetpp
# ============================================================
set -euo pipefail

DATASET="${1:-OCTA500-6M}"
MODEL="${2:-swinunetr}"
EXP_ID="${3:-}"
CONFIG="configs/defaults.yaml"
OUTDIR="runs"
DATA_ROOT="/data1/suhohan/JiT/data"

cd "$(dirname "$0")/.."

CMD="python train.py \
  --config   $CONFIG \
  --dataset  $DATASET \
  --model    $MODEL \
  --data_root $DATA_ROOT \
  --outdir   $OUTDIR"

if [ -n "$EXP_ID" ]; then
  CMD="$CMD --exp_id $EXP_ID"
fi

echo "[run_train.sh] Running: $CMD"
eval $CMD
