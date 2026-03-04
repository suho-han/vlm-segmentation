#!/bin/bash

# reproduction script: run all experiments with 1000 epochs + early stopping
# usage: bash scripts/run_reproduce_all.sh

DATA_ROOT="./data"
OUT_DIR="runs_repro"
EPOCHS=1000
PATIENCE=50

# list of experiments
DRIVE_EXPS=(
    "DRIVE-B0-SwinUNETR"
    "DRIVE-B1-UNetPP"
    "DRIVE-B2-nnUNet"
    "DRIVE-B3-SegFormer"
    "DRIVE-B4-TransUNet"
    "DRIVE-V0-SwinUNETR-VLM"
    "DRIVE-V1-SwinUNETR-VLM"
    "DRIVE-V1-SwinUNETR-VLM-Topology"
    "DRIVE-B3-SegFormer-VLM"
    "DRIVE-B4-TransUNet-VLM"
)

OCTA_EXPS=(
    "OCTA6M-B0-SwinUNETR"
    "OCTA6M-B1-UNetPP"
    "OCTA6M-B2-nnUNet"
    "OCTA6M-B3-SegFormer"
    "OCTA6M-B4-TransUNet"
    "OCTA6M-V0-SwinUNETR-VLM"
    "OCTA6M-V1-SwinUNETR-VLM"
    "OCTA6M-V1-SwinUNETR-VLM-Topology"
    "OCTA6M-B3-SegFormer-VLM"
    "OCTA6M-B4-TransUNet-VLM"
)

run_exp() {
    local exp_id=$1
    local gpu=$2
    local config="configs/exp_cards/${exp_id}.yaml"
    
    echo "[START] $exp_id on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu env -u VIRTUAL_ENV uv run python train.py \
        --config "$config" \
        --exp_id "$exp_id" \
        --data_root "$DATA_ROOT" \
        --outdir "$OUT_DIR" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE"
    
    echo "[FINISH] $exp_id"
}

# Run DRIVE and OCTA sequentially in parallel on GPU 1
(
    for exp in "${DRIVE_EXPS[@]}"; do
        run_exp "$exp" 1
    done
) > /tmp/repro_drive.log 2>&1 &

(
    for exp in "${OCTA_EXPS[@]}"; do
        run_exp "$exp" 1
    done
) > /tmp/repro_octa.log 2>&1 &

wait
echo "All experiments completed."
