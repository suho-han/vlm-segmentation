# [Key Commands]

```bash
# Testing
env -u VIRTUAL_ENV uv run pytest -q

# Baseline Execution (UNETR, TransUNet, nnU-Net)
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py --config configs/exp_cards/OCTA6M-A0-UNETR.yaml --exp_id OCTA6M-A0-UNETR --outdir runs
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py --config configs/exp_cards/OCTA6M-A0-TransUNet.yaml --exp_id OCTA6M-A0-TransUNet --outdir runs
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py --config configs/exp_cards/OCTA6M-B2-nnUNet.yaml --exp_id OCTA6M-B2-nnUNet --outdir runs

# V1 Execution (Selected ViT Backbones)
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py --config configs/exp_cards/OCTA6M-V1-UNETR.yaml --exp_id OCTA6M-V1-UNETR --outdir runs
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py --config configs/exp_cards/OCTA6M-V1-TransUNet.yaml --exp_id OCTA6M-V1-TransUNet --outdir runs

# SwinUNETR-V1 (Reference)
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml --exp_id OCTA6M-V1-SwinUNETR-VLM --outdir runs
```
