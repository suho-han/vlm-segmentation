# [Key Commands]

```bash
# Testing
env -u VIRTUAL_ENV uv run pytest -q

# Full Reproduction (1000 epochs, early stopping, all models)
# Runs DRIVE on GPU 0 and OCTA on GPU 1 in parallel
bash scripts/run_reproduce_all.sh

# Individual Training Example (with new defaults)
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml \
    --exp_id OCTA6M-V1-SwinUNETR-VLM \
    --epochs 1000 --patience 50
```
