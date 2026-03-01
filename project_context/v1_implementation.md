# [V1 Implementation: COMPLETE (2026-03-01)]

### What was built

- `src/models/swinunetr_vlm_v1.py` — `SwinUNETR2DVLMV1`
  - `_SpatialVLMInjection`: bilinear upsample 14×14 → decoder size, `Conv2d(512→C, 1×1)`, `feat + alpha * proj(vlm_feat)`
  - Zero-init proj (identity at init) + learnable `alpha=0.1` per stage
  - `vlm_mode`: `"image"` (default) | `"text"` | `"both"`
- `src/models/vlm_prior.py` — extended with `get_image_features(x)` → `(B, 512, 14, 14)`
  - BiomedCLIP visual = `TimmModel` → `trunk.forward_features()` + `head` projection
  - Clip model now lives on GPU for per-batch inference
- `configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml`, `DRIVE-V1-SwinUNETR-VLM.yaml`
- `tests/test_vlm_image_prior.py` — 20 new tests (52 total, all pass)
- `train.py` / `eval.py`: `--vlm_mode`, `--vlm_alpha_init` flags; `_vlm_mode()` + `_model_forward()` helpers

### Next: Run V1 experiments

```bash
# OCTA500-6M
CUDA_VISIBLE_DEVICES=2 env -u VIRTUAL_ENV uv run python train.py 
    --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml 
    --exp_id OCTA6M-V1-SwinUNETR-VLM --data_root ./data --outdir runs

# DRIVE
CUDA_VISIBLE_DEVICES=2 env -u VIRTUAL_ENV uv run python train.py 
    --config configs/exp_cards/DRIVE-V1-SwinUNETR-VLM.yaml 
    --exp_id DRIVE-V1-SwinUNETR-VLM --data_root ./data --outdir runs
```

### Pending V1 result rows (fill after training)

| exp_id                  | Dice | IoU | hd95 | β0 err | β1 err | best_val |
| ----------------------- | ---- | --- | ---- | ------ | ------ | -------- |
| DRIVE-V1-SwinUNETR-VLM  | —    | —   | —    | —      | —      | —        |
| OCTA6M-V1-SwinUNETR-VLM | —    | —   | —    | —      | —      | —        |

### After V1: Next Candidates

1. **Topology loss (V2):** Add `betti_beta0 + betti_beta1` auxiliary loss. Works with V1 architecture.
2. **Gate LR scheduling:** 10× LR for injection layers vs backbone.
3. **Backbone expansion:** ViT-based segmenters (TransUNet, SegFormer-B2) for VLM injection comparison study.
