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
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py 
    --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml 
    --exp_id OCTA6M-V1-SwinUNETR-VLM --data_root ./data --outdir runs

# DRIVE
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py 
    --config configs/exp_cards/DRIVE-V1-SwinUNETR-VLM.yaml 
    --exp_id DRIVE-V1-SwinUNETR-VLM --data_root ./data --outdir runs
```

### V1 Result (SwinUNETR-VLM)

| exp_id                  | Dice   | IoU    | hd95 | β0 err | β1 err | best_val |
| ----------------------- | ------ | ------ | ---- | ------ | ------ | -------- |
| DRIVE-V1-SwinUNETR-VLM  | 0.7764 | 0.6349 | 8.96 | 42.10  | 19.80  | 0.7814   |
| OCTA6M-V1-SwinUNETR-VLM | 0.8456 | 0.7331 | 1.87 | 29.69  | 5.46   | 0.8402   |

### V1 Observations:
- Spatial injection (Image-branch) shows slight Dice/IoU improvement over V0/B0 on both datasets.
- **DRIVE:** hd95 regressed back to B0 levels (8.96), but β1 error (holes) is at its lowest (19.80).
- **OCTA:** Achieved lowest hd95 (1.87) among SwinUNETR variants.

---

### V2 Implementation: COMPLETE (2026-03-03)

**Objective:** Direct optimization for micro-vessel connectivity and stable injection-layer convergence.

- **Topology-aware Loss:** `SoftclDiceLoss` implemented (soft-skeletonization via differentiable erosion/dilation). Integrated into `TopologyAwareLoss` alongside BCE and Dice.
- **Optimizer LR Split:** `build_optimizer` implemented to detect VLM-related modules (`gates`, `vlm_proj`, etc.) and apply a higher learning rate (10x base LR) for faster semantic feature adaptation.
- **Resulting Experiments:** 
  - `ISIC2018-V1-Topology` (GPU 0)
  - `OCTA6M-V1-Topology` (GPU 0)
  - `DRIVE-V1-Topology` (GPU 0)
  - `MoNuSeg-V1-Topology` (GPU 0)

