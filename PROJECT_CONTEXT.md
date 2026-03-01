# Project Objectives

Improve boundary accuracy (hd95) and topological connectivity (Betti β0/β1) beyond Dice/IoU by injecting Vision-Language Model (VLM) feature priors into 2D vessel segmentation (OCTA500-6M, DRIVE).

**Core Hypotheses:**

1. Injecting semantic features from frozen VLM image encoders into the decoder → Reduces topological errors.
2. Gated fusion (1×1 Conv + Sigmoid) → Enhances boundary accuracy (hd95) compared to static fusion.
3. Predicting QC scores using VLM embeddings + internal segmentation features → Detects bad cases.

## Environment

- **Path:** `/data1/suhohan/vlm-segmentation`
- **venv:** `.venv/` (Python 3.11, torch 2.10.0+cu128, monai 1.5.2, einops, open_clip_torch 3.3.0, transformers)
- **Package Management:** `uv` exclusive
- **Note:** `VIRTUAL_ENV=/data1/suhohan/JiT/.venv` is set in the shell.
  - → Always use `env -u VIRTUAL_ENV uv run ...`
  - → `uv pip install` requires `--python .venv/bin/python`

### Setup (uv)

```bash
uv venv                        # .venv 생성 (Python 3.11)
source .venv/bin/activate

# Install dependencies
uv pip install -e .
uv pip install monai einops    # SwinUNETR deps
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify
uv run pytest -q
```

---

## [GPU Usage Policy]

### Default GPU Assignment

Unless explicitly specified otherwise, **use GPU 0 or 1 (available ones) instead of GPU 2**.

### Execution Rule

All training / evaluation commands must set:

CUDA_VISIBLE_DEVICES=0  # or 1

Example:

```bash
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py ...
```

---

## [Project Structure]

```text
vlm-segmentation/
├── src/
│   ├── datasets/   octa500.py, drive.py, dummy.py, discovery.py, transforms.py
│   ├── models/     swinunetr.py, unetpp.py, vlm_prior.py
│   │               swinunetr_vlm.py (V0), swinunetr_vlm_v1.py (V1)
│   │               nnunet_2d.py (B2), vit_template.py (ViT abstract base)
│   ├── losses/     dice.py, bce.py
│   ├── metrics/    dice_iou.py, hd95.py, topology.py
│   └── utils/      exp.py, seed.py, io.py
├── configs/
│   ├── defaults.yaml
│   └── exp_cards/  OCTA6M-B{0,1,2}-*.yaml, DRIVE-B{0,1,2}-*.yaml
│                   OCTA6M-V0-SwinUNETR-VLM.yaml, DRIVE-V0-SwinUNETR-VLM.yaml
│                   OCTA6M-V1-SwinUNETR-VLM.yaml, DRIVE-V1-SwinUNETR-VLM.yaml
├── train.py, eval.py
├── tests/          test_metrics.py, test_smoke_train.py,
│                   test_vlm_prior.py, test_vlm_image_prior.py,
│                   test_nnunet.py  (71 passed)
└── runs/           (gitignored, exists only locally)
```

---

## [Runs Output Contract]

`runs/{dataset}/{model}/{exp_id}/`

- `config.yaml`, `git_commit.txt`, `metrics.json`, `train_history.json`
- `ckpt/best.pt`, `ckpt/last.pt`
- `pred_vis/` (20 images)

## [Backbone Compare Automation — Structural Metrics Focus]

### Script\

Use `scripts/backbone_compare.py` to automatically aggregate `metrics.json` across runs and produce:

- `runs/backbone_compare/summary.md`
- `runs/backbone_compare/summary.csv`

### Command

```bash
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python scripts/backbone_compare.py \
  --runs_dir runs \
  --out_dir runs/backbone_compare
```

---

## [Metrics Keys (LOCK)]

Required keys in `metrics.json`: `Dice`, `IoU`, `hd95`, `betti_beta0`, `betti_beta1`

| Key           | Description         | Edge Case                     |
| ------------- | ------------------- | ----------------------------- |
| `Dice`        | Sørensen–Dice       | empty pred/gt → ~0 (Laplace)  |
| `IoU`         | Jaccard Index       | empty pred/gt → ~0            |
| `hd95`        | 95th Hausdorff (px) | empty pred or gt → **inf**    |
| `betti_beta0` | \|β0_pred - β0_gt\| | connectivity (skimage label)  |
| `betti_beta1` | \|β1_pred - β1_gt\| | holes (complement components) |

### Betti Proxy Implementation

- **β0:** `skimage.measure.label` 8-connectivity.
- **β1:** complement 1-connectivity components - 1.
- Euler characteristic based proxy (not full persistent homology).

---

## [Data]

### DATA_ROOT Priority

1. **CLI:** `--data_root /path`
2. **Env:** `export DATA_ROOT=/path`
3. **Config:** `configs/defaults.yaml` (`data_root: "./data"`)

### Auto-discovery

Folders under `DATA_ROOT` are matched by keywords:

- **OCTA500-6M:** `OCTA500_6M`, `OCTA500-6M`, `OCTA500`
- **DRIVE:** `DRIVE`, `drive`

### Structure

```text
data/
├── OCTA500_6M/          # img: *.bmp, label: *.bmp
│   ├── train/ images/ + labels/
│   ├── val/   images/ + labels/
│   └── test/  images/ + labels/
└── DRIVE/               # img: *.tif, mask: *.gif
    ├── train/ images/ + masks/
    └── test/  images/ + masks/
```

| Dataset        | Path                                                  | Format                    | Split                              |
| -------------- | ----------------------------------------------------- | ------------------------- | ---------------------------------- |
| **OCTA500-6M** | `./data/OCTA500_6M/{train,val,test}/{images,labels}/` | `*.bmp`                   | 180 / 20 / 100                     |
| **DRIVE**      | `./data/DRIVE/{train,test}/{images,masks}/`           | img=`*.tif`, mask=`*.gif` | 20 / 20 (val=train 80/20, seed=42) |

---

## [Model Constraints]

- **SwinUNETR** (monai 1.5.2): `img_size` % 32 == 0 → OCTA=384, DRIVE=512
  - `spatial_dims=2`, no `img_size` argument (changed in monai≥1.4)
- **UNet++**: `in_ch = nb_filter[i] * j + nb_filter[i+1]` (j skips + 1 upsample)
- **SwinUNETR VLM smoke test**: Must use exp_card config (not `--image_size` flag) since `defaults.yaml` `swinunetr.img_size=400` takes precedence over CLI.

---

## [All Results]

**Config:** epochs=100, AdamW lr=1e-4, Dice+BCE loss. DRIVE batch=2, OCTA batch=4.

### DRIVE

| exp_id                 | Dice   | IoU    | hd95     | β0 err | β1 err    | best_val |
| ---------------------- | ------ | ------ | -------- | ------ | --------- | -------- |
| DRIVE-B0-SwinUNETR     | 0.7746 | 0.6324 | 8.94     | 38.95  | 21.15     | 0.7802   |
| DRIVE-B1-UNetPP        | 0.7988 | 0.6652 | **4.36** | 41.15  | **15.25** | 0.8071   |
| DRIVE-B2-nnUNet        | 0.7585 | 0.6113 | 8.41     | 50.50  | 10.30     | 0.7650   |
| DRIVE-V0-SwinUNETR-VLM | 0.7761 | 0.6344 | 8.14     | 47.75  | 20.90     | 0.7811   |
| DRIVE-V1-SwinUNETR-VLM | 0.7764 | 0.6349 | 8.96     | 42.10  | 19.80     | 0.7814   |

### OCTA500-6M

| exp_id                  | Dice       | IoU        | hd95     | β0 err    | β1 err   | best_val |
| ----------------------- | ---------- | ---------- | -------- | --------- | -------- | -------- |
| OCTA6M-B0-SwinUNETR     | 0.8454     | 0.7329     | 1.90     | 27.77     | 5.66     | 0.8390   |
| OCTA6M-B1-UNetPP        | **0.8864** | **0.7969** | **1.83** | **25.90** | 6.15     | 0.8814   |
| OCTA6M-B2-nnUNet        | 0.8438     | 0.7305     | 1.99     | 29.04     | 6.79     | 0.8381   |
| OCTA6M-V0-SwinUNETR-VLM | 0.8450     | 0.7323     | 1.89     | 27.21     | **5.16** | 0.8393   |
| OCTA6M-V1-SwinUNETR-VLM | 0.8456     | 0.7331     | 1.87     | 29.69     | 5.46     | 0.8402   |

---

## [V0 Analysis: What the Text Feature Actually Does]

**What it is:** A single fixed `(1, 512)` unit-norm vector, computed once per run from BiomedCLIP's text encoder on a fixed string prompt. The same vector is used for every image, every batch, every epoch.

**How it's injected** (`swinunetr_vlm.py` — `_GatedFusion`):

```python
gate = tanh(Linear(512, C)(text_embed))  # (1, C) — zero-init → identity at start
out  = decoder_feat * (1 + gate)         # channel-wise residual scale
```

Applied at 4 decoder levels: `C = [384, 192, 96, 48]` (coarse → fine).

**What the gate learns:** Because the text embed is constant, `tanh(W·text_embed + b)` converges to a **fixed per-channel scalar** — effectively a learned channel attention mask conditioned on the task description ("retinal vessel segmentation"). It is *not* image-specific.

**Why V0 barely moves vs B0:** The text provides only task-level context. No per-image VLM signal. The gate degenerates to a small static channel rescaling with zero image-specific information.

**V0 Observations:**

- **DRIVE hd95:** 8.94 → 8.14 (−0.80) ✓
- **OCTA β1 error:** 5.66 → 5.16 (−0.50) ✓
- **DRIVE β0 error:** 38.95 → 47.75 (+8.80) ✗ — gates need topology-aware supervision
- **OCTA Dice/IoU:** essentially flat — expected for text-only, no image prior

---

## [V1 Implementation: COMPLETE (2026-03-01)]

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

### Pending V1 result rows (fill after training)

| exp_id                  | Dice | IoU | hd95 | β0 err | β1 err | best_val |
| ----------------------- | ---- | --- | ---- | ------ | ------ | -------- |
| DRIVE-V1-SwinUNETR-VLM  | —    | —   | —    | —      | —      | —        |
| OCTA6M-V1-SwinUNETR-VLM | —    | —   | —    | —      | —      | —        |

### After V1: Next Candidates

1. **Topology loss (V2):** Add `betti_beta0 + betti_beta1` auxiliary loss. Works with V1 architecture.
2. **Gate LR scheduling:** 10× LR for injection layers vs backbone.
3. **Backbone expansion:** ViT-based segmenters (TransUNet, SegFormer-B2) for VLM injection comparison study.

---

## [Backbone Expansion: Top-2 Selection & Strategy (2026-03-01)]

### Top-2 ViT Selection (VLM Focus)

1. **UNETR-2D**:
   - **Rationale**: Non-hierarchical ViT-Base encoder aligns perfectly with BiomedCLIP-L/14 patch structure.
   - **Injection**: Token-level addition (`vit_block + alpha * vlm_patch`).
2. **TransUNet**:
   - **Rationale**: Industry standard hybrid CNN-ViT; robust edge detection via ResNet-50 + global connectivity via ViT.
   - **Injection**: Decoder bottleneck gating + skip-connection refinement.

### Implementation Strategy

- **SegFormer-B2 Style**: Use `timm.create_model('pvt_v2_b2')` as the MiT-B2 equivalent encoder.
- **Blueprint Reference**: `papers/direction/backbone_expand/top2_blueprint.md`
- **Metric Targets**:
  - hd95: -10% vs SwinUNETR
  - Betti errors: -15% vs SwinUNETR

### Progress Summary

- **Phase A (Baselines)**: nnU-Net (B2) implementation complete.
- **Phase B (VLM Feasibility)**: UNETR-2D and TransUNet blueprints defined.
- **Phase C (Execution)**: Pending training for `A0` (Baseline) and `V1` (Image-branch) for selected models.

---

## [Paper Knowledge Base (2026-02-27)]

- **Status:** 46 papers collected (`papers/index.csv`, `papers/cards/`)

### Core Categories

- **Open-Vocabulary Segmentation (12):** LSeg, OVSeg, CAT-Seg (emphasizes cost aggregation)
- **Feature Injection / Adapter (11):** SAN (Side Adapter), DenseCLIP (Language prior), SAM-Adapter
- **Medical VLM (7):** Med-SAM Adapter, BiomedCLIP, MedSAM3 (2025), Medical SAM3 (2026)
- **Promptable (7):** SAM, HQ-SAM (Boundary improvement), Grounded-SAM

### Key Insights (See `papers/direction/`)

1. **Feature Prior:** SAN (Side Adapter) method is advantageous for injecting vessel-specific knowledge while preserving frozen foundation features.
2. **Medical Prior:** BiomedCLIP (trained on PMC-15M) provides a stronger medical semantic prior compared to generic CLIP.
3. **Boundary/Topology:** HQ-SAM's Global-Local fusion and CAT-Seg's Cost Aggregation are key techniques for improving micro-vessel connectivity.
4. **Gap:** Very few VLM-based models directly optimize topology (connectivity) or use it for QC → A key differentiator for this project.

---

## [Git]

```text
b7a2186  feat: Add vlm_prior.py model
1abdff5  feat: Add .gitignore to ignore run and data directories
fad83fc  docs: Add PROJECT_CONTEXT.md
d920336  Remove runs/ from version control
5e6792d  Initial commit: 2D vessel segmentation framework
```

- **Ignored paths:** `runs/`, `.venv/`, `data/`, `CLAUDE.md`, `GEMINI.md`, `PROJECT_CONTEXT.md`, `.github/copilot-instructions.md`

---

## [Key Commands]

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

---
