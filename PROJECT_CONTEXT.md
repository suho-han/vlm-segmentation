# PROJECT_CONTEXT.md — Compressed Status (2026-02-27)

---

## [Project Objectives]

Improve boundary accuracy (hd95) and topological connectivity (Betti β0/β1) beyond Dice/IoU by injecting Vision-Language Model (VLM) feature priors into 2D vessel segmentation (OCTA500-6M, DRIVE).

Core Hypotheses:

1. Injecting semantic features from frozen VLM image encoders into the decoder → Reduces topological errors.
2. Gated fusion (1×1 Conv + Sigmoid) → Enhances boundary accuracy (hd95) compared to static fusion.
3. Predicting QC scores using VLM embeddings + internal segmentation features → Detects bad cases.

---

## [Environment]

- Path: `/data1/suhohan/vlm-segmentation`
- venv: `.venv/` (Python 3.11, torch 2.10.0+cu128, monai 1.5.2, einops)
- Package Management: `uv` exclusive
- **Note**: `VIRTUAL_ENV=/data1/suhohan/JiT/.venv` is set in the shell.
  → Always use `env -u VIRTUAL_ENV uv run ...`
  → `uv pip install` requires `--python .venv/bin/python`

---

## [Project Structure]

```
vlm-segmentation/
├── src/
│   ├── datasets/   octa500.py, drive.py, dummy.py, discovery.py, transforms.py
│   ├── models/     swinunetr.py, unetpp.py
│   ├── losses/     dice.py, bce.py
│   ├── metrics/    dice_iou.py, hd95.py, topology.py
│   └── utils/      exp.py, seed.py, io.py
├── configs/
│   ├── defaults.yaml
│   └── exp_cards/  OCTA6M-B{0,1}-*.yaml, DRIVE-B{0,1}-*.yaml
├── train.py, eval.py
├── tests/          test_metrics.py, test_smoke_train.py  (19 passed)
├── docs/EXPERIMENTS.md
└── runs/           (gitignored, exists only locally)
```

---

## [Runs Output Contract]

`runs/{dataset}/{model}/{exp_id}/`

- `config.yaml`, `git_commit.txt`, `metrics.json`, `train_history.json`
- `ckpt/best.pt`, `ckpt/last.pt`
- `pred_vis/` (20 images)

---

## [Metrics Keys (LOCK)]

Required keys in `metrics.json`: `Dice`, `IoU`, `hd95`, `betti_beta0`, `betti_beta1`

- hd95: Empty pred/gt → `inf`
- betti_beta0/1: `|pred - gt|` (skimage proxy)

---

## [Data]

| Dataset | Path | Format | Split |
|---------|------|------|------|
| OCTA500-6M | `./data/OCTA500_6M/{train,val,test}/{images,labels}/` | *.bmp | 180/20/100 |
| DRIVE | `./data/DRIVE/{train,test}/{images,masks}/` | img=*.tif, mask=*.gif | 20/20 (val=train 80/20, seed=42) |

---

## [Model Constraints]

- SwinUNETR (monai 1.5.2): `img_size` % 32 == 0 → OCTA=384, DRIVE=512
- `spatial_dims=2`, no `img_size` argument (changed in monai≥1.4)
- UNet++: `in_ch = nb_filter[i] * j + nb_filter[i+1]` (j skips + 1 upsample)

---

## [Baseline Results (Completed 2026-02-27)]

### DRIVE

| exp_id | Dice | IoU | hd95 | β0 err | β1 err | best_val |
|--------|------|-----|------|--------|--------|---------|
| DRIVE-B0-SwinUNETR | 0.7746 | 0.6324 | 8.94 | 38.95 | 21.15 | 0.7802 |
| DRIVE-B1-UNetPP    | 0.7988 | 0.6652 | 4.36 | 41.15 | 15.25 | 0.8071 |

### OCTA500-6M

| exp_id | Dice | IoU | hd95 | β0 err | β1 err | best_val |
|--------|------|-----|------|--------|--------|---------|
| OCTA6M-B0-SwinUNETR | 0.8454 | 0.7329 | 1.90 | 27.77 | 5.66 | 0.8390 |
| OCTA6M-B1-UNetPP    | 0.8864 | 0.7969 | 1.83 | 25.90 | 6.15 | 0.8814 |

Config: epochs=100, Adam lr=1e-4, Dice+BCE loss. DRIVE batch=2, OCTA batch=4.

---

## [Git]

```
d14f0ff  Initial commit: 2D vessel segmentation framework
1ce6be8  Remove runs/ from version control
```

- `runs/`, `.venv/`, `data/`, `CLAUDE.md`, `GEMINI.md`, `PROJECT_CONTEXT.md`, `.github/copilot-instructions.md` → gitignore

---

## [Paper Knowledge Base (2026-02-27)]

**Status**: 46 papers collected (`papers/index.csv`, `papers/cards/`)
**Core Categories**:

- **Open-Vocabulary Segmentation (12)**: LSeg, OVSeg, CAT-Seg (emphasizes cost aggregation)
- **Feature Injection / Adapter (11)**: SAN (Side Adapter), DenseCLIP (Language prior), SAM-Adapter
- **Medical VLM (7)**: Med-SAM Adapter, BiomedCLIP, MedSAM3 (2025), Medical SAM3 (2026)
- **Promptable (7)**: SAM, HQ-SAM (Boundary improvement), Grounded-SAM

**Key Insights (Refer to `papers/direction/`)**:

1. **Feature Prior**: SAN (Side Adapter) method is advantageous for injecting vessel-specific knowledge while preserving frozen foundation features.
2. **Medical Prior**: BiomedCLIP (trained on PMC-15M) provides a stronger medical semantic prior compared to generic CLIP.
3. **Boundary/Topology**: HQ-SAM's Global-Local fusion and CAT-Seg's Cost Aggregation are key techniques for improving micro-vessel connectivity.
4. **Gap**: Very few VLM-based models directly optimize topology (connectivity) or use it for QC → A key differentiator for this project.

---

## [Next Steps: VLM Feature Prior]

### Implementation Plan (In Progress)

**Goal**: Inject frozen VLM features into the SwinUNETR decoder using gating (BiomedCLIP prioritized).

**New Files**:

```
src/models/vlm_prior.py          # BiomedCLIP/CLIP wrapper
src/models/swinunetr_vlm.py      # Side-Adapter or Gated Fusion structure
configs/exp_cards/OCTA6M-V0-SwinUNETR-VLM.yaml
configs/exp_cards/DRIVE-V0-SwinUNETR-VLM.yaml
tests/test_vlm_prior.py
```

**Modified Files**:

```
src/models/__init__.py  → Add "swinunetr_vlm" to build_model
train.py                → Pass text_embed to model
eval.py                 → Same as above
```

**VLM Selection** (TBD):

- Priority 1: BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — Medical specific, 512-dim
- Fallback: CLIP ViT-B/16 (OpenAI) — General purpose, 512-dim
- Installation: `uv pip install open_clip_torch --python .venv/bin/python`

**Injection Method**: Decoder channel-wise gating (4 scales)

```
text_embed (512d) → Linear(512, feat_dim) → Sigmoid gate
decoder_feat = decoder_feat * (1 + gate)   # residual, safe init
```

- 4 decoder levels: feat_dim = [48, 96, 192, 384]

**Fixed Text Prompts**:

```python
PROMPTS = {
    "OCTA500-6M": "retinal vessel segmentation in OCTA fundus image",
    "DRIVE":      "retinal blood vessel segmentation in fundus photograph",
}
```

**Accessing SwinUNETR Intermediate Features**:
Can directly access MONAI SwinUNETR's `swinViT`, `encoder1~4`, `encoder10`, and `decoder1~5` in `swinunetr_vlm.py` via step-by-step calls.

**Experiment Naming**:

- `OCTA6M-V0-SwinUNETR-VLM`
- `DRIVE-V0-SwinUNETR-VLM`

**Validation Sequence**:

1. `pytest tests/test_vlm_prior.py`
2. Dummy smoke test: `--dummy --model swinunetr_vlm --epochs 2`
3. Actual training (epochs=100)
4. Compare metrics.json: V0 vs B0 (SwinUNETR baseline)

---

## [Key Commands]

```bash
# Testing
env -u VIRTUAL_ENV uv run pytest -q

# Dummy training
env -u VIRTUAL_ENV uv run python train.py --dummy --model unetpp --exp_id smoke --epochs 2 --outdir runs

# Actual training (Example)
env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-B0-SwinUNETR.yaml \
    --exp_id OCTA6M-B0-SwinUNETR --data_root ./data --outdir runs

# Evaluation
env -u VIRTUAL_ENV uv run python eval.py \
    --config configs/exp_cards/OCTA6M-B0-SwinUNETR.yaml \
    --exp_id OCTA6M-B0-SwinUNETR \
    --ckpt runs/OCTA500-6M/swinunetr/OCTA6M-B0-SwinUNETR/ckpt/best.pt \
    --data_root ./data --outdir runs
```
