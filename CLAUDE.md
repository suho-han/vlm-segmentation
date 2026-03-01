# CLAUDE.md — Code Agent Instructions (uv + Pure PyTorch)

## Project Context

**Canonical Source of Truth:** Before starting any work, you MUST read and update `project_context/` (split files).
Index: `PROJECT_CONTEXT.md` → links to all sub-files under `project_context/`.

- Task: 2D vessel segmentation (OCTA500-6M, DRIVE)
- Framework: **Pure PyTorch only** (NO PyTorch Lightning, NO Hydra)
- Package manager: **uv exclusively** (no pip/conda)
- Models: SwinUNETR · UNet++ · NNUNet2D · SwinUNETR+VLM V0 (text gate) · SwinUNETR+VLM V1 (image spatial)
- Abstract base: ViTSegBase (`vit_template.py`) — for future ViT backbone additions
- Metrics: Dice, IoU, hd95, Betti β0/β1
- Outputs: reproducible via seed + config snapshot + git hash + metrics.json

---

## Environment

```bash
# Always unset the shell-resident JiT venv AND set GPU before running
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py ...
env -u VIRTUAL_ENV uv run pytest -q   # tests don't need GPU selection

# Install packages into project venv
uv pip install --python .venv/bin/python <package>
```

- venv: `.venv/` (Python 3.11)
- Installed: torch 2.10.0+cu128, monai 1.5.2, einops, open_clip_torch 3.3.0, transformers
- **GPU policy: use GPU 0 or 1** (`CUDA_VISIBLE_DEVICES=0`) for all training/eval commands
- SwinUNETR smoke tests must use exp_card configs (not `--image_size`) because
  `defaults.yaml swinunetr.img_size=400` overrides the CLI flag

---

## Project Structure (current)

```
vlm-segmentation/
├── configs/
│   ├── defaults.yaml
│   └── exp_cards/
│       ├── OCTA6M-B0-SwinUNETR.yaml      # baseline
│       ├── DRIVE-B0-SwinUNETR.yaml
│       ├── OCTA6M-B1-UNetPP.yaml         # baseline
│       ├── DRIVE-B1-UNetPP.yaml
│       ├── OCTA6M-B2-nnUNet.yaml         # baseline (Phase A)
│       ├── DRIVE-B2-nnUNet.yaml
│       ├── OCTA6M-V0-SwinUNETR-VLM.yaml  # text-gate
│       ├── DRIVE-V0-SwinUNETR-VLM.yaml
│       ├── OCTA6M-V1-SwinUNETR-VLM.yaml  # image spatial injection
│       └── DRIVE-V1-SwinUNETR-VLM.yaml
├── src/
│   ├── datasets/   octa500.py, drive.py, dummy.py, discovery.py, transforms.py
│   ├── models/
│   │   ├── swinunetr.py         SwinUNETR2D         — baseline
│   │   ├── unetpp.py            UNetPlusPlus        — baseline
│   │   ├── nnunet_2d.py         NNUNet2D            — baseline (B2)
│   │   ├── vit_template.py      ViTSegBase          — abstract base for ViT additions
│   │   ├── vlm_prior.py         VLMPrior            — frozen BiomedCLIP encoder
│   │   ├── swinunetr_vlm.py     SwinUNETR2DVLM      — V0 text gate
│   │   └── swinunetr_vlm_v1.py  SwinUNETR2DVLMV1    — V1 image spatial
│   ├── losses/     dice.py, bce.py
│   ├── metrics/    dice_iou.py, hd95.py, topology.py
│   └── utils/      exp.py, seed.py, io.py
├── tests/
│   ├── test_metrics.py
│   ├── test_smoke_train.py
│   ├── test_vlm_prior.py          (V0 text tests)
│   ├── test_vlm_image_prior.py    (V1 image tests)
│   └── test_nnunet.py             (B2 nnU-Net tests — 71 total passing)
├── project_context/               (split context files — canonical source of truth)
├── train.py, eval.py
└── runs/   (gitignored)
```

---

## Run Output Contract (strict)

`runs/{dataset}/{model}/{exp_id}/`

- `config.yaml`      — merged config snapshot (re-saved after VLM init to include backbone name)
- `metrics.json`     — `{aggregate: {Dice,IoU,hd95,betti_beta0,betti_beta1,...}, per_sample: [...]}`
- `train_history.json`
- `ckpt/best.pt`, `ckpt/last.pt`
- `pred_vis/`        — ≥20 sample visualisations
- `git_commit.txt`

`exp_id`: CLI arg or auto `YYYYMMDD_HHMMSS_<shorthash>`.

---

## CLI Contract

### Training

```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/defaults.yaml \
    --dataset OCTA500-6M --model swinunetr \
    --exp_id OCTA6M-B0-SwinUNETR --outdir runs

# B2 nnU-Net
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-B2-nnUNet.yaml \
    --exp_id OCTA6M-B2-nnUNet --data_root ./data --outdir runs

# VLM V0 (text gate) — MUST use exp_card so img_size is set correctly
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-V0-SwinUNETR-VLM.yaml \
    --exp_id OCTA6M-V0-SwinUNETR-VLM --data_root ./data --outdir runs

# VLM V1 (image spatial)
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml \
    --exp_id OCTA6M-V1-SwinUNETR-VLM --data_root ./data --outdir runs

# Smoke test (dummy data, no real dataset needed)
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml \
    --dummy --epochs 2 --exp_id smoke_v1 --outdir runs
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-B2-nnUNet.yaml \
    --dummy --epochs 2 --exp_id smoke_nnunet --outdir runs
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python eval.py \
    --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml \
    --exp_id OCTA6M-V1-SwinUNETR-VLM \
    --ckpt runs/OCTA500-6M/swinunetr_vlm_v1/OCTA6M-V1-SwinUNETR-VLM/ckpt/best.pt \
    --data_root ./data --outdir runs

CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python eval.py \
    --config configs/exp_cards/OCTA6M-B2-nnUNet.yaml \
    --exp_id OCTA6M-B2-nnUNet \
    --ckpt runs/OCTA500-6M/nnunet_2d/OCTA6M-B2-nnUNet/ckpt/best.pt \
    --data_root ./data --outdir runs
```

### VLM-specific flags (both train.py and eval.py)

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--vlm_mode` | `text\|image\|both` | `image` for v1, `text` for v0 | Injection mode |
| `--vlm_alpha_init` | float | `0.1` | Initial injection scale |

---

## Implemented Models

### `build_model(cfg)` dispatch table

| cfg `model` key | Class | Forward signature |
|----------------|-------|-------------------|
| `swinunetr` | `SwinUNETR2D` | `model(x)` |
| `unetpp` / `unet++` | `UNetPlusPlus` | `model(x)` |
| `nnunet_2d` / `nnunet` | `NNUNet2D` | `model(x, text_embed=None, image_vlm_feat=None)` |
| `swinunetr_vlm` | `SwinUNETR2DVLM` | `model(x, text_embed)` |
| `swinunetr_vlm_v1` | `SwinUNETR2DVLMV1` | `model(x, text_embed=None, image_vlm_feat=None)` |

`_model_forward(model, imgs, cfg, text_embed, image_vlm_feat)` in `train.py`/`eval.py`
handles dispatch automatically based on `cfg["model"]`.

### `NNUNet2D` config keys (`cfg["nnunet"]`)

| Key | Default | Notes |
|-----|---------|-------|
| `base_channels` | `32` | Channel width at shallowest level |
| `n_pool` | `5` | Downsampling stages — input must be divisible by `2^n_pool` |
| `max_channels` | `320` | Channel cap |
| `deep_supervision` | `false` | Return list of logits when true |

### `ViTSegBase` (`vit_template.py`) — abstract base for new ViT additions

Subclass and implement:
- `_build_encoder()` — build and register backbone
- `_encode(x)` → `list[Tensor]` coarsest-first multi-scale features
- `_decode(features)` → `(B, out_ch, H, W)` logit map

VLM injection hooks (override to activate):
- `_apply_vlm(feat, level, text_embed, image_vlm_feat)` → `feat` (identity default)
- `_project_vlm(image_vlm_feat, h, w, level)` — bilinear upsample + 1×1 Conv projection
- `_register_vlm_projs(feat_dims)` — zero-init Conv2d projections per decoder level

---

## VLM Architecture Reference

### VLMPrior (`src/models/vlm_prior.py`)

Loads frozen BiomedCLIP (fallback: CLIP ViT-B/16) and exposes:

```python
prior = VLMPrior(dataset="OCTA500-6M", device=device)
text_embed = prior.get_text_embed()        # (1, 512) — cached, L2-normalised
img_feat   = prior.get_image_features(x)  # (B, 512, 14, 14) — per-batch, no grad
```

**BiomedCLIP internal structure** (open_clip 3.3.0):
- `model.visual` is a `TimmModel` (NOT `VisionTransformer`)
- `model.visual.trunk` = timm `VisionTransformer`
- `model.visual.head` = `Sequential(Dropout, Linear(768→512))`
- `trunk.forward_features(x)` → `(B, 197, 768)` — all tokens incl. CLS at `[:,0]`
- Patch tokens: `[:, 1:]` → `(B, 196, 768)` → apply `head` → `(B, 196, 512)` → reshape `(B, 512, 14, 14)`

The clip model is moved to `device` on load (GPU needed for efficient per-batch inference).

### V0 — Text-gated channel modulation (`swinunetr_vlm.py`)

```
gate = tanh( Linear(512 → C)(text_embed) )   # zero-init → identity at start
out  = decoder_feat * (1 + gate)             # channel-wise residual scale
```

Applied at 4 decoder stages: C = [384, 192, 96, 48] (coarse→fine).
`text_embed` is constant per run → gate degenerates to static channel mask.

### V1 — Per-image spatial injection (`swinunetr_vlm_v1.py`)

```
vlm_up = bilinear_upsample(image_vlm_feat, size=decoder_feat.shape[2:])
proj   = Conv2d(512 → C, 1×1)(vlm_up)     # zero-init → identity at start
out    = decoder_feat + alpha * proj       # alpha: learnable scalar (init=0.1)
```

Applied at 4 decoder stages. `image_vlm_feat` is `(B, 512, 14, 14)` computed
per batch from the frozen BiomedCLIP image encoder.

`vlm_mode` controls which branches are active:
- `"image"` (default for V1) — only spatial injection
- `"text"` — only text gate (V0 behaviour)
- `"both"` — both simultaneously

---

## Metrics Contract (LOCKED)

Required keys in `metrics.json["aggregate"]`:
`Dice`, `IoU`, `hd95`, `betti_beta0`, `betti_beta1`

- `hd95`: `inf` when either pred or gt is empty
- `betti_beta0/1`: `|pred_count − gt_count|` (skimage proxy)

---

## Coding Constraints

- Pure PyTorch. No Lightning, Hydra, or new heavy dependencies.
- All paths explicit; no hidden side effects.
- Short docstrings for non-trivial logic.
- Do not break existing tests (`uv run pytest -q` must stay green — currently 71 passing).
- New model variants: add to `src/models/__init__.py` dispatch table,
  add exp_card YAML, add pytest test file.
- New ViT backbones: subclass `ViTSegBase` from `vit_template.py`.

---

## Key Commands

```bash
# Run all tests
env -u VIRTUAL_ENV uv run pytest -q

# Smoke tests
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml \
    --dummy --epochs 2 --exp_id smoke_v1 --outdir runs
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-B2-nnUNet.yaml \
    --dummy --epochs 2 --exp_id smoke_nnunet --outdir runs

# B2 nnU-Net training
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-B2-nnUNet.yaml \
    --exp_id OCTA6M-B2-nnUNet --data_root ./data --outdir runs
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/DRIVE-B2-nnUNet.yaml \
    --exp_id DRIVE-B2-nnUNet --data_root ./data --outdir runs

# V1 training
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/OCTA6M-V1-SwinUNETR-VLM.yaml \
    --exp_id OCTA6M-V1-SwinUNETR-VLM --data_root ./data --outdir runs
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/DRIVE-V1-SwinUNETR-VLM.yaml \
    --exp_id DRIVE-V1-SwinUNETR-VLM --data_root ./data --outdir runs
```
