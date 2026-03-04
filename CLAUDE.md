# Claude Code Prompt (C) — VLM Segmentation

You are Claude Code working inside the repo:
`/data1/suhohan/vlm-segmentation`

## Non-negotiable constraints

- Pure PyTorch only (NO PyTorch Lightning, NO Hydra).
- Configs are plain YAML under `configs/exp_cards/` + argparse overrides.
- Metrics must always include: `Dice`, `IoU`, `hd95`, `betti_beta0`, `betti_beta1` and be saved to `runs/{dataset}/{model}/{exp_id}/metrics.json`.
- **GPU policy:** use GPU 0,1  only unless explicitly instructed otherwise.
  - Always run with: `CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run ...`

## Context snapshot (what exists)

- Baselines already implemented: SwinUNETR (B0), UNet++ (B1), nnU-Net 2D (B2), SegFormer-like (B3), TransUNet (B4).
- VLM feature prior exists for SwinUNETR:
  - V0 = text-only gated channel scaling.
  - V1 = image spatial injection from BiomedCLIP image encoder (frozen).

---

## Gemini CLI — Project Oversight & Review

When a task involves cross-file impact, architectural decisions, or large-scale
consistency checks, delegate project-wide review to **Gemini CLI** before finalizing
any implementation. Gemini acts as the **project lead and reviewer**; Claude Code
acts as the **implementer**.

### Role Division

| Role | Tool |
|------|------|
| File edits, running commands, tests | **Claude Code** |
| Architecture review, cross-file consistency, large-context reasoning | **Gemini CLI** |

### When to invoke Gemini CLI

Invoke Gemini CLI in the following situations:

- Before starting a new task (C1–C4): scan the full repo for relevant existing code
- After implementing a task: review for consistency with existing patterns
- When a bug is non-obvious: hand the full src tree to Gemini for diagnosis
- Before any PR/commit: full project-wide sanity check

### Standard review commands for this repo

```bash
# Full project scan before starting a task
gemini -p "@src/ @configs/ @tests/ Review the overall architecture and summarize relevant existing patterns before I implement a new feature"

# Post-implementation review (e.g., after C1)
gemini -p "@src/models/ @tests/ I just added segformer_vlm_v1.py and transunet_vlm_v1.py. Review for consistency with existing VLM injection patterns in swinunetr_vlm_v1.py and flag any issues"

# Cross-file consistency check (naming, config wiring, factory registration)
gemini -p "@src/ @configs/exp_cards/ @train.py @eval.py Check that all new models are consistently registered in the model factory and have matching exp card configs"

# Topology loss review (C3)
gemini -p "@src/ @losses/ I implemented topology_loss.py. Review its integration with the training loop and verify Betti metric logging is consistent across all model variants"

# Debug convergence issue (C2)
gemini -p "@src/models/transunet.py @src/models/transunet_vlm_v1.py @train.py The TransUNet DRIVE 512px run diverges at start. Diagnose the positional embedding mismatch and suggest the best fix"

# Save review output for Claude Code to act on
gemini -p "@. Full project review: check for bugs, inconsistencies, and missing wiring" > gemini_review.md
```

### Integration pattern (per task)

1. **Before implementation** — run a Gemini scan to understand existing patterns
2. **Implement** using Claude Code tools
3. **Post-implementation review** — run Gemini on touched files + related context
4. **Apply feedback** — Claude Code patches issues raised by Gemini
5. **Final check** — re-run Gemini if changes were non-trivial

---

## Your mission (do these in order)

### Task C1 — Implement V1 VLM injection for SegFormer and TransUNet

**Goal:** Provide apples-to-apples comparison of VLM injection across backbones.

1) Add model files:

- `src/models/segformer_vlm_v1.py`
- `src/models/transunet_vlm_v1.py`

1) Reuse the existing VLM feature extractor:

- Use `src/models/vlm_prior.py:get_image_features(x) -> (B, 512, 14, 14)`.

1) Injection design (keep it simple, robust):

- For each backbone, inject VLM features into decoder stages via:
  - bilinear upsample VLM grid to stage spatial size
  - `Conv2d(512 -> C, 1x1)` projection (zero-init)
  - `feat = feat + alpha * proj(vlm_feat)` where `alpha` is learnable per stage (init 0.1)

1) Add exp cards:

- `configs/exp_cards/OCTA6M-V1-SegFormer-VLM.yaml`
- `configs/exp_cards/DRIVE-V1-SegFormer-VLM.yaml`
- `configs/exp_cards/OCTA6M-V1-TransUNet-VLM.yaml`
- `configs/exp_cards/DRIVE-V1-TransUNet-VLM.yaml`

1) Wire into `train.py` / `eval.py` model factory.

2) Add unit tests:

- shape tests, forward pass, deterministic seed smoke test
- ensure VLM path is frozen (no grad) unless explicitly enabled.

**Done criteria:**

- `env -u VIRTUAL_ENV uv run pytest -q` passes.
- One-epoch smoke training on dummy dataset passes for each new model.

---

### Task C2 — Fix TransUNet DRIVE(512px) convergence issue (pos embedding mismatch)

**Problem:** prior TransUNet run on DRIVE with 512px input failed due to ViT positional embedding/input grid mismatch.

Implement one of these fixes (choose best engineering tradeoff):

- Interpolate pos embeddings to match token grid at runtime.
- Or enforce input resolution that matches patch/grid assumptions via resizing in dataset transforms for TransUNet.

**Requirement:** The fix must be explicit in code + documented in the TransUNet config.

**Done criteria:**

- TransUNet baseline (A0/B4) and TransUNet-V1 runs no longer diverge at start; validation loss decreases.

---

### Task C3 — Add V2 (Topology-aware auxiliary loss) for V1 models

**Goal:** Improve structure metrics, not only Dice.

1) Implement `losses/topology_loss.py`:

- Use the existing Betti proxy signals (β0/β1) to build a differentiable surrogate.
- If full differentiability is hard, implement a proxy that correlates with connectivity (e.g., soft skeleton loss, clDice-style centerline loss) and keep Betti as evaluation.

1) Add exp cards for SwinUNETR-V1-V2 and at least one additional backbone:

- `OCTA6M-V2-SwinUNETR-VLM.yaml`, `DRIVE-V2-SwinUNETR-VLM.yaml`

1) Log structural metrics each epoch.

**Done criteria:**

- At least one dataset shows improvement in **either** hd95 or Betti errors vs the corresponding V1 baseline.

---

### Task C4 — Learning-rate split for injection layers

Implement optimizer param groups:

- backbone lr = 1e-4
- injection/project/gate layers lr = 1e-3 (10×)

Add to configs via a simple flag (no Hydra).

---

## Commands (examples)

```bash
# Tests
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run pytest -q

# Train example
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
  --config configs/exp_cards/OCTA6M-V1-SegFormer-VLM.yaml \
  --exp_id OCTA6M-V1-SegFormer-VLM \
  --epochs 1000 --patience 50 --data_root ./data --outdir runs
```

## Output contract

For every change, provide:

- files touched
- how to run (exact commands)
- expected outputs under `runs/`
- brief notes on risks/failure modes
