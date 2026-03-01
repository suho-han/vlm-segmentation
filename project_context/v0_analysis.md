# [V0 Analysis: What the Text Feature Actually Does]

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
