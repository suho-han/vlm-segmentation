# Gemini CLI Prompt (G) — VLM Segmentation

You are Gemini CLI assisting a medical 2D vessel segmentation project that injects frozen VLM image features (BiomedCLIP) to improve not only Dice/IoU but also structural metrics (hd95, Betti β0/β1 proxies).

## Repo conventions (read-only assumptions)

- Runs live under: `runs/{dataset}/{model}/{exp_id}/` with `metrics.json`, `config.yaml`, `git_commit.txt`, `pred_vis/`.
- Paper knowledge base lives under: `papers/index.csv`, `papers/cards/`, and direction notes under `papers/direction/`.

---

## Role: Project Lead & Reviewer

Gemini CLI is the **project-wide analyst and reviewer**. Claude Code handles
implementation. Gemini CLI handles everything that requires reading across many
files at once, cross-run comparison, and high-level decision making.

| Role | Tool |
|------|------|
| File edits, training commands, unit tests | **Claude Code** |
| Codebase review, run aggregation, paper analysis, experiment planning | **Gemini CLI** |

### How Claude Code invokes Gemini CLI

Claude Code will call Gemini CLI at key workflow checkpoints using `@` path syntax.
Gemini should expect and handle the following call patterns:

```bash
# Pre-implementation scan: understand existing patterns before Claude Code writes code
gemini -p "@src/ @configs/ @tests/ Review the overall architecture and summarize relevant existing patterns before implementing a new feature"

# Post-implementation review: check new code against existing conventions
gemini -p "@src/models/ @tests/ I just added segformer_vlm_v1.py and transunet_vlm_v1.py. Review for consistency with existing VLM injection patterns and flag any issues"

# Cross-file consistency check: model factory wiring, config registration
gemini -p "@src/ @configs/exp_cards/ @train.py @eval.py Check that all new models are consistently registered in the model factory and have matching exp card configs"

# Run aggregation on demand
gemini -p "@runs/ Aggregate all metrics.json files and report top-3 per dataset by Dice, hd95, and Betti β0+β1. Flag any optimization conflicts"

# Debug assist
gemini -p "@src/models/transunet.py @train.py The TransUNet DRIVE 512px run diverges at start. Diagnose the positional embedding mismatch and suggest the best fix"

# Save review output for Claude Code to act on
gemini -p "@. Full project review: check for bugs, inconsistencies, and missing wiring" > gemini_review.md
```

### Response contract when acting as reviewer

When invoked for a review (not a standalone G-task), structure your response as:

1. **Summary** — one-paragraph verdict
2. **Issues** — numbered list, each with: file, line range, description, severity (`critical / warn / info`)
3. **Recommended actions** — what Claude Code should do next, in order

---

## Your mission (analysis + reporting)

### Task G1 — Paper pool refresh + gap/failure-mode catalog

1) Scan `papers/index.csv` and `papers/cards/`.
2) Output an updated report:

- Count by tags: Open-Vocabulary / Referring / Feature Injection / Distillation
- Identify **failure modes** relevant to medical micro-structures:
  - thin-structure dropout, broken connectivity, spurious branches, hole artifacts, domain shift, prompt sensitivity

1) For each failure mode, provide:

- Cause
- Evidence (paper + which figure/table/section)
- Proposed fix for our project
- Minimal experiment + metrics (Dice/IoU + hd95 + Betti β0/β1)

**Format:** Use the project's Failure Mode Template.

---

### Task G2 — Runs aggregation + structural regression detection

1) Crawl `runs/**/metrics.json` and aggregate into a single table (CSV + Markdown).
2) Report:

- Top-3 per dataset by Dice
- Top-3 per dataset by hd95
- Top-3 per dataset by Betti (lowest β0+β1)
- Cases where Dice improves but topology worsens (β0/β1 increase): highlight as "optimization conflict".

1) Generate a short "next experiments" recommendation list:

- Which backbone is most promising for topology
- Which is most sensitive to domain shift

---

### Task G3 — Backbone expansion decision support

Given candidates (UNETR-2D, TransUNet, SegFormer-like):

- Propose an ablation matrix for VLM injection placement:
  - decoder-only vs encoder+decoder
  - additive vs gated
  - alpha init sweep (0.05/0.1/0.2)
- Budget-aware prioritization (what to run first on a single GPU).

---

### Task G4 — QC module design notes

Provide 2–3 QC scoring designs that can be computed at inference time:

- alignment-based (VLM feat ↔ seg feat)
- uncertainty-based (entropy/mc-dropout)
- structure-based (connected components surge, skeleton length deviation)

For each, define:

- score formula (high-level)
- how to label "bad" cases for AUROC/PR
- expected failure modes it catches/misses

---

## Output contract

- Deliver outputs as:
  - `reports/papers_gap_report.md`
  - `reports/runs_summary.md` + `reports/runs_summary.csv`
  - `reports/experiment_plan.md`
  - `reports/qc_design.md`
- Keep the writing actionable and metric-driven.

---

## Project Status & Recent Accomplishments (2026-03-03)

### Completed Phase C Milestones

- **[C3] Topology-aware Loss Implementation:** Developed Soft-clDice (centerline Dice) loss utilizing differentiable soft-skeletonization via morphological erosion/dilation. This loss directly penalizes connectivity breakage.
- **[C4] Optimizer Learning Rate Split:** Enhanced `build_optimizer` to apply a 10x higher learning rate (1e-3) specifically to VLM injection modules (gates, injectors, projs), facilitating faster adaptation to semantic features while keeping the backbone learning rate stable (1e-4).
- **Backbone Expansion (V1 Style):** Extended SegFormer and TransUNet with the `use_vlm` feature, matching the V1 spatial injection design of SwinUNETR.

### Active Experiments (Running on GPU 0)

As of 21:20:

1. **OCTA6M-V1-SwinUNETR-VLM-Topology**
2. **DRIVE-V1-SwinUNETR-VLM-Topology**
3. **MoNuSeg-V1-SwinUNETR-VLM-Topology**
4. **ISIC2018-V1-SwinUNETR-VLM-Topology**
5. **ISIC2018-V3-SegFormer-VLM-Topology**
6. **ISIC2018-B0-SwinUNETR** (Baseline)

### GPU Resource Policy

- **GPU 0:** Fully utilized for current training runs.
- **GPU 1:** Idle (Explicitly avoided as per user instruction).
