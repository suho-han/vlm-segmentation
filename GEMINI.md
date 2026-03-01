# GEMINI.md — Scan/Summarization Agent Instructions (uv-aware, Papers + Runs)

## Purpose

**Canonical Source of Truth:** Before starting any scan, you MUST read and update the context files in `project_context/`. They contain the live, compressed summary of the project's research direction. All paper scanning and summarization should be guided by the hypotheses and goals outlined in those documents.

You help by handling large-scale scanning and producing **structured outputs**:

1) **Paper pipeline**: find/download papers (PDFs when possible) and generate structured Paper Cards
2) **Run pipeline**: scan experiment outputs (`runs/`) and produce leaderboards + failure mode reports
3) **Backbone pipeline**: scout for new architecture candidates and produce Backbone Cards
4) **QC pipeline** (later): summarize bad-case detection performance (AUROC/PR) and threshold rules

---

## Environment Note (uv)

This project uses **uv**. When you provide command examples, use:

- `uv run python ...`

(You do not need to execute commands unless explicitly instructed; focus on producing structured outputs.)

---

## Folder Conventions

### Papers

- `papers/inbox/`       # newly downloaded PDFs
- `papers/processed/`   # renamed/organized PDFs
- `papers/cards/`       # one markdown per paper (Paper Card)
- `papers/bib/`         # bibtex exports (optional)
- `papers/index.csv`    # master index (title, year, venue, tags, link, local_path)
- `papers/direction/`   # research justification and roadmap documents
  - `papers/direction/backbone_expand/`  # backbone expansion reports

Naming rule for PDFs:
- `{year}_{firstauthor}_{shorttitle}.pdf`

### Backbones

- `papers/direction/backbone_expand/backbone_cards/`  # one markdown per backbone architecture (Backbone Card)

### Runs & Analysis

- `runs/{dataset}/{model}/{exp_id}/metrics.json`
- `runs/analysis/`      # structured comparison reports (e.g., v0_vs_baseline.md)

---

## PAPER PIPELINE (REQUIRED)

### Goals
- Collect papers in two groups:
  1) **VLM applied to segmentation** (incl. open-vocabulary / promptable / referring segmentation)
  2) **VLM applied to other vision tasks** that can transfer to segmentation (grounding, detection, retrieval, distillation, adapters)

### Output format: Paper Card (one file per paper)
- **Title**, **Authors**, **Year / Venue**, **Links**, **Task**
- **Core idea**, **How VLM is used**, **Architecture**, **Training**
- **Datasets & Metrics**, **Key results**, **Limitations**
- **Relevance to our project (2D vessel segmentation)**
- **Tags** (open-vocab, referring, medical, feature-injection, qc, topology, boundary, etc.)

---

## BACKBONE PIPELINE (NEW)

### Goals
- Identify and document 2D segmentation backbones suitable for VLM feature-prior injection.
- Focus on ViT-based models (TransUNet, SegFormer, UNETR) and classical medical baselines (nnU-Net).

### Output format: Backbone Card
- **Name / Year / Venue**
- **Official links** (Verified arXiv/GitHub)
- **Core architecture** (encoder/decoder type)
- **Vessel connectivity benefits** (why it helps thin structures)
- **VLM injection friendliness** (where to inject, alignment needed)
- **Scores (0-5)**: Injection Suitability, hd95 Potential, Topology Preservation, Engineering Effort
- **Recommended role** (Primary / Secondary / Ablation)

---

## RUN PIPELINE (REQUIRED)

### Outputs
1. **Leaderboards** (per dataset, per model)
   - Top-3 runs sorted by Dice
   - Report: Dice, IoU, hd95, betti_beta0, betti_beta1
   - Highlight regressions (hd95↑, topology worsens)

2. **Failure mode report** (clustered)
   - discontinuity / broken vessels
   - missing thin vessels
   - false positives / noisy speckles
   - boundary leakage / overgrowth

3. **Comparison Analysis**
   - Formal reports in `runs/analysis/` comparing VLM variants (V0, V1) vs. Baselines (B0, B1).

---

## QC PIPELINE (WHEN AVAILABLE)
- Report AUROC/PR for bad-case detection
- State bad-case definition (Dice bottom 20% OR hd95 top 20%)

---

## What you should ask if missing (only if blocking)
- Target paper list/keywords, or minimum number of papers desired
- Exact metric keys used in `metrics.json` for topology (Betti vs proxy)
