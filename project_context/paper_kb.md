# [Paper Knowledge Base (2026-02-27)]

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
