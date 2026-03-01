# [Project Objectives]

Improve boundary accuracy (hd95) and topological connectivity (Betti β0/β1) beyond Dice/IoU by injecting Vision-Language Model (VLM) feature priors into 2D vessel segmentation (OCTA500-6M, DRIVE).

**Core Hypotheses:**

1. Injecting semantic features from frozen VLM image encoders into the decoder → Reduces topological errors.
2. Gated fusion (1×1 Conv + Sigmoid) → Enhances boundary accuracy (hd95) compared to static fusion.
3. Predicting QC scores using VLM embeddings + internal segmentation features → Detects bad cases.
