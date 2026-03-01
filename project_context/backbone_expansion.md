# [Backbone Expansion: Top-2 Selection & Strategy (2026-03-01)]

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
- **Phase A (Baselines)**: nnU-Net (B2), SegFormer (B3), TransUNet (B4) implementation and training complete.
- **Phase B (VLM Feasibility)**: UNETR-2D and TransUNet blueprints defined.
- **Phase C (Execution)**: 
  - SwinUNETR-V1 (Image-branch) training complete.
  - Pending V1 implementation and training for SegFormer and TransUNet.

