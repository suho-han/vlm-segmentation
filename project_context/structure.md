# [Project Structure]

```text
vlm-segmentation/
├── src/
│   ├── datasets/   octa500.py, drive.py, isic2018.py, monuseg.py
│   │               dummy.py, discovery.py, transforms.py
│   ├── models/     swinunetr.py, unetpp.py, vlm_prior.py
│   │               swinunetr_vlm.py (V0), swinunetr_vlm_v1.py (V1)
│   │               nnunet_2d.py (B2), vit_template.py (ViT abstract base)
│   │               segformer_b2.py (B3), transunet.py (B4)
│   ├── losses/     dice.py, bce.py, topology.py (Soft-clDice)
│   ├── metrics/    dice_iou.py, hd95.py, topology.py
│   └── utils/      exp.py, seed.py, io.py
├── configs/
│   ├── defaults.yaml
│   └── exp_cards/
│       # OCTA6M
│       OCTA6M-B0-SwinUNETR.yaml, OCTA6M-B1-UNetPP.yaml, OCTA6M-B2-nnUNet.yaml
│       OCTA6M-B3-SegFormer.yaml, OCTA6M-B4-TransUNet.yaml
│       OCTA6M-V0-SwinUNETR-VLM.yaml, OCTA6M-V1-SwinUNETR-VLM.yaml
│       OCTA6M-V1-SwinUNETR-VLM-Topology.yaml
│       OCTA6M-B3-SegFormer-VLM.yaml, OCTA6M-B4-TransUNet-VLM.yaml
│       OCTA6M-V1-SwinUNETR-VLM-Topology.yaml
│       # DRIVE
│       DRIVE-B0-SwinUNETR.yaml, DRIVE-B1-UNetPP.yaml, DRIVE-B2-nnUNet.yaml
│       DRIVE-B3-SegFormer.yaml, DRIVE-B4-TransUNet.yaml
│       DRIVE-V0-SwinUNETR-VLM.yaml, DRIVE-V1-SwinUNETR-VLM.yaml
│       DRIVE-V1-SwinUNETR-VLM-Topology.yaml
│       DRIVE-B3-SegFormer-VLM.yaml, DRIVE-B4-TransUNet-VLM.yaml
│       # ISIC2018
│       ISIC2018-B0-SwinUNETR.yaml, ISIC2018-B1-UNetPP.yaml, ISIC2018-B2-nnUNet.yaml
│       ISIC2018-B3-SegFormer.yaml, ISIC2018-B4-TransUNet.yaml
│       ISIC2018-V0-SwinUNETR-VLM.yaml, ISIC2018-V1-SwinUNETR-VLM.yaml
│       ISIC2018-V1-SwinUNETR-VLM-Topology.yaml
│       ISIC2018-B3-SegFormer-VLM.yaml, ISIC2018-B4-TransUNet-VLM.yaml
│       ISIC2018-V3-SegFormer-VLM-Topology.yaml
│       # MoNuSeg
│       MoNuSeg-B0-SwinUNETR.yaml, MoNuSeg-B1-UNetPP.yaml, MoNuSeg-B2-nnUNet.yaml
│       MoNuSeg-B3-SegFormer.yaml, MoNuSeg-B4-TransUNet.yaml
│       MoNuSeg-V0-SwinUNETR-VLM.yaml, MoNuSeg-V1-SwinUNETR-VLM.yaml
│       MoNuSeg-V1-SwinUNETR-VLM-Topology.yaml
│       MoNuSeg-B3-SegFormer-VLM.yaml, MoNuSeg-B4-TransUNet-VLM.yaml
├── train.py, eval.py
├── tests/          test_metrics.py, test_smoke_train.py,
│                   test_vlm_prior.py, test_vlm_image_prior.py,
│                   test_nnunet.py, test_segformer.py, test_transunet.py
│                   test_losses.py, test_optimizer.py
│                   (100+ tests passing)
└── runs/           (gitignored, exists only locally)
```
