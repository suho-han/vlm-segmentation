# [Project Structure]

```text
vlm-segmentation/
├── src/
│   ├── datasets/   octa500.py, drive.py, dummy.py, discovery.py, transforms.py
│   ├── models/     swinunetr.py, unetpp.py, vlm_prior.py
│   │               swinunetr_vlm.py (V0), swinunetr_vlm_v1.py (V1)
│   │               nnunet_2d.py (B2), vit_template.py (ViT abstract base)
│   ├── losses/     dice.py, bce.py
│   ├── metrics/    dice_iou.py, hd95.py, topology.py
│   └── utils/      exp.py, seed.py, io.py
├── configs/
│   ├── defaults.yaml
│   └── exp_cards/  OCTA6M-B{0,1,2}-*.yaml, DRIVE-B{0,1,2}-*.yaml
│                   OCTA6M-V0-SwinUNETR-VLM.yaml, DRIVE-V0-SwinUNETR-VLM.yaml
│                   OCTA6M-V1-SwinUNETR-VLM.yaml, DRIVE-V1-SwinUNETR-VLM.yaml
├── train.py, eval.py
├── tests/          test_metrics.py, test_smoke_train.py,
│                   test_vlm_prior.py, test_vlm_image_prior.py,
│                   test_nnunet.py  (71 passed)
└── runs/           (gitignored, exists only locally)
```
