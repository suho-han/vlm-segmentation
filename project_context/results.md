# [All Results]

**학습 설정:** epochs=1000 (max), early_stopping=val_loss (patience=50), AdamW lr=1e-4, Dice+BCE loss, seed=42  
**배치 크기:** OCTA=4, DRIVE=2, MoNuSeg=8, ISIC2018=8  
**입력 크기:** OCTA=384px, DRIVE=512px, MoNuSeg=256px, ISIC2018=256px

---

## OCTA500-6M

배치=4, 입력=384px, in_channels=1

| exp_id | Dice | IoU | hd95 | β0 err | β1 err | epochs | best_val |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OCTA6M-B0-SwinUNETR | 0.8468 | 0.7350 | **1.828** | 22.67 | 4.74 | — | — |
| OCTA6M-B1-UNetPP | **0.8859** | 0.7961 | 1.924 | 25.12 | 5.65 | — | — |
| OCTA6M-B2-nnUNet | 0.8444 | 0.7314 | 1.932 | 29.37 | 6.93 | — | — |
| OCTA6M-B3-SegFormer | 0.8052 | 0.6747 | 2.710 | 63.26 | 11.66 | — | — |
| OCTA6M-B3-SegFormer-VLM | 0.8057 | 0.6753 | 2.670 | 58.21 | 10.95 | — | — |
| OCTA6M-B4-TransUNet | 0.8411 | 0.7270 | 2.042 | 23.37 | 6.24 | — | — |
| OCTA6M-B4-TransUNet-VLM | 0.8410 | 0.7263 | 2.504 | 18.50 | 6.78 | — | — |
| OCTA6M-V0-SwinUNETR-VLM | 0.8469 | 0.7351 | 1.922 | 22.78 | 5.32 | — | — |
| OCTA6M-V1-SwinUNETR-VLM | 0.8471 | 0.7354 | 1.904 | 18.59 | **4.38** | — | — |
| 20260303_212007_7373275 | 0.8466 | 0.7346 | 2.025 | **13.43** | 4.58 | — | — |


---

## DRIVE

배치=2, 입력=512px, in_channels=3. †B4-TransUNet: pos_embed 불일치로 수렴 불량

| exp_id | Dice | IoU | hd95 | β0 err | β1 err | epochs | best_val |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE-B0-SwinUNETR | 0.7791 | 0.6387 | 9.021 | 35.15 | 19.40 | — | — |
| DRIVE-B1-UNetPP | **0.8003** | 0.6674 | **4.783** | 44.65 | **16.20** | — | — |
| DRIVE-B2-nnUNet | 0.7597 | 0.6129 | 10.489 | 43.05 | 23.85 | — | — |
| DRIVE-B3-SegFormer | 0.7578 | 0.6103 | 10.615 | 32.00 | 25.55 | — | — |
| DRIVE-B3-SegFormer-VLM | 0.7577 | 0.6102 | 9.010 | 31.90 | 24.05 | — | — |
| DRIVE-B4-TransUNet | 0.5850 | 0.4140 | 23.191 | **30.70** | 37.90 | — | — |
| DRIVE-B4-TransUNet-VLM | 0.5278 | 0.3590 | 25.695 | 31.05 | 32.15 | — | — |
| DRIVE-V0-SwinUNETR-VLM | 0.7768 | 0.6355 | 10.273 | 36.60 | 22.35 | — | — |
| DRIVE-V1-SwinUNETR-VLM | 0.7769 | 0.6358 | 9.628 | 43.85 | 20.75 | — | — |
| 20260303_212007_7373275 | 0.7750 | 0.6330 | 9.240 | 32.35 | 22.70 | — | — |


---

## MoNuSeg

37 train → 80/20 split (29 train / 8 val), 14 test. RGB 1000×1000 → 256×256.

| exp_id | Dice | IoU | hd95 | β0 err | β1 err | epochs | best_val |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MoNuSeg-B0-SwinUNETR | 0.7827 | 0.6450 | 3.924 | 112.79 | 13.29 | — | — |
| MoNuSeg-B1-UNetPP | **0.8090** | 0.6803 | **3.156** | **47.86** | 11.21 | — | — |
| MoNuSeg-B2-nnUNet | 0.7836 | 0.6455 | 3.756 | 92.43 | **10.00** | — | — |
| MoNuSeg-B3-SegFormer | 0.7055 | 0.5460 | 3.615 | 144.36 | 14.57 | — | — |
| MoNuSeg-B3-SegFormer-VLM | 0.6963 | 0.5349 | 3.727 | 152.64 | 18.64 | — | — |
| MoNuSeg-B4-TransUNet | 0.7418 | 0.5906 | 3.556 | 66.50 | 19.21 | — | — |
| MoNuSeg-B4-TransUNet-VLM | 0.7466 | 0.5963 | 3.492 | 54.29 | 20.00 | — | — |
| MoNuSeg-V0-SwinUNETR-VLM | 0.7836 | 0.6462 | 3.734 | 110.21 | 12.86 | — | — |
| MoNuSeg-V1-SwinUNETR-VLM | 0.7866 | 0.6493 | 3.660 | 107.50 | 10.93 | — | — |
| 20260303_212007_7373275 | 0.7850 | 0.6468 | 3.300 | 58.86 | 15.79 | — | — |


---

## ISIC2018

배치=8, 입력=256px, in_channels=3. 2594 train / 100 val / 1000 test.

| exp_id | Dice | IoU | hd95 | β0 err | β1 err | epochs | best_val |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ISIC2018-B0-SwinUNETR | **0.8925** | 0.8222 | **20.949** | 5.69 | 7.89 | — | — |
| ISIC2018-B1-UNetPP | 0.8841 | 0.8123 | 23.421 | 0.87 | **2.61** | — | — |
| ISIC2018-B2-nnUNet | 0.8811 | 0.8112 | 21.602 | 3.45 | 7.66 | — | — |
| ISIC2018-B3-SegFormer | 0.8751 | 0.7971 | 22.591 | 0.75 | 2.61 | — | — |
| ISIC2018-B3-SegFormer-VLM | 0.8826 | 0.8090 | 22.962 | 0.78 | 2.64 | — | — |
| ISIC2018-B4-TransUNet | 0.8734 | 0.7947 | 24.039 | 4.32 | 10.74 | — | — |
| ISIC2018-B4-TransUNet-VLM | 0.8876 | 0.8155 | 23.698 | 4.71 | 5.78 | — | — |
| ISIC2018-V0-SwinUNETR-VLM | 0.8851 | 0.8141 | 22.220 | 6.27 | 10.07 | — | — |
| ISIC2018-V1-SwinUNETR-VLM | 0.8906 | 0.8168 | 21.749 | 5.60 | 16.27 | — | — |
| ISIC2018-V1-SwinUNETR-VLM-Topology | 0.8625 | 0.7710 | 24.132 | 1.13 | 7.18 | — | — |
| 20260303_210506_7373275 | 0.8454 | 0.7487 | 30.243 | **0.74** | 8.92 | — | — |


---

## 종합 비교 (Dice)

| 모델 | OCTA500-6M | DRIVE | MoNuSeg | ISIC2018 |
| --- | --- | --- | --- | --- |
| ISIC2018-B0-SwinUNETR | — | — | — | 0.8925 |
| MoNuSeg-B0-SwinUNETR | — | — | 0.7827 | — |
| DRIVE-B0-SwinUNETR | — | 0.7791 | — | — |
| OCTA6M-B0-SwinUNETR | 0.8468 | — | — | — |
| ISIC2018-B1-UNetPP | — | — | — | 0.8841 |
| DRIVE-B1-UNetPP | — | 0.8003 | — | — |
| OCTA6M-B1-UNetPP | 0.8859 | — | — | — |
| MoNuSeg-B1-UNetPP | — | — | 0.8090 | — |
| MoNuSeg-B2-nnUNet | — | — | 0.7836 | — |
| DRIVE-B2-nnUNet | — | 0.7597 | — | — |
| OCTA6M-B2-nnUNet | 0.8444 | — | — | — |
| ISIC2018-B2-nnUNet | — | — | — | 0.8811 |
| OCTA6M-B3-SegFormer-VLM | 0.8057 | — | — | — |
| ISIC2018-B3-SegFormer-VLM | — | — | — | 0.8826 |
| MoNuSeg-B3-SegFormer-VLM | — | — | 0.6963 | — |
| DRIVE-B3-SegFormer-VLM | — | 0.7577 | — | — |
| ISIC2018-B3-SegFormer | — | — | — | 0.8751 |
| DRIVE-B3-SegFormer | — | 0.7578 | — | — |
| OCTA6M-B3-SegFormer | 0.8052 | — | — | — |
| MoNuSeg-B3-SegFormer | — | — | 0.7055 | — |
| OCTA6M-B4-TransUNet-VLM | 0.8410 | — | — | — |
| DRIVE-B4-TransUNet | — | 0.5850 | — | — |
| ISIC2018-B4-TransUNet-VLM | — | — | — | 0.8876 |
| MoNuSeg-B4-TransUNet-VLM | — | — | 0.7466 | — |
| OCTA6M-B4-TransUNet | 0.8411 | — | — | — |
| MoNuSeg-B4-TransUNet | — | — | 0.7418 | — |
| ISIC2018-B4-TransUNet | — | — | — | 0.8734 |
| DRIVE-B4-TransUNet-VLM | — | 0.5278 | — | — |
| ISIC2018-V0-SwinUNETR-VLM | — | — | — | 0.8851 |
| DRIVE-V0-SwinUNETR-VLM | — | 0.7768 | — | — |
| OCTA6M-V0-SwinUNETR-VLM | 0.8469 | — | — | — |
| MoNuSeg-V0-SwinUNETR-VLM | — | — | 0.7836 | — |
| DRIVE-V1-SwinUNETR-VLM | — | 0.7769 | — | — |
| MoNuSeg-V1-SwinUNETR-VLM | — | — | 0.7866 | — |
| ISIC2018-V1-SwinUNETR-VLM | — | — | — | 0.8906 |
| OCTA6M-V1-SwinUNETR-VLM | 0.8471 | — | — | — |
| ISIC2018-V1-SwinUNETR-VLM-Topology | — | — | — | 0.8625 |
| 20260303_210506_7373275 | — | — | — | 0.8454 |
| 20260303_212007_7373275 | 0.8466 | 0.7750 | 0.7850 | — |


---

## 완료된 주요 과제 (Phase C Milestone)

- **[C1]** SegFormer/TransUNet VLM V1: `use_vlm` flag로 spatial injection 구현 완료.
- **[C2]** TransUNet 512px pos_embed 수정: `_resize_pos_embed()` 적용 완료.
- **[C3]** Topology-aware loss (SoftclDice): OCTA6M/DRIVE/MoNuSeg/ISIC2018 실험 완료.
- **[C4]** LR split (backbone 1e-4 / injection 1e-3): `build_optimizer` param groups 완료.

---

## 시각화

```bash
env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset OCTA500-6M
env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset DRIVE
env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset MoNuSeg
env -u VIRTUAL_ENV uv run python scripts/visualize_compare.py --runs_dir runs --dataset ISIC2018
```
