# vlm-segmentation

Pure PyTorch 2D vessel segmentation 실험 프레임워크.

> **주의**: 이 레포는 PL/Hydra를 사용하지 않습니다. argparse + PyYAML 기반으로 동작합니다.

---

## 지원 데이터셋 & 모델

| 데이터셋 | 모델 |
|---------|------|
| OCTA500-6M | SwinUNETR (monai 기반) |
| DRIVE | UNet++ (pure PyTorch) |

---

## 폴더 구조

```
vlm-segmentation/
├── configs/
│   ├── defaults.yaml          # 기본 설정
│   └── exp_cards/             # 실험별 오버라이드 YAML
├── src/
│   ├── datasets/              # octa500.py, drive.py, transforms.py
│   ├── models/                # swinunetr.py, unetpp.py
│   ├── losses/                # dice.py, bce.py
│   ├── metrics/               # dice_iou.py, hd95.py, topology.py
│   └── utils/                 # exp.py, seed.py, io.py
├── scripts/
│   ├── run_train.sh
│   └── run_eval.sh
├── tests/
│   ├── test_metrics.py
│   └── test_smoke_train.py
├── docs/
│   └── EXPERIMENTS.md
├── train.py
└── eval.py
```

---

## 설치

```bash
# conda/mamba 환경 권장
pip install torch torchvision
pip install monai          # SwinUNETR 용
pip install pyyaml numpy scipy scikit-image natsort
pip install pytest         # 테스트 실행 시
```

---

## 빠른 시작

### 학습

```bash
python train.py \
  --dataset   OCTA500-6M \
  --model     swinunetr \
  --config    configs/defaults.yaml \
  --data_root /data1/suhohan/JiT/data \
  --exp_id    OCTA6M-B0-SwinUNETR \
  --outdir    runs
```

### 평가

```bash
python eval.py \
  --dataset   OCTA500-6M \
  --model     swinunetr \
  --ckpt      runs/OCTA500-6M/swinunetr/OCTA6M-B0-SwinUNETR/ckpt/best.pt \
  --exp_id    OCTA6M-B0-SwinUNETR \
  --data_root /data1/suhohan/JiT/data \
  --outdir    runs
```

### 스크립트

```bash
bash scripts/run_train.sh OCTA500-6M swinunetr OCTA6M-B0-SwinUNETR
bash scripts/run_eval.sh  OCTA500-6M swinunetr OCTA6M-B0-SwinUNETR
```

---

## 테스트

```bash
pytest tests/ -v
```

---

## 결과물 위치

```
runs/{dataset}/{model}/{exp_id}/
  ├── config.yaml
  ├── git_commit.txt
  ├── train_history.json
  ├── metrics.json
  ├── ckpt/best.pt
  ├── ckpt/last.pt
  └── pred_vis/*.png
```

자세한 실험 실행 가이드는 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)를 참조하세요.

---

## 의존성 메모

- **monai**: SwinUNETR 모델 구현체 사용 (`src/models/swinunetr.py`). `spatial_dims=2`로 2D 모드 사용.
- **scipy**: HD95 계산 (`src/metrics/hd95.py`).
- **scikit-image**: Betti number 근사 (`src/metrics/topology.py`).
- **natsort**: 이미지 파일 정렬.
- UNet++는 monai 불필요 (pure PyTorch).
