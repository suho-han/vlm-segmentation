# Experiments Guide

## 환경 설정 (uv)

```bash
# 프로젝트 루트에서
uv venv                        # .venv 생성 (Python 3.11)
source .venv/bin/activate

# 의존성 설치 (torch 별도)
uv pip install -e .
uv pip install monai einops    # SwinUNETR 필수 의존성

# PyTorch (CUDA 12.8)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 테스트
uv run pytest -q
```

**주의:** shell에 `VIRTUAL_ENV=/다른경로`가 설정된 경우, `env -u VIRTUAL_ENV uv run ...` 또는 `source .venv/bin/activate` 필요.

---

## 데이터셋 설정

### DATA_ROOT 설정 방법 (우선순위 순)

1. **CLI 인자** (최우선): `--data_root /path/to/data`
2. **환경 변수**: `export DATA_ROOT=/path/to/data`
3. **configs/defaults.yaml**: `data_root: "./data"` (기본값)

### 자동 탐지 (Auto-discovery)

`--data_root` 아래에서 다음 이름의 폴더를 자동 탐지합니다:

| dataset | 탐지 후보 폴더명 |
|---------|----------------|
| OCTA500-6M | OCTA500_6M, OCTA500-6M, OCTA5006M, OCTA500 |
| DRIVE | DRIVE, drive |

탐지 실패 시: `runs/_debug/dataset_scan_{dataset}.txt` 에 디버그 리포트 생성.

### 데이터 디렉토리 구조

```
data/
├── OCTA500_6M/          # 이미지: *.bmp, 라벨: *.bmp
│   ├── train/ images/ + labels/
│   ├── val/   images/ + labels/
│   └── test/  images/ + labels/
└── DRIVE/               # 이미지: *.tif, 마스크: *.gif (공식 DRIVE 형식)
    ├── train/ images/ + masks/
    └── test/  images/ + masks/
```

파일 확장자는 **자동 탐지** (`image_ext: null`, `label_ext: null` in config).

---

## 4개 Baseline 실행 명령어

### Baseline 1: OCTA500-6M + SwinUNETR

```bash
# 학습
uv run python train.py \
    --config configs/exp_cards/OCTA6M-B0-SwinUNETR.yaml \
    --exp_id OCTA6M-B0-SwinUNETR \
    --data_root ./data \
    --outdir runs

# 평가
uv run python eval.py \
    --config configs/exp_cards/OCTA6M-B0-SwinUNETR.yaml \
    --exp_id OCTA6M-B0-SwinUNETR \
    --ckpt runs/OCTA500-6M/swinunetr/OCTA6M-B0-SwinUNETR/ckpt/best.pt \
    --data_root ./data \
    --outdir runs
```

### Baseline 2: OCTA500-6M + UNet++

```bash
uv run python train.py \
    --config configs/exp_cards/OCTA6M-B1-UNetPP.yaml \
    --exp_id OCTA6M-B1-UNetPP \
    --data_root ./data \
    --outdir runs

uv run python eval.py \
    --config configs/exp_cards/OCTA6M-B1-UNetPP.yaml \
    --exp_id OCTA6M-B1-UNetPP \
    --ckpt runs/OCTA500-6M/unetpp/OCTA6M-B1-UNetPP/ckpt/best.pt \
    --data_root ./data \
    --outdir runs
```

### Baseline 3: DRIVE + SwinUNETR

```bash
uv run python train.py \
    --config configs/exp_cards/DRIVE-B0-SwinUNETR.yaml \
    --exp_id DRIVE-B0-SwinUNETR \
    --data_root ./data \
    --outdir runs

uv run python eval.py \
    --config configs/exp_cards/DRIVE-B0-SwinUNETR.yaml \
    --exp_id DRIVE-B0-SwinUNETR \
    --ckpt runs/DRIVE/swinunetr/DRIVE-B0-SwinUNETR/ckpt/best.pt \
    --data_root ./data \
    --outdir runs
```

### Baseline 4: DRIVE + UNet++

```bash
uv run python train.py \
    --config configs/exp_cards/DRIVE-B1-UNetPP.yaml \
    --exp_id DRIVE-B1-UNetPP \
    --data_root ./data \
    --outdir runs

uv run python eval.py \
    --config configs/exp_cards/DRIVE-B1-UNetPP.yaml \
    --exp_id DRIVE-B1-UNetPP \
    --ckpt runs/DRIVE/unetpp/DRIVE-B1-UNetPP/ckpt/best.pt \
    --data_root ./data \
    --outdir runs
```

---

## 실험 결과 위치 (runs output contract)

```
runs/{dataset}/{model}/{exp_id}/
├── config.yaml        ← 실행에 사용된 전체 설정 스냅샷 (STRICT)
├── git_commit.txt     ← 실험 시점 git hash (STRICT)
├── train_history.json ← epoch별 train_loss, val_dice, val_iou
├── metrics.json       ← eval 결과 aggregate + per_sample (STRICT)
├── ckpt/
│   ├── best.pt        ← val Dice 최고 체크포인트 (STRICT)
│   └── last.pt        ← 마지막 epoch 체크포인트 (STRICT)
└── pred_vis/          ← image | gt | pred 시각화 (기본 20장)
```

---

## metrics.json 키 사양 (A3 lock)

모든 eval은 다음 키를 반드시 포함합니다:

| 키 | 설명 | 엣지 케이스 |
|----|------|------------|
| `Dice` | Sørensen–Dice coefficient | empty pred/gt → ~0 (Laplace smoothing) |
| `IoU` | Intersection over Union (Jaccard) | empty pred/gt → ~0 |
| `hd95` | 95th-percentile Hausdorff distance (px) | empty pred or gt → **inf** |
| `betti_beta0` | β0 오차 = \|β0_pred - β0_gt\| (연결 성분 수) | empty mask → 0 components |
| `betti_beta1` | β1 오차 = \|β1_pred - β1_gt\| (구멍 수) | empty mask → 0 holes |

**betti 프록시 구현 (`src/metrics/topology.py`):**
- β0 (connected components): `skimage.measure.label` 8-connectivity
- β1 (holes): 보색(complement) 1-connectivity 컴포넌트 수 - 1 (배경 제외)
  → 진정한 persistent homology가 아닌 Euler characteristic 기반 프록시

---

## SwinUNETR 입력 크기 제약

MONAI SwinUNETR은 입력 크기가 **32의 배수**여야 합니다:
- OCTA500-6M: `image_size: 384` (400 → 384로 조정)
- DRIVE: `image_size: 512`

---

## 더미 데이터 (실제 데이터 없이 실행)

```bash
# 학습 (dummy)
uv run python train.py --dummy --model unetpp --exp_id smoke-unetpp \
    --epochs 2 --image_size 64 --outdir runs

# 평가 (dummy)
uv run python eval.py --dummy --model unetpp --exp_id smoke-unetpp \
    --ckpt runs/dummy/unetpp/smoke-unetpp/ckpt/best.pt --outdir runs
```

dummy 실행 시 `metrics.json`에 `"_note": "dummy_data"` 표시됨.

---

## 베이스라인 결과 (2026-02-27)

### DRIVE

| exp_id | Dice | IoU | hd95 | β0 err | β1 err | best val Dice |
|--------|------|-----|------|--------|--------|--------------|
| DRIVE-B0-SwinUNETR | 0.7746 | 0.6324 | 8.94 | 38.95 | 21.15 | 0.7802 |
| DRIVE-B1-UNetPP    | 0.7988 | 0.6652 | 4.36 | 41.15 | 15.25 | 0.8071 |

### OCTA500-6M

| exp_id | Dice | IoU | hd95 | β0 err | β1 err | best val Dice |
|--------|------|-----|------|--------|--------|--------------|
| OCTA6M-B0-SwinUNETR | 0.8454 | 0.7329 | 1.90 | 27.77 | 5.66 | 0.8390 |
| OCTA6M-B1-UNetPP    | 0.8864 | 0.7969 | 1.83 | 25.90 | 6.15 | 0.8814 |

**비고:**
- 모든 실험: epochs=100, optimizer=Adam, lr=1e-4, loss=Dice+BCE
- DRIVE: batch=2, SwinUNETR image_size=512, UNet++ image_size=512
- OCTA500-6M: batch=4, SwinUNETR image_size=384, UNet++ image_size=400
- hd95 단위: 픽셀 (px)
- β0/β1 오차: |pred - gt| (연결 성분 / 구멍 수)
