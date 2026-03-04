# 전처리 파이프라인 (Preprocessing Pipeline)

이 문서는 원본 이미지와 마스크가 세그멘테이션 모델에 입력되기 전에 어떻게 로드되고, 변환되고, 배치로 묶이는지 설명합니다.

---

## 1. 진입점: `get_loaders(cfg)`

`src/datasets/__init__.py`가 유일한 진입점입니다.
호출자는 config dict를 전달하며, 함수는 `cfg["dataset"]` 값에 따라 적절한 데이터셋 로더로 분기합니다:

```
cfg["dataset"]          → loader
─────────────────────────────────────
dummy                   → get_dummy_loaders      (스모크 테스트용, 랜덤 텐서)
OCTA500-6M / OCTA500-3M → get_octa500_loaders
DRIVE                   → get_drive_loaders
MoNuSeg                 → get_monuseg_loaders
ISIC2018                → get_isic2018_loaders
```

각 로더는 `(train_loader, val_loader, test_loader)` 튜플을 반환합니다.

---

## 2. 데이터셋 자동 탐색 (Auto-Discovery)

모든 실제 데이터셋 로더는 파일을 열기 전에 `src/datasets/discovery.py`의
`find_dataset_root(data_root, dataset_name)`을 호출합니다.

**탐색 전략 (2단계):**

1. `data_root`의 직계 하위 폴더를 스캔하여 후보 이름 목록과 일치하는 폴더를 찾습니다 (대소문자 무시).
2. 찾지 못하면 한 단계 더 내려가서 동일하게 반복합니다.

**데이터셋별 허용 폴더 이름:**

| 데이터셋     | 허용되는 폴더 이름 |
|--------------|--------------------|
| OCTA500-6M   | OCTA500_6M, OCTA500-6M, OCTA5006M, OCTA500 |
| OCTA500-3M   | OCTA500_3M, OCTA500-3M, OCTA5003M |
| DRIVE        | DRIVE, drive, Drive |
| MoNuSeg      | MoNuSeg, monuseg, MONUSEG, MoNuSeg2018 |
| ISIC2018     | ISIC2018, ISIC_2018, isic2018, ISIC-2018 |

**유효성 검사:** 후보 경로 내부에 기대되는 하위 경로(예: `train/images/`)가 하나 이상 존재할 때만 해당 경로를 유효한 것으로 인정합니다.

**탐색 실패 시:** `runs/_debug/dataset_scan_<name>.txt`에 `data_root` 아래에서 발견된 내용을 담은 디버그 리포트가 자동으로 생성됩니다.

---

## 3. 디렉토리 구조

각 데이터셋이 기대하는 폴더 구조입니다:

### OCTA500 (6M / 3M)
```
<root>/
  train/images/*.bmp   train/labels/*.bmp
  val/images/*.bmp     val/labels/*.bmp
  test/images/*.bmp    test/labels/*.bmp
```
train/val/test 분할이 미리 제공됨 — 코드에서 별도 분할 불필요.

### DRIVE
```
<root>/
  train/images/*.tif   train/masks/*.gif
  test/images/*.tif    test/masks/*.gif
```
val 분할 미제공. 코드에서 train을 80/20으로 결정론적 분할 (seed는 cfg에서).

### ISIC 2018
```
<root>/
  train/images/ISIC_XXXXXXX.jpg      train/labels/ISIC_XXXXXXX_segmentation.png
  val/images/ISIC_XXXXXXX.jpg        val/labels/ISIC_XXXXXXX_segmentation.png
  test/images/ISIC_XXXXXXX.jpg       test/labels/ISIC_XXXXXXX_segmentation.png
```
미리 제공된 분할 (2594 / 100 / 1000장).
이미지-레이블 매칭 방식: `<stem>_segmentation.png`.

### MoNuSeg
```
<root>/
  train/images/TCGA-*.tif   train/labels/TCGA-*.npy   (37장 → 29/8 분할)
  test/images/TCGA-*.tif    test/labels/TCGA-*.npy    (14장)
```
레이블은 NumPy `.npy` 파일, shape `(H, W, 1)`, 값 `{0, 255}`.
37장의 train 이미지를 80/20으로 결정론적 분할 (seed는 cfg에서).

---

## 4. 확장자 자동 감지

OCTA500과 DRIVE 로더는 `cfg["image_ext"]` / `cfg["label_ext"]`가 `null`(기본값)일 때
`_detect_ext(directory)`를 호출합니다.
해당 디렉토리에서 알려진 이미지 확장자를 모두 세어 가장 많이 사용된 확장자를 반환합니다.
지원 확장자: `.png .jpg .jpeg .tif .tiff .bmp .gif`

---

## 5. 변환 파이프라인 (Transform Pipeline)

모든 변환은 `src/datasets/transforms.py`에 정의되어 있습니다.
**쌍(paired) 방식**으로 동작합니다: 각 변환은 `(img: PIL.Image, mask: PIL.Image)`를 받아
동일한 쌍을 반환하므로, 이미지와 마스크 간의 공간적 일관성이 보장됩니다.

### 사용 가능한 변환 클래스

| 클래스 | 인자 | 적용 대상 |
|--------|------|-----------|
| `Resize(size)` | `size: int \| (H,W)` | 둘 다 (마스크는 `NEAREST` 보간) |
| `RandomCrop(size)` | `size: int \| (H,W)` | 둘 다 (동일한 크롭 좌표 사용) |
| `RandomHorizontalFlip(p=0.5)` | 플립 확률 | 둘 다 |
| `RandomVerticalFlip(p=0.5)` | 플립 확률 | 둘 다 |
| `RandomRotation(degrees)` | `int` 또는 `(min,max)` | 둘 다 |
| `ColorJitter(brightness, contrast, saturation, hue)` | 지터 강도 | **이미지만** |
| `ToTensor()` | — | 둘 다 — `float32` 변환, [0,1] 정규화, 채널 차원 앞으로 이동 |
| `Normalize(mean, std)` | ImageNet 통계값 | **이미지만** — 1채널 이미지도 올바르게 처리 |
| `Compose(transforms)` | 변환 리스트 | 둘 다 |

### Train vs Val/Test 파이프라인

모든 데이터셋이 따르는 공통 원칙:

| 단계 | 증강 | 리사이즈 | 정규화 |
|------|------|----------|--------|
| **train** | RandomCrop / Flip / Rotation / ColorJitter | 있음 | 있음 (ImageNet) |
| **val / test** | 없음 | Resize만 | 있음 (ImageNet) |

#### 데이터셋별 차이

**OCTA500** — train에서 `Resize` 생략 (입력이 이미 `image_size` 이상이라 가정; 작으면 패딩):
```
train:    RandomCrop → HFlip → VFlip → ColorJitter(0.4/0.4/0/0) → ToTensor → Normalize
val/test: Resize → ToTensor → Normalize
```

**DRIVE** — train에서 크롭 전에 `Resize` 먼저 적용:
```
train:    Resize → RandomCrop → HFlip → VFlip → ColorJitter(0.3/0.3/0/0) → ToTensor → Normalize
val/test: Resize → ToTensor → Normalize
```

**ISIC2018** — 피부 이미지 특성상 채도/색조 증강 포함:
```
train:    Resize → HFlip → VFlip → ColorJitter(0.3/0.3/0.2/0.05) → ToTensor → Normalize
val/test: Resize → ToTensor → Normalize
```

**MoNuSeg** — 조직병리 이미지 특성상 `RandomRotation(90)` 및 채도/색조 증강 추가:
```
train:    Resize → HFlip → VFlip → Rotation(90) → ColorJitter(0.3/0.3/0.2/0.05) → ToTensor → Normalize
val/test: Resize → ToTensor → Normalize
```

---

## 6. 마스크 이진화 (Mask Binarisation)

변환 파이프라인 이후, 모든 데이터셋에서 하드 임계값이 적용됩니다:

```python
lbl = (lbl > 0.5).float()   # 소프트 [0,1] float → 이진 {0.0, 1.0}
```

MoNuSeg는 PIL 이미지 구성 전에 `.npy` 레이블 배열을 추가로 정규화합니다:
```python
lbl_arr = (lbl_arr > 127).astype(np.uint8) * 255
```

---

## 7. Train/Val 분할

val 분할이 미리 제공되지 않는 두 데이터셋(DRIVE, MoNuSeg)은 동일한 전략을 사용합니다:

1. 전체 train 폴더 위에 **train transforms**를 적용한 `Dataset` 생성.
2. 고정 seed(`cfg["seed"]`, 기본값 42)로 인덱스 리스트를 셔플.
3. 앞 80%를 train, 뒤 20%를 val로 사용.
4. 동일한 폴더에 **val transforms**를 적용한 두 번째 `Dataset`을 생성한 뒤,
   `torch.utils.data.Subset`으로 val 인덱스만 사용하도록 제한.

이를 통해 val 분할은 항상 추론 시점과 동일한 파이프라인(증강 없음)을 사용하면서도
train 분할과 동일한 파일 목록을 공유합니다.

---

## 8. DataLoader 설정

| 파라미터 | Train | Val | Test |
|----------|-------|-----|------|
| `batch_size` | `cfg["batch_size"]` | 1 | 1 |
| `shuffle` | True | False | False |
| `drop_last` | True | False | False |
| `num_workers` | `cfg["num_workers"]` (기본값 4) | 동일 | 동일 |
| `pin_memory` | `cfg["pin_memory"]` (기본값 True) | 동일 | 동일 |

---

## 9. 전처리를 제어하는 Config 키

모든 키는 `configs/defaults.yaml`에 정의되어 있으며, 실험별로
`configs/exp_cards/`에서 덮어쓸 수 있습니다.

| 키 | 기본값 | 역할 |
|----|--------|------|
| `dataset` | `OCTA500-6M` | 사용할 데이터셋 로더 |
| `data_root` | `./data` | 데이터셋을 탐색할 루트 디렉토리 |
| `image_size` | `400` | 리사이즈/크롭 목표 크기 (정사각형) |
| `image_ext` | `null` | 이미지 확장자 — `null` = 자동 감지 |
| `label_ext` | `null` | 레이블 확장자 — `null` = 자동 감지 |
| `image_mean` | `[0.485, 0.456, 0.406]` | 정규화 평균 (ImageNet) |
| `image_std` | `[0.229, 0.224, 0.225]` | 정규화 표준편차 (ImageNet) |
| `batch_size` | `4` | Train 배치 크기 |
| `num_workers` | `4` | DataLoader 워커 프로세스 수 |
| `pin_memory` | `true` | GPU 전송 가속을 위한 메모리 고정 |
| `seed` | `42` | Train/Val 분할용 랜덤 시드 |

---

## 10. 전체 데이터 흐름 다이어그램

```
cfg["data_root"]
      │
      ▼
find_dataset_root()          ← 데이터셋 폴더 자동 탐색
      │
      ▼
Dataset.__init__()
  ├── 확장자 자동 감지 (_detect_ext)
  └── 파일 목록 구성 (natsorted / glob)
      │
      ▼
Dataset.__getitem__(idx)
  ├── PIL.Image.open(image)
  ├── PIL.Image.open(label).convert("L")
  ├── [MoNuSeg] np.load → PIL.fromarray
  ├── [OCTA500] image_size보다 작으면 패딩
  ├── transforms(img, mask)  ← Compose 파이프라인
  │       ├── Resize / RandomCrop
  │       ├── Flip / Rotation (train 전용)
  │       ├── ColorJitter (train 전용, 이미지만)
  │       ├── ToTensor → float32, [0,1], CHW
  │       └── Normalize (이미지만)
  └── (lbl > 0.5).float()   ← 마스크 이진화
      │
      ▼
DataLoader → (imgs: B×C×H×W, masks: B×1×H×W)
      │
      ▼
model(imgs, ...)
```
