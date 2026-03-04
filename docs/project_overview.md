# 프로젝트 개요 및 흐름 (Project Overview & Flow)

본 프로젝트는 냉동된(Frozen) 시각-언어 모델(VLM, 특히 BiomedCLIP)의 특징 정보를 주입하여, 의료용 2D 혈관 세그멘테이션(Vessel Segmentation)의 성능을 향상시키는 것을 목표로 합니다. 단순한 픽셀 단위 정확도(Dice/IoU)를 넘어, 혈관의 연속성과 구조적 무결성을 평가하는 지표(HD95, Betti number)를 최적화하는 데 집중합니다.

## 1. 주요 목표 (Core Mission)

- **VLM 특징 주입:** BiomedCLIP과 같은 도메인 특화 VLM의 풍부한 시각/언어 표현을 세그멘테이션 모델에 결합.
- **구조적 성능 개선:** 혈관의 끊어짐(thin-structure dropout)이나 비정상적인 구멍(hole artifacts)을 줄이기 위해 위상학적 지표(Betti β0, β1)를 관리.
- **백본 확장성:** SwinUNETR, UNet++, TransUNet, SegFormer 등 다양한 최신 백본에 VLM 모듈을 유연하게 결합.

## 2. 전체 연구 및 실험 프로세스

### 단계 1: 논문 분석 및 실패 사례 도출 (Task G1)
- 관련 논문(Open-Vocabulary, Feature Injection 등)의 기법을 분석하고 `papers/` 폴더에 정리합니다.
- 의료 영상에서 혈관이 끊기거나 노이즈가 발생하는 **Failure Mode**를 정의하고, 이를 해결하기 위한 VLM 활용 방안을 도출합니다.

### 단계 2: 실험 수행 및 데이터 관리 (Task G2)
- `train.py`와 `configs/exp_cards/`를 통해 다양한 실험 변수(VLM 주입 위치, 게이팅 방식 등)를 테스트합니다.
- `runs/` 디렉토리에 실험 결과(모델 가중치, 예측 결과, 지표 JSON)를 저장합니다.

### 단계 3: 결과 집계 및 회귀 분석
- 모든 실험 결과를 `reports/runs_summary.md`로 취합합니다.
- Dice 점수는 높지만 위상(Topology) 지표가 나빠지는 "최적화 충돌(Optimization Conflict)" 사례를 감지하고 분석합니다.

### 단계 4: QC(품질 관리) 모듈 설계 (Task G4)
- 추론 시점에서 세그멘테이션의 신뢰도를 평가할 수 있는 QC 스코어링(정렬 기반, 불확실성 기반 등)을 설계하고 적용합니다.

## 3. 지원 데이터셋

- **OCTA500-6M:** 망막 광간섭 단층 촬영 혈관 조영술(OCTA) 이미지.
- **DRIVE:** 망막 안저 사진 혈관 데이터셋.
- 기타 ISIC, MoNuSeg 등 확장 가능.
