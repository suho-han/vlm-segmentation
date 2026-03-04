# 모델 구조 및 VLM 주입 상세 (Model Architecture & VLM Injection)

본 프로젝트의 핵심 모델인 `SwinUNETR2DVLM`과 VLM 정보를 추출하는 `VLMPrior`의 기술적 구조를 설명합니다.

## 1. VLMPrior (VLM 특징 추출 모듈)

VLM(BiomedCLIP 또는 CLIP)으로부터 두 가지 형태의 정보를 추출합니다.

- **Text Embedding (V0):** 데이터셋에 특화된 프롬프트(예: "retinal vessel segmentation in OCTA fundus image")를 텍스트 인코더에 통과시켜 (1, 512) 크기의 L2 정규화된 벡터를 생성합니다.
- **Image Spatial Features (V1):** 입력 이미지를 VLM의 시각 인코더(ViT)에 통과시켜 중간 레이어의 패치 토큰(Patch Tokens)을 추출합니다. 이는 (B, 512, 14, 14) 크기의 공간적 특징 맵으로 변환되어 모델에 전달됩니다.

## 2. SwinUNETR2DVLM 구조

MONAI의 SwinUNETR 2D 버전을 기반으로, 디코더 단계에서 VLM 텍스트 임베딩을 주입하는 구조입니다.

### Gated Fusion (게이트 융합)

디코더의 각 단계(dec3, dec2, dec1, dec0) 직후에 **채널별 게이팅 모듈(\_GatedFusion)**이 추가됩니다.

- **작동 원리:**
  1. 텍스트 임베딩을 선형 레이어(Linear)를 통해 해당 디코더 레이어의 채널 수만큼 투영합니다.
  2. `tanh` 활성화 함수를 적용하여 -1에서 1 사이의 게이트 값을 생성합니다.
  3. `out = feat * (1 + gate)` 식을 통해 디코더 특징 맵을 변조합니다.
- **초기화 전략:** 선형 레이어의 가중치와 편향을 0으로 초기화하여, 학습 초기에는 `gate=tanh(0)=0`이 되어 원본 특징 맵이 그대로 전달(Identity Mapping)되도록 설계되었습니다.

### 주입 위치 및 차원 (feature_size=48 기준)

- **dec3:** 384 채널 (1/16 해상도)
- **dec2:** 192 채널 (1/8 해상도)
- **dec1:** 96 채널 (1/4 해상도)
- **dec0:** 48 채널 (1/2 해상도)

## 3. 손실 함수 및 평가지표

- **Loss:** Binary Cross Entropy(BCE)와 Dice Loss를 결합하여 픽셀 정확도를 높입니다.
- **Topology Metrics:**
  - **HD95:** 경계선 예측의 최대 거리 오차를 측정.
  - **Betti Numbers (β0, β1):** 혈관의 연결성(Connected Components)과 구멍(Holes)의 개수를 분석하여 구조적 무결성을 평가합니다.
