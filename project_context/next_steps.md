# Next Steps

_Last updated: 2026-03-04 03:00_

---

## 전체 실험 완료 현황

### 완료 ✅

| Dataset     | 완료된 실험 |
|-------------|-------------|
| OCTA500-6M  | B0~B4, B3-VLM, B4-VLM†, V0, V1, V1-Topology |
| DRIVE       | B0~B4, B3-VLM, B4-VLM†, V0, V1, V1-Topology |
| MoNuSeg     | B0~B4, V0, V1, V1-Topology |
| ISIC2018    | B0, B1, V3-SegFormer-VLM-Topology |

† OCTA6M/DRIVE B4-VLM: `vlm_prior` 키가 train.py에 연결 안 됨 → 실제로는 VLM 없이 학습됨 (plain TransUNet). 결과가 B4-baseline과 거의 동일한 것으로 확인.

### 진행 중 🟡

| 실험 | 상태 |
|------|------|
| ISIC2018-B2-nnUNet | epoch ~29, GPU 0 |

### 자동 큐 ⏳

**Main queue** (`/tmp/isic2018_remaining.log`, PID 3059454):
```
B3-SegFormer → B4-TransUNet → V0-SwinUNETR-VLM → V1-SwinUNETR-VLM
```

**Post-queue** (`/tmp/post_queue.log`, PID 3414215):
```
V1-SwinUNETR-VLM-Topology 학습
→ 전체 ISIC2018 eval (B0~V1, V1-Topo, V3-SegFormer-Topo)
→ update_results.py → visualize_compare.py (4 datasets)
```

**Follow-up queue** (`/tmp/run_followup.sh`, PID — see below):
```
MoNuSeg-B3-SegFormer-VLM 학습+eval
→ MoNuSeg-B4-TransUNet-VLM 학습+eval
→ ISIC2018-B3-SegFormer-VLM 학습+eval
→ ISIC2018-B4-TransUNet-VLM 학습+eval
→ update_results.py → visualize_compare.py (전체 재생성)
```

---

## 남은 작업 및 이유

### 1. MoNuSeg B3/B4 VLM variants (일관성)

OCTA/DRIVE는 C1 작업으로 B3-SegFormer-VLM, B4-TransUNet-VLM 결과 있음.
MoNuSeg와 ISIC2018는 없음 → cross-dataset VLM 효과 비교를 위해 필요.

Configs 생성 완료:
- `configs/exp_cards/MoNuSeg-B3-SegFormer-VLM.yaml`
- `configs/exp_cards/MoNuSeg-B4-TransUNet-VLM.yaml`
- `configs/exp_cards/ISIC2018-B3-SegFormer-VLM.yaml`
- `configs/exp_cards/ISIC2018-B4-TransUNet-VLM.yaml`

### 2. OCTA6M/DRIVE B4-TransUNet-VLM 재실행 (버그 수정)

`vlm_prior: "image"` 키가 train.py에서 무시됨 → 실제 VLM 미적용.
재실행 필요:
- `OCTA6M-B4-TransUNet-VLM` (use_vlm: true로 수정)
- `DRIVE-B4-TransUNet-VLM` (use_vlm: true로 수정)

### 3. results.md 최종 정리 (자동)

post-queue + follow-up queue 완료 후 `scripts/update_results.py` 실행됨.

---

## 수동 실행 명령 (자동 큐 실패 시)

```bash
# ISIC2018 eval (B3~V1)
for exp in ISIC2018-B3-SegFormer ISIC2018-B4-TransUNet ISIC2018-V0-SwinUNETR-VLM \
           ISIC2018-V1-SwinUNETR-VLM ISIC2018-V1-SwinUNETR-VLM-Topology; do
  model=$(grep "^model:" configs/exp_cards/${exp}.yaml | awk '{print $2}' | tr -d '"')
  CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python eval.py \
    --config configs/exp_cards/${exp}.yaml \
    --ckpt runs/ISIC2018/${model}/${exp}/ckpt/best.pt \
    --exp_id $exp --outdir runs --data_root ./data
done

# MoNuSeg/ISIC2018 B3/B4 VLM train
for exp in MoNuSeg-B3-SegFormer-VLM MoNuSeg-B4-TransUNet-VLM \
           ISIC2018-B3-SegFormer-VLM ISIC2018-B4-TransUNet-VLM; do
  CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py \
    --config configs/exp_cards/${exp}.yaml \
    --exp_id $exp --epochs 1000 --patience 50 --data_root ./data --outdir runs
done

# results.md 업데이트
env -u VIRTUAL_ENV uv run python scripts/update_results.py
```

---

## 모니터링

```bash
tail -f /tmp/isic2018_remaining.log   # main queue
tail -f /tmp/post_queue.log           # step 2,1,3,4
tail -f /tmp/followup.log             # MoNuSeg/ISIC2018 B3/B4 VLM
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```
