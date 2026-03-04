#!/usr/bin/env bash
# git_checkpoint.sh — 주기적 커밋 및 태그 생성 스크립트
#
# 사용법:
#   ./scripts/git_checkpoint.sh              # 변경사항 자동 커밋
#   ./scripts/git_checkpoint.sh --tag v0.4-results "결과 수렴 완료"
#   ./scripts/git_checkpoint.sh --push       # 커밋 후 origin push
#   ./scripts/git_checkpoint.sh --tag v1.0-paper-submission "논문 제출" --push
#
# 크론 예시 (매일 오전 9시):
#   0 9 * * * cd /data1/suhohan/vlm-segmentation && bash scripts/git_checkpoint.sh --push >> /tmp/git_checkpoint.log 2>&1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 인자 파싱 ──────────────────────────────────────────────────
TAG_NAME=""
TAG_MSG=""
DO_PUSH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG_NAME="$2"
            TAG_MSG="${3:-}"
            shift 2
            [[ -n "$TAG_MSG" && "$TAG_MSG" != "--"* ]] && shift
            ;;
        --push) DO_PUSH=true; shift ;;
        *) shift ;;
    esac
done

# ── 스테이징 대상 (runs/, data/, *.pt 등 제외) ──────────────────
STAGE_PATHS=(
    "src/"
    "configs/"
    "tests/"
    "scripts/"
    "docs/"
    "project_context/"
    "train.py"
    "eval.py"
    "CLAUDE.md"
    "GEMINI.md"
)

git add "${STAGE_PATHS[@]}" 2>/dev/null || true

# 삭제된 추적 파일 처리
git add -u 2>/dev/null || true

# ── 변경사항 확인 ───────────────────────────────────────────────
if git diff --cached --quiet; then
    echo "[checkpoint] 변경사항 없음 — 커밋 스킵"
else
    # 변경 파일 요약
    CHANGED=$(git diff --cached --name-only | head -20 | tr '\n' ' ')
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    SHORT_HASH=$(git rev-parse --short HEAD)

    git commit -m "chore: checkpoint ${TIMESTAMP} [${SHORT_HASH}]

Changed: ${CHANGED}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

    echo "[checkpoint] 커밋 완료: $(git rev-parse --short HEAD)"
fi

# ── 태그 생성 ───────────────────────────────────────────────────
if [[ -n "$TAG_NAME" ]]; then
    if git tag -l | grep -q "^${TAG_NAME}$"; then
        echo "[tag] '${TAG_NAME}' 이미 존재 — 스킵"
    else
        FULL_MSG="${TAG_MSG:-${TAG_NAME}}"
        git tag -a "$TAG_NAME" -m "$FULL_MSG"
        echo "[tag] '${TAG_NAME}' 생성 완료"
        if $DO_PUSH; then
            git push origin "$TAG_NAME"
            echo "[push] 태그 push 완료"
        fi
    fi
fi

# ── Push ────────────────────────────────────────────────────────
if $DO_PUSH; then
    git push origin "$(git branch --show-current)"
    echo "[push] origin push 완료"
fi

echo "[done] $(date '+%Y-%m-%d %H:%M:%S')"
