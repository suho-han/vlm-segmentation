# Claude Code 상태바 설정 가이드

Claude Code 하단 상태바에 **세션 사용률 / 주간 사용률 / 컨텍스트 / 모델명**을 표시하는 방법입니다.

## 결과물 미리보기

```
Sess: 42% | Week: 18% | Ctx: 87% | claude-sonnet-4-6
```

| 항목        | 의미                           |
| ----------- | ------------------------------ |
| `Sess`      | 5시간 윈도우 내 토큰 사용률    |
| `Week`      | 7일 누적 사용률                |
| `Ctx`       | 현재 대화의 남은 컨텍스트 비율 |
| 마지막 필드 | 현재 사용 중인 모델명          |

---

## 설치 방법

### 1단계: 스크립트 파일 생성

```bash
cat > ~/.claude/statusline.sh << 'EOF'
#!/bin/bash
input=$(cat)

# 기본 정보 (status line JSON에서 추출)
model=$(echo "$input" | jq -r '.model.display_name')
remaining=$(echo "$input" | jq -r '.context_window.remaining_percentage // 0')

remaining_int=${remaining%.*}

# Usage Limit (API 호출, 30초 캐시)
CACHE_FILE="/tmp/claude-usage-cache.json"
CACHE_MAX_AGE=30

usage_session="?"
usage_week="?"

# 캐시 확인
if [ -f "$CACHE_FILE" ]; then
  cache_age=$(( $(date +%s) - $(stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0) ))
else
  cache_age=999
fi

if [ "$cache_age" -gt "$CACHE_MAX_AGE" ]; then
  TOKEN=$(jq -r '.claudeAiOauth.accessToken // empty' \
    "$HOME/.claude/.credentials.json" 2>/dev/null)
  if [ -n "$TOKEN" ] && [ "$TOKEN" != "null" ]; then
    RESP=$(curl -s --max-time 3 -X GET \
      "https://api.anthropic.com/api/oauth/usage" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/json" \
      -H "anthropic-beta: oauth-2025-04-20" 2>/dev/null)
    if echo "$RESP" | jq -e '.five_hour' > /dev/null 2>&1; then
      echo "$RESP" > "$CACHE_FILE"
    fi
  fi
fi

if [ -f "$CACHE_FILE" ]; then
  usage_session=$(jq -r '.five_hour.utilization // "?"' "$CACHE_FILE")
  usage_week=$(jq -r '.seven_day.utilization // "?"' "$CACHE_FILE")
fi

echo "Sess: ${usage_session}% | Week: ${usage_week}% | Ctx: ${remaining_int}% | ${model}"
EOF
```

### 2단계: 실행 권한 부여

```bash
chmod +x ~/.claude/statusline.sh
```

### 3단계: Claude Code 설정에 등록

`~/.claude/settings.json` 파일을 열어 아래 내용을 추가합니다.

```json
{
  "statusLine": {
    "type": "command",
    "command": "bash /home/<YOUR_USERNAME>/.claude/statusline.sh"
  }
}
```

> `<YOUR_USERNAME>` 부분을 실제 리눅스 유저명으로 교체하세요 (예: `minkyukim`).

---

## 의존성

| 패키지 | 용도           |
| ------ | -------------- |
| `jq`   | JSON 파싱      |
| `curl` | Usage API 호출 |

대부분의 리눅스 배포판에 기본 설치되어 있습니다. 없다면:

```bash
# Ubuntu/Debian
sudo apt install jq curl

# RHEL/CentOS
sudo yum install jq curl
```

---

## 동작 원리

1. Claude Code가 상태바를 갱신할 때마다 `statusline.sh`를 실행하며 JSON을 stdin으로 전달합니다.
2. 스크립트는 JSON에서 **모델명**과 **컨텍스트 잔여량**을 추출합니다.
3. `~/.claude/.credentials.json`의 OAuth 토큰으로 `api.anthropic.com/api/oauth/usage`를 호출해 **세션/주간 사용률**을 가져옵니다.
4. API 응답은 `/tmp/claude-usage-cache.json`에 30초간 캐시되어 불필요한 네트워크 요청을 방지합니다.

---

## 주의사항

- `Sess` / `Week`가 `?`로 표시되면 OAuth 토큰이 없거나 만료된 것입니다. `claude` 명령으로 재로그인하세요.
