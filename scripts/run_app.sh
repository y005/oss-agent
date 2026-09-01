#!/usr/bin/env bash
# 웹 채팅 서버를 띄운다. llama-server 가 먼저 떠 있어야 한다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! curl -sf -m 3 "${LLM_BASE_URL:-http://127.0.0.1:8080/v1}/models" >/dev/null 2>&1; then
  echo "⚠️  llama-server 응답이 없습니다. 다른 터미널에서 ./scripts/serve_llm.sh 를 먼저 실행하세요." >&2
fi

exec .venv/bin/python -m uvicorn dl_agent.server:app --host 127.0.0.1 --port "${APP_PORT:-8000}" "$@"
