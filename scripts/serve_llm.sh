#!/usr/bin/env bash
# Qwen3-4B Q4_K_M 를 llama-server(OpenAI 호환)로 띄운다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF="${GGUF_PATH:-$ROOT/models/Qwen3-4B-Q4_K_M.gguf}"
PORT="${LLM_PORT:-8080}"

if [[ ! -f "$GGUF" ]]; then
  echo "GGUF 파일이 없습니다: $GGUF" >&2
  echo "  huggingface.co/Qwen/Qwen3-4B-GGUF 에서 Qwen3-4B-Q4_K_M.gguf 를 받아 models/ 에 두세요." >&2
  exit 1
fi

exec llama-server \
  --model "$GGUF" \
  --alias qwen3-4b-q4km \
  --host 127.0.0.1 --port "$PORT" \
  --ctx-size 8192 \
  --n-gpu-layers 99 \
  --parallel 1 \
  --temp 0 \
  --jinja \
  --log-file "$ROOT/results/llama-server.log"
