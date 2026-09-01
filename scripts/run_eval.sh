#!/usr/bin/env bash
# 전체 실험 재현: 검증셋 탐색 → 최종 테스트 1회.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python

$PY benchmark/split.py

echo "── 베이스라인 (규칙 기반, LLM 없음)"
$PY benchmark/run_eval.py --split val --system keyword --tag val-keyword >/dev/null

echo "── 검증셋: few-shot 개수 / 삽입 방식 / 프롬프트 버전 탐색"
run_val() {  # k, style, prompt
  $PY benchmark/run_eval.py --split val --few-shot "$1" --few-shot-style "$2" --prompt "$3" \
      --tag "val-$3-k$1-$2" >/dev/null
}
# 프롬프트(v0/v1/v2) × few-shot(k=0/10/20) 이중 ablation
run_val  0 inline v0
run_val 10 inline v0
run_val  0 inline v1
run_val 10 inline v1
run_val  0 inline v2
run_val 10 inline v2
run_val 20 inline v2
run_val 10 chat   v2   # 예시 삽입 방식 비교 (대화 턴 vs 시스템 프롬프트)

echo "── 검증셋: 선택 설정에 기권 적용"
$PY benchmark/run_eval.py --split val --few-shot 0 --prompt v2 --threshold 0.7 \
    --tag val-v2-k0-tau07 >/dev/null

echo "── 검증셋: 상위 모델 참조 (ollama qwen2.5:7b)"
$PY benchmark/run_eval.py --split val --system ollama --model qwen2.5:7b --few-shot 10 --prompt v2 \
    --tag val-ollama-qwen25-7b >/dev/null || echo "  (ollama 미기동 — 건너뜀)"

$PY benchmark/compare.py val

echo
echo "── 최종 테스트 (검증셋에서 고른 설정: v2, k=0, τ=0.7)"
$PY benchmark/run_eval.py --split test --few-shot 0 --prompt v2 --threshold "${TAU:-0.7}" \
    --tag test-final
$PY benchmark/run_eval.py --split test --few-shot 0 --prompt v2 --threshold 0.0 \
    --tag test-final-tau00 >/dev/null
$PY benchmark/run_eval.py --split test --system keyword --tag test-keyword >/dev/null
$PY benchmark/run_eval.py --split test --system ollama --model qwen2.5:7b --few-shot 0 --prompt v2 \
    --tag test-ollama-qwen25-7b >/dev/null || echo "  (ollama 미기동 — 건너뜀)"

$PY benchmark/compare.py test
echo
$PY benchmark/analyze.py
