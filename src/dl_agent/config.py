"""환경변수 기반 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv는 선택 의존성
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _i(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # llama-server (OpenAI 호환 엔드포인트)
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "sk-no-key-required")
    llm_model: str = os.environ.get("LLM_MODEL", "qwen3-4b-q4km")
    request_timeout: float = _f("LLM_TIMEOUT", 120.0)

    # 분류 하이퍼파라미터
    temperature: float = _f("CLS_TEMPERATURE", 0.0)
    top_logprobs: int = _i("CLS_TOP_LOGPROBS", 10)
    # 검증셋에서 고른 기권 임계값. 이 값보다 확신이 낮으면 시나리오 10으로 보낸다.
    abstain_threshold: float = _f("CLS_ABSTAIN_THRESHOLD", 0.7)
    # few-shot 예시 개수. 검증셋 실험 결과 0(zero-shot)이 가장 좋았다.
    # 예시를 늘릴수록 정확도가 떨어진다 — docs/evaluation.md 3.2 참고.
    few_shot_k: int = _i("CLS_FEW_SHOT_K", 0)

    # 데이터 경로
    data_dir: Path = ROOT / "benchmark" / "data"
    results_dir: Path = ROOT / "results"
    model_path: Path = Path(
        os.environ.get("GGUF_PATH", str(ROOT / "models" / "Qwen3-4B-Q4_K_M.gguf"))
    )

    # 스텁 도구가 남기는 감사 로그
    audit_log: Path = ROOT / "results" / "tool_audit.log"


settings = Settings()
