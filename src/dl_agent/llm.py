"""llama-server(OpenAI 호환) 클라이언트.

핵심 아이디어
-------------
분류를 "자유 텍스트 생성"이 아니라 **단일 토큰 선택 문제**로 바꾼다.

* GBNF 문법 ``root ::= [A-J]`` 로 출력 어휘를 10개 레이블로 강제한다.
  → 파싱 실패율이 구조적으로 0이 된다.
* ``logprobs`` / ``top_logprobs`` 로 그 한 토큰의 확률분포를 그대로 받는다.
  → 시나리오 10개에 대한 진짜 사후확률을 얻고, 이를 신뢰도로 써서
    "확신 없으면 사람에게 넘긴다"(기권)를 임계값 하나로 제어할 수 있다.

Qwen3는 기본이 thinking 모드라 첫 토큰이 ``<think>`` 가 되어 문법과 충돌한다.
``chat_template_kwargs.enable_thinking=false`` 와 ``/no_think`` 소프트 스위치를
함께 걸어 비활성화한다.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx

from .config import settings
from .schema import LABELS

LABEL_GRAMMAR = "root ::= [{first}-{last}]".format(first=LABELS[0], last=LABELS[-1])


class LLMUnavailable(RuntimeError):
    """llama-server에 붙지 못했을 때."""


@dataclass
class LabelPrediction:
    label: str
    confidence: float
    probs: dict[str, float]
    raw: str
    latency_ms: float
    usage: dict[str, int]


def _softmax_from_logprobs(pairs: Iterable[tuple[str, float]]) -> dict[str, float]:
    """top_logprobs 조각을 레이블 집합 위에서 재정규화한다.

    top_logprobs는 상위 n개만 오므로 합이 1이 아니다. 레이블에 해당하는
    항목만 남기고 다시 정규화해야 "10개 시나리오 중 어디"라는 질문에 대한
    확률이 된다.
    """
    kept = {label: lp for label, lp in pairs if label in LABELS}
    if not kept:
        return {}
    top = max(kept.values())
    exp = {k: math.exp(v - top) for k, v in kept.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in sorted(exp.items(), key=lambda kv: -kv[1])}


class LlamaClient:
    """llama-server / ollama 공통 OpenAI 호환 클라이언트."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        use_grammar: bool = True,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.timeout = timeout or settings.request_timeout
        self.use_grammar = use_grammar
        self._client = httpx.Client(timeout=self.timeout)

    # -- 헬스체크 ----------------------------------------------------------

    def healthy(self) -> bool:
        for path in ("/models", "/../health"):
            try:
                r = self._client.get(f"{self.base_url}{path}", headers=self._headers())
                if r.status_code < 400:
                    return True
            except httpx.HTTPError:
                continue
        return False

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    # -- 분류 --------------------------------------------------------------

    def classify_label(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> LabelPrediction:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.temperature if temperature is None else temperature,
            "max_tokens": 4,
            "logprobs": True,
            "top_logprobs": settings.top_logprobs,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.use_grammar:
            payload["grammar"] = LABEL_GRAMMAR

        started = time.perf_counter()
        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - 네트워크 경로
            raise LLMUnavailable(f"{self.base_url}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        data = r.json()
        choice = data["choices"][0]
        raw = (choice["message"].get("content") or "").strip()

        probs = self._extract_probs(choice)
        label = self._pick_label(raw, probs)
        confidence = probs.get(label, 1.0 if label else 0.0)

        usage = data.get("usage") or {}
        return LabelPrediction(
            label=label,
            confidence=confidence,
            probs=probs,
            raw=raw,
            latency_ms=latency_ms,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )

    @staticmethod
    def _extract_probs(choice: dict[str, Any]) -> dict[str, float]:
        content = ((choice.get("logprobs") or {}).get("content")) or []
        for item in content:  # 레이블을 담은 첫 토큰을 찾는다
            tops = item.get("top_logprobs") or []
            pairs = [(t.get("token", "").strip(), t.get("logprob", -99.0)) for t in tops]
            probs = _softmax_from_logprobs(pairs)
            if probs:
                return probs
        return {}

    @staticmethod
    def _pick_label(raw: str, probs: dict[str, float]) -> str:
        """문법이 없는 백엔드(ollama 등)를 위한 관대한 파서."""
        if probs:
            return max(probs, key=probs.__getitem__)
        m = re.search(rf"[{LABELS[0]}-{LABELS[-1]}]", raw.upper())
        return m.group(0) if m else ""

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LlamaClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
