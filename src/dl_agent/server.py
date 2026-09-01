"""FastAPI 웹 채팅 서버.

응답만 주는 게 아니라 에이전트의 **판단 근거**(분류 확률, 실행 경로, 도구
호출 인자)를 같이 내려준다. 데모에서 "왜 이렇게 답했나"를 화면으로 보여줄 수
있어야 하기 때문이다.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import settings
from .graph import build_graph
from .llm import LlamaClient
from .schema import BY_ID, SCENARIOS

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="DL 문의 대응 에이전트", version="0.1.0")

_client = LlamaClient()
_graph = build_graph(client=_client)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    scenario: int
    scenario_name: str
    action: str
    confidence: float
    abstained: bool
    top_probs: list[dict[str, Any]]
    slots: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    trace: list[str]
    elapsed_ms: float


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "llm_backend": settings.llm_base_url,
        "llm_reachable": _client.healthy(),
        "model": settings.llm_model,
        "abstain_threshold": settings.abstain_threshold,
        "few_shot_k": settings.few_shot_k,
    }


@app.get("/api/scenarios")
def scenarios() -> list[dict[str, Any]]:
    return [
        {"id": s.id, "label": s.label, "name": s.name, "action": s.action.value, "summary": s.summary}
        for s in SCENARIOS
    ]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    state = _graph.invoke({"inquiry": req.message})
    elapsed = (time.perf_counter() - started) * 1000

    cls = state.get("classification") or {}
    probs: dict[str, float] = cls.get("probs") or {}
    top = [
        {
            "label": label,
            "scenario": next((s.id for s in SCENARIOS if s.label == label), 0),
            "name": next((s.name for s in SCENARIOS if s.label == label), label),
            "prob": round(p, 4),
        }
        for label, p in sorted(probs.items(), key=lambda kv: -kv[1])[:4]
    ]

    scenario_id = state.get("scenario", 10)
    return ChatResponse(
        response=state.get("response", ""),
        scenario=scenario_id,
        scenario_name=BY_ID[scenario_id].name,
        action=state.get("action", ""),
        confidence=round(float(cls.get("confidence") or 0.0), 4),
        abstained=bool(cls.get("abstained")),
        top_probs=top,
        slots=dict(state.get("slots") or {}),
        tool_calls=[
            {"name": c["name"], "args": c["args"], "result": c["result"]}
            for c in (state.get("tool_calls") or [])
        ],
        trace=list(state.get("trace") or []),
        elapsed_ms=round(elapsed, 1),
    )
