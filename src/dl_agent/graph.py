"""LangGraph 파이프라인 조립.

    START
      ↓
    ingest      문의 정규화 + 규칙 기반 슬롯 추출 (DL 코드/메일/시간/제목)
      ↓
    classify    Qwen3-4B(llama-server) 단일 토큰 분류 + 확률 기반 기권
      ↓
    route       ── action == fixed_response ─→ act_fixed
                ── action == draft_approval ─→ act_draft
                └─ action == forward_mail   ─→ act_forward
      ↓
    compose     템플릿 + 도구 결과로 최종 답변 조립
      ↓
     END

경량 LLM은 `classify` 한 노드에서만 쓰인다. 나머지는 전부 결정론적이다.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from . import responses, tools
from .config import settings
from .extract import extract_slots
from .llm import LabelPrediction, LlamaClient, LLMUnavailable
from .prompts import DEFAULT_PROMPT_VERSION, build_messages
from .schema import (
    BY_ID,
    BY_LABEL,
    FALLBACK_SCENARIO_ID,
    ActionType,
    AgentState,
    Classification,
    ToolCall,
)

# 시나리오 → 기안 서식 종류
DRAFT_TYPE = {1: "dl_rename", 8: "dynamic_dl", 9: "dl_api_acl"}
# 시나리오 → 위키 토픽
WIKI_TOPIC = {5: "sender_permission", 9: "rest_api"}


# --- 노드 -------------------------------------------------------------------


def ingest(state: AgentState) -> dict[str, Any]:
    inquiry = (state.get("inquiry") or "").strip()
    slots = extract_slots(inquiry)
    found = ", ".join(f"{k}={v}" for k, v in slots.items()) or "없음"
    return {"inquiry": inquiry, "slots": slots, "trace": [f"ingest: 슬롯 {found}"]}


def make_classify(
    client: LlamaClient,
    few_shot_k: int | None = None,
    few_shot_style: str = "inline",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
):
    """분류 노드를 만든다. 클라이언트를 주입받아 벤치마크에서 재사용한다."""

    def classify(state: AgentState) -> dict[str, Any]:
        inquiry = state["inquiry"]
        messages = build_messages(
            inquiry, k=few_shot_k, style=few_shot_style, version=prompt_version
        )
        try:
            pred: LabelPrediction = client.classify_label(messages)
        except LLMUnavailable as exc:
            cls: Classification = {
                "scenario": FALLBACK_SCENARIO_ID,
                "label": "J",
                "confidence": 0.0,
                "probs": {},
                "abstained": True,
                "raw": f"LLM_UNAVAILABLE: {exc}",
                "latency_ms": 0.0,
                "backend": client.base_url,
            }
            return {
                "classification": cls,
                "scenario": FALLBACK_SCENARIO_ID,
                "action": ActionType.FORWARD_MAIL.value,
                "trace": ["classify: LLM 연결 실패 → 관리자 전달로 폴백"],
            }

        scenario_obj = BY_LABEL.get(pred.label)
        scenario_id = scenario_obj.id if scenario_obj else FALLBACK_SCENARIO_ID

        abstained = pred.confidence < settings.abstain_threshold
        if abstained:
            scenario_id = FALLBACK_SCENARIO_ID

        cls = {
            "scenario": scenario_id,
            "label": pred.label,
            "confidence": pred.confidence,
            "probs": pred.probs,
            "abstained": abstained,
            "raw": pred.raw,
            "latency_ms": pred.latency_ms,
            "backend": client.base_url,
        }
        note = (
            f"classify: {pred.label}({pred.confidence:.2f}) "
            f"{'→ 확신 부족으로 기권, 시나리오 10' if abstained else f'→ 시나리오 {scenario_id}'}"
            f" [{pred.latency_ms:.0f}ms]"
        )
        return {
            "classification": cls,
            "scenario": scenario_id,
            "action": BY_ID[scenario_id].action.value,
            "trace": [note],
        }

    return classify


def route(state: AgentState) -> str:
    return state.get("action") or ActionType.FORWARD_MAIL.value


def act_fixed(state: AgentState) -> dict[str, Any]:
    """고정 응답 시나리오(2·3·4·5). 필요한 경우에만 조회성 도구를 부른다."""
    scenario_id = state["scenario"]
    calls: list[ToolCall] = []
    trace: list[str] = []

    if scenario_id == 4:
        for code in (state.get("slots") or {}).get("dl_codes", [])[:1]:
            result = tools.dl_api_get(code)
            calls.append({"name": "dl_api_get", "args": {"dl_code": code}, "result": result})
            trace.append(f"tool: dl_api_get({code}) → 외부메일수신={result['external_mail_receive']}")

    topic = WIKI_TOPIC.get(scenario_id)
    if topic:
        result = tools.get_wiki_link(topic)
        calls.append({"name": "get_wiki_link", "args": {"topic": topic}, "result": result})
        trace.append(f"tool: get_wiki_link({topic})")

    if not trace:
        trace.append("act: 도구 호출 없이 고정 응답")
    return {"tool_calls": calls, "trace": trace}


def act_draft(state: AgentState) -> dict[str, Any]:
    """기안 결재 시나리오(1·8·9)."""
    scenario_id = state["scenario"]
    slots = state.get("slots") or {}
    calls: list[ToolCall] = []
    trace: list[str] = []

    if scenario_id == 9:
        result = tools.get_wiki_link("rest_api")
        calls.append({"name": "get_wiki_link", "args": {"topic": "rest_api"}, "result": result})
        trace.append("tool: get_wiki_link(rest_api)")

    request_type = DRAFT_TYPE.get(scenario_id, "dl_rename")
    prefill: dict[str, Any] = {}
    if slots.get("dl_codes"):
        prefill["dl_code"] = slots["dl_codes"][0]
    if slots.get("dl_addresses"):
        prefill["mail_address"] = slots["dl_addresses"][0]

    result = tools.open_draft_form(request_type, prefill)
    calls.append(
        {
            "name": "open_draft_form",
            "args": {"request_type": request_type, "prefill": prefill},
            "result": result,
        }
    )
    trace.append(f"tool: open_draft_form({request_type}) → {result['ticket_id']}")
    return {"tool_calls": calls, "trace": trace}


def act_forward(state: AgentState) -> dict[str, Any]:
    """메일 전달 시나리오(6·7·10)."""
    scenario_id = state["scenario"]
    inquiry = state["inquiry"]

    if scenario_id == 7:
        from .extract import missing_for_mail_triage

        missing = missing_for_mail_triage(state.get("slots") or {})
        result = tools.forward_to_works_cs(inquiry, missing)
        args = {"to": result["to"], "missing": missing}
    else:
        result = tools.forward_to_admin(inquiry, BY_ID[scenario_id].name)
        args = {"to": result["to"], "scenario_name": BY_ID[scenario_id].name}

    call: ToolCall = {"name": "send_mail", "args": args, "result": result}
    return {"tool_calls": [call], "trace": [f"tool: send_mail → {result['to']}"]}


def compose(state: AgentState) -> dict[str, Any]:
    cls = state.get("classification") or {}
    text = responses.compose(
        scenario_id=state["scenario"],
        inquiry=state["inquiry"],
        slots=state.get("slots") or {},
        tool_calls=state.get("tool_calls") or [],
        confidence=float(cls.get("confidence") or 0.0),
        abstained=bool(cls.get("abstained")),
    )
    return {"response": text, "trace": ["compose: 응답 조립 완료"]}


# --- 그래프 -----------------------------------------------------------------


def build_graph(
    client: LlamaClient | None = None,
    few_shot_k: int | None = None,
    few_shot_style: str = "inline",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
):
    client = client or LlamaClient()
    g = StateGraph(AgentState)

    g.add_node("ingest", ingest)
    g.add_node("classify", make_classify(client, few_shot_k, few_shot_style, prompt_version))
    g.add_node("act_fixed", act_fixed)
    g.add_node("act_draft", act_draft)
    g.add_node("act_forward", act_forward)
    g.add_node("compose", compose)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "classify")
    g.add_conditional_edges(
        "classify",
        route,
        {
            ActionType.FIXED_RESPONSE.value: "act_fixed",
            ActionType.DRAFT_APPROVAL.value: "act_draft",
            ActionType.FORWARD_MAIL.value: "act_forward",
        },
    )
    for node in ("act_fixed", "act_draft", "act_forward"):
        g.add_edge(node, "compose")
    g.add_edge("compose", END)

    return g.compile()


def run(inquiry: str, graph=None) -> AgentState:
    graph = graph or build_graph()
    return graph.invoke({"inquiry": inquiry})
