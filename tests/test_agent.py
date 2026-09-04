"""LLM 없이 돌아가는 스모크 테스트.

규칙 기반 분류기를 그래프에 꽂아 라우팅·도구 호출·응답 조립이 시나리오
정의와 어긋나지 않는지 확인한다. llama-server가 없어도 실행된다.

    .venv/bin/python -m pytest tests/ -q
    .venv/bin/python tests/test_agent.py        # pytest 없이도 실행 가능
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from baselines import KeywordClient  # noqa: E402

from dl_agent.extract import extract_slots  # noqa: E402
from dl_agent.graph import build_graph  # noqa: E402
from dl_agent.schema import BY_ID, LABELS, SCENARIOS, ActionType  # noqa: E402

GRAPH = build_graph(client=KeywordClient(), few_shot_k=0)


def test_scenario_table_is_consistent() -> None:
    assert len(SCENARIOS) == 10
    assert len(set(LABELS)) == 10, "레이블 중복"
    assert LABELS == tuple("ABCDEFGHIJ")
    assert all(BY_ID[s.id] is s for s in SCENARIOS)


def test_slot_extraction() -> None:
    slots = extract_slots(
        "vendor@partner.co.kr 에서 DL345678 로 8/28 14:20 에 보낸 '견적서 회신' 메일이 안 옵니다."
    )
    assert slots["dl_codes"] == ["DL345678"]
    assert "vendor@partner.co.kr" in slots["emails"]
    assert slots["sent_at"].startswith("8/28")
    assert slots["subject"] == "견적서 회신"


def test_dl_code_variants() -> None:
    assert extract_slots("DL 123456 확인 부탁")["dl_codes"] == ["DL123456"]
    assert extract_slots("dl-987654")["dl_codes"] == ["DL987654"]
    assert "dl_codes" not in extract_slots("DL 관련 문의입니다")


def test_every_scenario_routes_to_its_action() -> None:
    """분류 결과를 강제로 주입해 시나리오→액션→도구 경로를 전부 밟는다."""
    for scenario in SCENARIOS:
        state = GRAPH.invoke(
            {"inquiry": "테스트 문의", "scenario": scenario.id, "action": scenario.action.value}
        )
        assert state["response"], f"시나리오 {scenario.id} 응답 없음"

        tools_used = {c["name"] for c in state.get("tool_calls") or []}
        action = BY_ID[state["scenario"]].action
        if action is ActionType.DRAFT_APPROVAL:
            assert "open_draft_form" in tools_used
        elif action is ActionType.FORWARD_MAIL:
            assert "send_mail" in tools_used


def test_dl_api_tool_fires_only_when_code_present() -> None:
    with_code = GRAPH.invoke({"inquiry": "DL345678 로 사외 메일이 수신되지 않습니다."})
    without = GRAPH.invoke({"inquiry": "사외에서 보낸 메일이 그룹으로 안 들어옵니다."})

    assert with_code["scenario"] == 4
    assert "dl_api_get" in {c["name"] for c in with_code["tool_calls"]}
    assert without["scenario"] == 4
    assert "dl_api_get" not in {c["name"] for c in (without.get("tool_calls") or [])}


def test_forward_targets() -> None:
    admin = GRAPH.invoke({"inquiry": "전체 메일링그룹 조회 메뉴에 접근 권한이 없다고 나옵니다."})
    cs = GRAPH.invoke({"inquiry": "보낸 메일이 도착하지 않았다고 합니다."})

    assert admin["tool_calls"][0]["result"]["to"] == "DL_SystemManager@example.com"
    assert cs["tool_calls"][0]["result"]["to"] == "_cs@example.com"


def test_responses_never_invent_links() -> None:
    """응답에 등장하는 URL은 전부 schema.py에 등록된 것이어야 한다."""
    import re

    from dl_agent.schema import DRAFT_FORM_URL, WIKI_REST_API, WIKI_SENDER_GUIDE

    allowed = {DRAFT_FORM_URL, WIKI_REST_API, WIKI_SENDER_GUIDE}
    prefixes = tuple(u.split("?")[0] for u in allowed) + (
        "http://dldev.example.com/api/",
        "https://dl.example.com/api/",
    )

    for scenario in SCENARIOS:
        state = GRAPH.invoke(
            {"inquiry": "테스트", "scenario": scenario.id, "action": scenario.action.value}
        )
        for url in re.findall(r"https?://[^\s)]+", state["response"]):
            assert url.startswith(prefixes), f"시나리오 {scenario.id}: 알 수 없는 URL {url}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print(f"\n{'실패 ' + str(failures) + '건' if failures else '전부 통과'}")
    sys.exit(1 if failures else 0)
