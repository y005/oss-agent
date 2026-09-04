"""시나리오 정의와 에이전트 상태 스키마.

10개 시나리오, 각 시나리오의 처리 방식(액션), 그리고 분류기가 사용하는
단일 토큰 레이블(A~J)의 매핑을 한 곳에서 관리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict


class ActionType(str, Enum):
    """시나리오별 처리 방식."""

    FIXED_RESPONSE = "fixed_response"  # 고정 응답 안내
    DRAFT_APPROVAL = "draft_approval"  # 기안(결재) 페이지 호출
    FORWARD_MAIL = "forward_mail"  # 관리자 /  CS 메일 전달


@dataclass(frozen=True)
class Scenario:
    id: int
    label: str  # 단일 토큰 레이블 (A~J) — 분류기 출력 어휘
    name: str
    action: ActionType
    summary: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        1,
        "A",
        "DL 이름 편집/변경 요청",
        ActionType.DRAFT_APPROVAL,
        "메일링그룹(DL)의 이름/표시명을 바꾸거나 오타를 고쳐 달라는 요청",
    ),
    Scenario(
        2,
        "B",
        "조직 DL 편집 문의",
        ActionType.FIXED_RESPONSE,
        "인사정보로 자동 생성되는 조직/부서 DL의 멤버를 직접 추가·제외하고 싶다는 문의",
    ),
    Scenario(
        3,
        "C",
        "만료/삭제된 DL 복구 요청",
        ActionType.FIXED_RESPONSE,
        "실수로 삭제했거나 만료되어 사라진 메일링그룹을 되살리고 싶다는 요청",
    ),
    Scenario(
        4,
        "D",
        "외부(사외) 메일이 DL로 수신되지 않는 이슈",
        ActionType.FIXED_RESPONSE,
        "사외/외부 발신자가 보낸 메일이 DL 주소로 들어오지 않거나 반송되는 문제",
    ),
    Scenario(
        5,
        "E",
        "DL 주소를 발신자로 지정하는 방법",
        ActionType.FIXED_RESPONSE,
        "DL 메일주소를 '보내는 사람'으로 써서 메일을 발송하고 싶다는 문의",
    ),
    Scenario(
        6,
        "F",
        "'DL 전체 메일링그룹 조회' 메뉴 접근 권한 문의",
        ActionType.FORWARD_MAIL,
        "전체 메일링그룹 조회 메뉴에 접근 권한이 없다고 나오는 문의",
    ),
    Scenario(
        7,
        "G",
        "일반 메일 송수신 문제 (DL 설정과 무관)",
        ActionType.FORWARD_MAIL,
        "DL 설정과 무관한 사내 메일 미수신·발송 실패·지연·반송 문제",
    ),
    Scenario(
        8,
        "H",
        "동적 DL 문의",
        ActionType.DRAFT_APPROVAL,
        "법인/직급/재직형태 등 조건으로 멤버가 자동 관리되는 동적 DL 생성·수정 문의",
    ),
    Scenario(
        9,
        "I",
        "DL API 사용 문의 (open API 연동, ACL 신청)",
        ActionType.DRAFT_APPROVAL,
        "DL open API 연동 절차, API 스펙, 운영 ACL 신청 방법 문의",
    ),
    Scenario(
        10,
        "J",
        "분류 불가 / 기타",
        ActionType.FORWARD_MAIL,
        "DL과 무관하거나 위 어디에도 해당하지 않아 담당자 확인이 필요한 문의",
    ),
)

BY_ID: dict[int, Scenario] = {s.id: s for s in SCENARIOS}
BY_LABEL: dict[str, Scenario] = {s.label: s for s in SCENARIOS}
LABELS: tuple[str, ...] = tuple(s.label for s in SCENARIOS)

FALLBACK_SCENARIO_ID = 10


# --- 담당 채널 -------------------------------------------------------------

ADMIN_MAIL = "DL_SystemManager@example.com"
_CS_MAIL = "_cs@example.com"

DRAFT_FORM_URL = (
    "https://apms.example.com/aprvDoc/draft/E016"
    "?req_service=29&req_system=79&req_type=0"
)
WIKI_SENDER_GUIDE = (
    "https://wiki.example.com/spaces/Global/pages/361216242/"
    "DL%EB%AA%85%EC%9C%BC%EB%A1%9C+%EB%A9%94%EC%9D%BC+%EB%B0%9C%EC%86%A1"
    "+%EA%B6%8C%ED%95%9C+%EC%B6%94%EA%B0%80+%EB%B0%A9%EB%B2%95"
)
WIKI_REST_API = (
    "https://wiki.example.com/spaces/oapi/pages/552115497/"
    "%EB%A9%94%EC%9D%BC%EB%A7%81%EA%B7%B8%EB%A3%B9+REST+API"
)


# --- 그래프 상태 -----------------------------------------------------------


class Slots(TypedDict, total=False):
    """문의에서 규칙 기반으로 뽑아낸 구조화 정보."""

    dl_codes: list[str]
    dl_addresses: list[str]
    emails: list[str]
    sent_at: str | None
    subject: str | None


class ToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class Classification(TypedDict, total=False):
    scenario: int
    label: str
    confidence: float
    probs: dict[str, float]
    abstained: bool
    raw: str
    latency_ms: float
    backend: str


def _keep_last(_old: Any, new: Any) -> Any:
    return new


class AgentState(TypedDict, total=False):
    """LangGraph 상태.

    노드들은 부분 딕셔너리를 반환하고 LangGraph가 병합한다.
    `trace`만 append 리듀서를 쓴다.
    """

    inquiry: str
    slots: Slots
    classification: Classification
    scenario: int
    action: str
    tool_calls: Annotated[list[ToolCall], lambda old, new: (old or []) + (new or [])]
    response: str
    trace: Annotated[list[str], lambda old, new: (old or []) + (new or [])]


Backend = Literal["llama-server", "ollama", "keyword"]
