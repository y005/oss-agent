"""비교용 베이스라인 분류기.

LLM을 쓴 게 실제로 이득인지 보이려면 "LLM 없이 얼마나 되는가"가 있어야 한다.
아래 규칙 기반 분류기는 train 스플릿만 보고 손으로 작성했다(val/test 미열람).
우선순위가 높은 규칙부터 먼저 매칭한다.
"""

from __future__ import annotations

import re
import time

RULES: list[tuple[int, str]] = [
    # 9. API / ACL — 고유 어휘가 많아 가장 먼저
    (9, r"(open\s?api|api|acl|egress|dldev|rest|연동\s?절차|호출\s?서버|엔드포인트)"),
    # 6. 전체 조회 메뉴 권한
    (6, r"(전체\s?(메일링\s?)?그룹.{0,6}조회|전체\s?조회|조회\s?메뉴).{0,20}(권한|접근|안\s?열|오류|없)"),
    (6, r"(권한|접근).{0,15}(전체\s?조회|조회\s?메뉴)"),
    # 8. 동적 DL
    (8, r"(동적\s?(dl|메일링))|(자동으로.{0,15}(들어가|포함|빠지|갱신|구성))|(조건.{0,10}(기반|으로).{0,15}(자동|그룹|멤버))|(인사\s?데이터.{0,10}연동)"),
    # 3. 만료 / 삭제 복구
    (3, r"(복구|복원|되살|되돌|삭제\s?취소)|((삭제|만료|사라).{0,20}(그룹|dl|복))"),
    # 5. 발신자 지정
    (5, r"(발신자|보내는\s?사람|보내는\s?메일|from\s?주소|명의로.{0,8}발송|발송\s?권한|대리\s?발송)"),
    # 4. 외부 메일 DL 수신
    (4, r"(사외|외부).{0,25}(수신|메일|반송|안\s?들어|못\s?받|안\s?와|안\s?옵|않)"),
    (4, r"(거래처|협력사|고객사|지원자|벤더).{0,30}(dl|그룹|메일링).{0,20}(안|못|반송)"),
    # 1. 이름 변경
    (1, r"(이름|명칭|표시명|타이틀|그룹명).{0,15}(변경|바꾸|수정|고치|정정|오타)"),
    (1, r"(변경|바꾸|수정).{0,10}(이름|명칭|그룹명)"),
    # 2. 조직 DL 편집
    (2, r"(조직|부서|본부).{0,10}(dl|메일링\s?그룹|그룹)"),
    (2, r"(멤버|인원|직원|파견|계약직|겸직|휴직).{0,20}(추가|제외|삭제|편집|수정|빼)"),
    # 7. 일반 송수신
    (7, r"(수신|발송|전송|반송|지연|도착).{0,20}(안|않|못|실패|지연|오류|없)"),
    (7, r"(메일).{0,15}(안\s?와|안\s?옵|안\s?들어|못\s?받|않|실패|반송|지연|사라)"),
]

COMPILED = [(scenario, re.compile(pattern, re.IGNORECASE)) for scenario, pattern in RULES]

# DL 업무와 무관해 보이는 문의를 10으로 밀어내는 거부 신호
OFF_TOPIC = re.compile(
    r"(vpn|노트북|연차|사원증|회의실|슬랙|메신저|캘린더|위키\s?계정|결재선|드라이브\s?용량)",
    re.IGNORECASE,
)
DL_HINT = re.compile(r"(dl|메일링\s?그룹|그룹\s?메일|메일)", re.IGNORECASE)


def keyword_classify(text: str) -> tuple[int, float]:
    """(시나리오, 신뢰도) 반환. 규칙 기반이라 신뢰도는 0/1에 가깝다."""
    if OFF_TOPIC.search(text):
        return 10, 0.9
    for scenario, pattern in COMPILED:
        if pattern.search(text):
            return scenario, 0.7
    if not DL_HINT.search(text):
        return 10, 0.6
    return 10, 0.3


class KeywordClient:
    """LlamaClient와 같은 인터페이스를 흉내내어 동일한 그래프에 꽂는다.

    베이스라인도 에이전트 전체(도구 호출·응답 조립)를 그대로 통과시켜야
    End-to-End 지표를 같은 조건에서 비교할 수 있다.
    """

    base_url = "keyword-baseline"

    def classify_label(self, messages: list[dict[str, str]], temperature: float | None = None):
        from dl_agent.llm import LabelPrediction
        from dl_agent.schema import BY_ID

        text = messages[-1]["content"].replace("/no_think", "").strip()
        started = time.perf_counter()
        scenario, confidence = keyword_classify(text)
        label = BY_ID[scenario].label
        return LabelPrediction(
            label=label,
            confidence=confidence,
            probs={label: confidence},
            raw=label,
            latency_ms=(time.perf_counter() - started) * 1000,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

    def healthy(self) -> bool:
        return True

    def close(self) -> None:
        return None
