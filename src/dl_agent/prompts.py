"""분류 프롬프트 구성.

프롬프트를 v0/v1/v2로 버전화해 각 구성요소의 기여를 따로 잴 수 있게 했다.

  v0  레이블 정의만
  v1  v0 + [구분 기준] (헷갈리는 쌍에 대한 판별 규칙)
  v2  v1 + [기타(J) 사용 제한] — val 오답 분석에서 나온 처방

few-shot 예시는 **train 스플릿에서만** 뽑는다. val/test 문항이 프롬프트에
새어 들어가면 평가가 오염되므로 로더가 스플릿 이름을 고정한다.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from .config import settings
from .schema import SCENARIOS

_LABELS = """당신은 사내 메일링그룹(DL) 문의를 분류하는 라우터입니다.
사용자 문의를 읽고 아래 10개 중 **가장 알맞은 하나의 레이블 문자**만 출력하세요.
설명, 문장부호, 공백 없이 대문자 한 글자만 출력합니다.

[레이블]
A. DL 이름 편집/변경 — 메일링그룹의 이름·표시명 변경, 이름 오타 수정
B. 조직 DL 편집 — 인사정보로 자동 생성되는 조직/부서 DL의 멤버를 직접 추가·제외하고 싶음
C. 만료/삭제 DL 복구 — 실수로 지웠거나 만료된 메일링그룹을 되살리고 싶음
D. 외부 메일이 DL로 수신 안 됨 — 사외/외부 발신자가 보낸 메일이 DL 주소로 안 들어오거나 반송됨
E. DL 주소를 발신자로 지정 — DL·팀·그룹 메일주소를 '보내는 사람(From)'으로 써서 발송하고 싶음
F. 전체 메일링그룹 조회 메뉴 권한 없음 — 조회 메뉴 진입 시 접근 권한 없음 안내가 뜸
G. 일반 메일 송수신 문제 — DL과 무관한 사내 메일 미수신·발송 실패·지연·반송
H. 동적 DL — 법인/직급/재직형태 등 조건으로 멤버가 자동 관리되는 DL 생성·수정
I. DL API 사용 — open API 연동 절차, API 스펙, 운영 ACL 신청
J. 분류 불가/기타 — DL·메일 업무와 관계없거나 내용이 없어 판단이 불가능한 문의"""

_RULES = """
[구분 기준]
- D vs G: 수신이 안 되는 대상이 **DL/그룹 주소**이면 D, 개인 메일주소이거나 그룹 언급이 없으면 G.
- B vs A: 멤버(사람)를 넣고 빼는 문제면 B, 그룹의 **이름**을 바꾸는 문제면 A.
- B vs H: 이미 있는 조직 DL을 손보고 싶으면 B, 조건 기반으로 **자동 구성되는 DL을 새로 만들거나 조건을 바꾸고** 싶으면 H.
- E vs F: 메일을 그룹 명의로 **보내는** 권한이면 E, 조회 **메뉴** 접근 권한이면 F.
- I vs F: API/ACL/연동/서버IP/운영반영 이야기가 나오면 I.
- C: '삭제', '만료', '사라짐', '복구' 가 핵심이면 C."""

_ABSTAIN_GUARD = """
[기타(J) 사용 제한]
J는 최후의 수단입니다. 아래에 해당할 때만 J를 고르세요.
- 메일·메일링그룹과 전혀 무관한 문의(VPN, 노트북, 연차, 사원증, 회의실, 메신저 등)
- 내용이 없어 무엇을 원하는지 알 수 없는 문의

반대로 **메일링그룹·DL·그룹 메일주소가 언급되면 J를 쓰지 마세요.** 확실하지 않더라도
A~I 중 의미가 가장 가까운 것을 고릅니다. 예를 들어
- 그룹 멤버를 직접 못 고친다 → B (관리자 권한 이야기가 나와도 B)
- 팀/그룹 이름으로 메일이 나가게 하고 싶다 → E ('From', '보내는 사람', '내 이름으로만 나간다' 포함)
- DL과 무관하다고 명시했고 개인 메일 송수신 문제면 → G"""


PROMPTS: dict[str, str] = {
    "v0": _LABELS,
    "v1": _LABELS + "\n" + _RULES,
    "v2": _LABELS + "\n" + _RULES + "\n" + _ABSTAIN_GUARD,
}

DEFAULT_PROMPT_VERSION = "v2"

# 하위 호환 (기존 import 경로)
SYSTEM_PROMPT = PROMPTS[DEFAULT_PROMPT_VERSION]


def _load_split(split: str) -> list[dict]:
    path: Path = settings.data_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=8)
def few_shot_examples(k: int) -> tuple[tuple[str, str], ...]:
    """train 스플릿에서 시나리오별로 고르게 k개를 뽑는다."""
    if k <= 0:
        return ()
    pool = _load_split("train")
    if not pool:
        return ()

    by_scenario: dict[int, list[dict]] = {}
    for row in pool:
        by_scenario.setdefault(row["scenario"], []).append(row)

    rng = random.Random(20250902)
    picked: list[dict] = []
    per = max(1, k // len(SCENARIOS))
    for scenario in sorted(by_scenario):
        rows = sorted(by_scenario[scenario], key=lambda r: r["id"])
        rng.shuffle(rows)
        picked.extend(rows[:per])
    rng.shuffle(picked)

    label_of = {s.id: s.label for s in SCENARIOS}
    return tuple((row["text"], label_of[row["scenario"]]) for row in picked[:k])


def build_messages(
    inquiry: str,
    k: int | None = None,
    style: str = "inline",
    version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """분류용 메시지를 만든다.

    style="chat"   예시를 user/assistant 턴 쌍으로 넣는다.
    style="inline" 예시를 시스템 프롬프트 안 목록으로 넣는다.

    4B 모델은 긴 가짜 대화 이력을 실제 문맥으로 오인해 마지막 문의에 집중하지
    못하는 경향이 있어 inline이 기본값이다(docs/evaluation.md 실험 참고).
    """
    k = settings.few_shot_k if k is None else k
    examples = few_shot_examples(k)

    system = PROMPTS[version]
    messages: list[dict[str, str]] = []

    if examples and style == "inline":
        block = "\n".join(f'- "{text}" → {label}' for text, label in examples)
        system = f"{system}\n\n[예시]\n{block}"

    messages.append({"role": "system", "content": system})

    if examples and style == "chat":
        for text, label in examples:
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": label})

    messages.append({"role": "user", "content": f"{inquiry.strip()} /no_think"})
    return messages
