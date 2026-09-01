"""문의에서 구조화 정보(슬롯)를 규칙 기반으로 뽑아낸다.

DL 코드나 메일주소처럼 **결정 가능한** 값은 LLM에 맡기지 않는다. 4B 모델의
토큰 단위 실수(자릿수 누락, 도메인 오타)를 원천 차단하고, 경량 LLM은 오직
분류에만 쓰기 위한 설계다.
"""

from __future__ import annotations

import re

from .schema import Slots

_DL_CODE = re.compile(r"\bDL\s?-?\s?(\d{4,8})\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SUBJECT = re.compile(
    r"(?:제목|메일\s*제목|title)\s*[:：]?\s*[\"'“”「『]?([^\"'“”」』\n]{2,60})"
)
_QUOTED = re.compile(r"[\"'“「『]([^\"'”」』\n]{2,60})[\"'”」』]")
_SENT_AT = re.compile(
    r"(\d{4}[-./]\d{1,2}[-./]\d{1,2}(?:\s*\d{1,2}[:시]\d{0,2}분?)?"
    r"|\d{1,2}[-./월]\s?\d{1,2}일?(?:\s*(?:오전|오후)?\s*\d{1,2}[:시]\d{0,2}분?)?"
    r"|(?:어제|오늘|그저께|엊그제)\s*(?:오전|오후)?\s*\d{0,2}\s*시?)"
)

_DL_HINT = re.compile(r"(dl|메일링\s?그룹|메일링그룹|그룹\s?메일)", re.IGNORECASE)


def extract_slots(text: str) -> Slots:
    slots: Slots = {}

    codes = [f"DL{m.group(1)}" for m in _DL_CODE.finditer(text)]
    if codes:
        slots["dl_codes"] = list(dict.fromkeys(codes))

    emails = list(dict.fromkeys(_EMAIL.findall(text)))
    if emails:
        slots["emails"] = emails
        dl_addrs = [e for e in emails if _DL_HINT.search(e.split("@")[0])]
        if dl_addrs:
            slots["dl_addresses"] = dl_addrs

    sent = _SENT_AT.search(text)
    if sent:
        slots["sent_at"] = sent.group(1).strip()

    subject = _SUBJECT.search(text)
    if subject:
        slots["subject"] = subject.group(1).strip()
    else:
        quoted = _QUOTED.search(text)
        if quoted and "@" not in quoted.group(1):
            slots["subject"] = quoted.group(1).strip()

    return slots


def missing_for_mail_triage(slots: Slots) -> list[str]:
    """메일 미수신/발송 실패 조사를 위해 담당자가 반드시 받아야 하는 항목."""
    need = {
        "발송자 메일주소": bool(slots.get("emails")),
        "수신자 메일주소": len(slots.get("emails") or []) >= 2,
        "발송 시간": bool(slots.get("sent_at")),
        "메일 제목": bool(slots.get("subject")),
    }
    return [k for k, present in need.items() if not present]
