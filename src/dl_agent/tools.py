"""스텁(모의) 도구 모음.

모든 도구는 **부작용이 없다.** 실제 결재 페이지를 열거나 메일을 보내지 않고,
호출 사실과 인자를 감사 로그(``results/tool_audit.log``)에 남긴 뒤 실제 API가
돌려줄 법한 모양의 응답을 만들어 반환한다.

평가에서 "어떤 도구를 어떤 인자로 불렀는가"가 그대로 채점 대상이 되므로
반환 스키마를 고정한다.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

from .config import settings
from .schema import (
    ADMIN_MAIL,
    DRAFT_FORM_URL,
    WIKI_REST_API,
    WIKI_SENDER_GUIDE,
    WORKS_CS_MAIL,
)

KST = timezone(timedelta(hours=9))


def _audit(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    settings.audit_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(KST).isoformat(timespec="seconds"),
        "tool": name,
        "args": args,
        "result": result,
    }
    with settings.audit_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ticket_id(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8].upper()
    return f"REQ-{digest}"


# --- 1. 기안(결재) 페이지 호출 ---------------------------------------------

DRAFT_FIELDS: dict[str, list[str]] = {
    "dl_rename": ["현재 그룹 이름", "변경할 그룹 이름", "DL 메일주소", "변경 사유"],
    "dynamic_dl": [
        "그룹 이름",
        "메일주소",
        "소속 법인",
        "상세 연동 조건(법인/정규직 여부/겸직 여부/직급 등)",
    ],
    "dl_api_acl": [
        "사용 서비스 명",
        "사용 서비스 도메인",
        "API 사용 목적",
        "호출 서버 IP (N3R Egress IP 포함)",
        "사용하려는 API 명",
        "대략적인 호출 빈도",
    ],
}


def open_draft_form(request_type: str, prefill: dict[str, Any] | None = None) -> dict[str, Any]:
    """[사내정보시스템 업무의뢰 > 메일링그룹(DL)] 기안 서식을 연다(모의)."""
    prefill = prefill or {}
    url = DRAFT_FORM_URL
    if prefill:
        url = f"{DRAFT_FORM_URL}&{urlencode({k: str(v) for k, v in prefill.items()})}"
    result = {
        "ok": True,
        "request_type": request_type,
        "draft_url": url,
        "required_fields": DRAFT_FIELDS.get(request_type, []),
        "ticket_id": _ticket_id(request_type + json.dumps(prefill, sort_keys=True)),
        "stub": True,
    }
    _audit("open_draft_form", {"request_type": request_type, "prefill": prefill}, result)
    return result


# --- 2. 메일 전달 -----------------------------------------------------------


def send_mail(to: str, subject: str, body: str) -> dict[str, Any]:
    """담당 채널로 문의를 전달한다(모의)."""
    result = {
        "ok": True,
        "to": to,
        "subject": subject,
        "message_id": f"<{_ticket_id(to + subject)}@dl-agent.local>",
        "queued_at": datetime.now(KST).isoformat(timespec="seconds"),
        "stub": True,
    }
    _audit("send_mail", {"to": to, "subject": subject, "body": body}, result)
    return result


def forward_to_admin(inquiry: str, scenario_name: str) -> dict[str, Any]:
    return send_mail(
        to=ADMIN_MAIL,
        subject=f"[DL 문의 전달] {scenario_name}",
        body=inquiry,
    )


def forward_to_works_cs(inquiry: str, missing: list[str] | None = None) -> dict[str, Any]:
    body = inquiry
    if missing:
        body += "\n\n[미확보 정보] " + ", ".join(missing)
    return send_mail(to=WORKS_CS_MAIL, subject="[메일 송수신 문의 전달]", body=body)


# --- 3. DL REST API 조회 -----------------------------------------------------

_EXTERNAL_RECV_BY_CODE: dict[str, str] = {}


def dl_api_get(dl_code: str) -> dict[str, Any]:
    """``GET /api/dl`` 모의 호출 — 외부메일 수신 설정 등을 조회한다.

    코드값을 시드로 쓰기 때문에 같은 DL 코드는 항상 같은 응답을 준다
    (평가 재현성).
    """
    code = dl_code.upper()
    if code not in _EXTERNAL_RECV_BY_CODE:
        rng = random.Random(int(hashlib.sha1(code.encode()).hexdigest()[:8], 16))
        _EXTERNAL_RECV_BY_CODE[code] = rng.choice(["ALLOW", "ALLOW", "DENY"])
    external = _EXTERNAL_RECV_BY_CODE[code]

    result = {
        "ok": True,
        "endpoint": "https://dl.navercorp.com/api/dl",
        "dl_code": code,
        "mail_address": f"{code.lower()}@navercorp.com",
        "external_mail_receive": external,  # ALLOW | DENY
        "member_count": 12 + (int(code[2:]) % 40 if code[2:].isdigit() else 7),
        "status": "ACTIVE",
        "stub": True,
    }
    _audit("dl_api_get", {"dl_code": code}, result)
    return result


# --- 4. 위키 가이드 링크 ------------------------------------------------------

WIKI_LINKS = {
    "sender_permission": ("DL 명의로 메일 발송 권한 추가 방법", WIKI_SENDER_GUIDE),
    "rest_api": ("메일링그룹 REST API 스펙 / ACL 신청", WIKI_REST_API),
}


def get_wiki_link(topic: str) -> dict[str, Any]:
    title, url = WIKI_LINKS.get(topic, ("메일링그룹 위키", WIKI_REST_API))
    result = {"ok": True, "topic": topic, "title": title, "url": url, "stub": True}
    _audit("get_wiki_link", {"topic": topic}, result)
    return result


TOOL_REGISTRY = {
    "open_draft_form": open_draft_form,
    "send_mail": send_mail,
    "dl_api_get": dl_api_get,
    "get_wiki_link": get_wiki_link,
}
