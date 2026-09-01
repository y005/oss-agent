"""시나리오별 최종 응답 조립.

응답 문장은 LLM이 생성하지 않는다. 가이드 문서에 적힌 내용을 그대로 옮긴
템플릿에 도구 호출 결과만 채워 넣는다. 4B 모델이 링크·메일주소·정책을
지어내는 사고(hallucination)를 구조적으로 막기 위한 선택이다.
"""

from __future__ import annotations

from typing import Any

from .extract import missing_for_mail_triage
from .schema import (
    ADMIN_MAIL,
    BY_ID,
    WORKS_CS_MAIL,
    Slots,
    ToolCall,
)

TRIAGE_FIELDS = "발송자 메일주소 / 수신자(DL) 메일주소 / 발송 시간 / 메일 제목"


def _tool(tool_calls: list[ToolCall], name: str) -> dict[str, Any] | None:
    for call in tool_calls:
        if call["name"] == name:
            return call["result"]
    return None


def _draft_block(result: dict[str, Any]) -> str:
    fields = "\n".join(f"  - {f}" for f in result.get("required_fields", []))
    body = f"아래 기안 서식으로 요청해 주시면 접수 후 처리해 드립니다.\n\n▸ 기안 서식: {result['draft_url']}"
    if fields:
        body += f"\n\n▸ 서식에 기재해 주실 항목\n{fields}"
    body += f"\n\n▸ 접수번호(가): {result['ticket_id']}"
    return body


def compose(
    scenario_id: int,
    inquiry: str,
    slots: Slots,
    tool_calls: list[ToolCall],
    confidence: float = 1.0,
    abstained: bool = False,
) -> str:
    scenario = BY_ID[scenario_id]
    head = f"[분류] {scenario_id}. {scenario.name} (확신도 {confidence:.0%})"
    body = _body(scenario_id, inquiry, slots, tool_calls, abstained)
    return f"{head}\n\n{body}"


def _body(
    scenario_id: int,
    inquiry: str,
    slots: Slots,
    tool_calls: list[ToolCall],
    abstained: bool,
) -> str:
    if scenario_id == 1:
        draft = _tool(tool_calls, "open_draft_form")
        return (
            "DL 이름 변경은 담당자가 임의로 수정하지 않고 기안(결재)으로 접수받아 처리합니다.\n\n"
            + _draft_block(draft)
            if draft
            else "DL 이름 변경은 기안(결재)으로 접수받아 처리합니다."
        )

    if scenario_id == 2:
        return (
            "조직(부서) DL은 **인사정보를 기준으로 자동 생성·관리**되는 그룹이라, "
            "관리자가 멤버를 직접 추가하거나 제외할 수 없습니다.\n\n"
            "파견직·계약직 제외 등 편집이 꼭 필요하시면 **소속 법인의 인사부서 담당자**에게 "
            "제외 요청을 해주셔야 합니다.\n\n"
            "⚠️ 조직 DL에서 제외되면 해당 DL과 연결된 WORKS 기능(드라이브 등) 접근이 "
            "불가해질 수 있어, 대상자에게 사전 고지한 뒤 진행됩니다."
        )

    if scenario_id == 3:
        return (
            "삭제·만료된 메일링그룹은 **직접 복구하실 수 있습니다.**\n\n"
            "▸ DL 관리 화면 > [만료된 메일링그룹] 메뉴에서 대상 그룹을 찾아 복구\n"
            "▸ 단, **삭제 후 1개월 이내**의 그룹만 조회·복구가 가능합니다.\n\n"
            "1개월이 지난 경우에는\n"
            "  - 메일링그룹만 사용하셨다면 → 새로 생성해 주시면 됩니다.\n"
            f"  - WORKS 드라이브 등 데이터 복구가 필요하다면 → {ADMIN_MAIL} 로 문의해 주세요."
        )

    if scenario_id == 4:
        api = _tool(tool_calls, "dl_api_get")
        lines = [
            "먼저 메일링그룹 상세 설정에서 **[외부메일 수신]** 값이 `허용`인지 확인해 주세요."
        ]
        if api:
            state = api["external_mail_receive"]
            korean = "허용" if state == "ALLOW" else "차단"
            lines.append(
                f"\n▸ 조회 결과 ({api['dl_code']} / {api['mail_address']}): "
                f"외부메일 수신 = **{korean}** ({state}), 상태 {api['status']}, 멤버 {api['member_count']}명"
            )
            if state == "DENY":
                lines.append(
                    "\n외부메일 수신이 **차단**되어 있어 사외에서 보낸 메일이 반송된 것으로 보입니다. "
                    "DL 관리 화면에서 [외부메일 수신]을 `허용`으로 변경해 주세요."
                )
                return "\n".join(lines)
        lines.append(
            "\n`허용` 상태인데도 수신되지 않는다면 발신제한 설정 가능성이 있어 관리자 확인이 필요합니다.\n"
            f"아래 정보를 담아 {ADMIN_MAIL} 로 문의해 주세요.\n"
            f"  - {TRIAGE_FIELDS.replace(' / ', chr(10) + '  - ')}"
        )
        missing = missing_for_mail_triage(slots)
        if missing:
            lines.append(f"\n(현재 문의에서 확인되지 않은 항목: {', '.join(missing)})")
        return "\n".join(lines)

    if scenario_id == 5:
        wiki = _tool(tool_calls, "get_wiki_link")
        url = wiki["url"] if wiki else ""
        return (
            "DL 주소를 발신자(보내는 사람)로 지정하려면 해당 DL에 대한 **발송 권한**을 "
            "먼저 부여받아야 합니다.\n\n"
            f"▸ 가이드: DL 명의로 메일 발송 권한 추가 방법\n  {url}\n\n"
            "위키 문서의 절차대로 권한을 추가하시면 메일 작성 화면의 보내는사람에서 "
            "DL 주소를 선택하실 수 있습니다."
        )

    if scenario_id == 6:
        mail = _tool(tool_calls, "send_mail")
        sent = f"\n\n▸ 전달 완료: {mail['to']} (메시지 ID {mail['message_id']})" if mail else ""
        return (
            "'DL 전체 메일링그룹 조회' 메뉴는 별도 권한이 필요한 메뉴입니다.\n"
            "문의 내용을 **DL 시스템 관리자**에게 전달했으며, 권한 처리 결과를 확인한 뒤 "
            "회신드리겠습니다." + sent
        )

    if scenario_id == 7:
        mail = _tool(tool_calls, "send_mail")
        missing = missing_for_mail_triage(slots)
        sent = f"\n\n▸ 전달 완료: {mail['to']} (메시지 ID {mail['message_id']})" if mail else ""
        ask = (
            f"\n\n빠른 확인을 위해 아래 정보를 회신해 주세요.\n  - {', '.join(missing)}"
            if missing
            else ""
        )
        return (
            "말씀하신 증상은 DL 설정과는 무관한 사내 메일 송수신 문제로 보입니다.\n"
            f"**WORKS 메일 담당(CS)** 으로 문의를 전달했습니다. ({WORKS_CS_MAIL})" + sent + ask
        )

    if scenario_id == 8:
        draft = _tool(tool_calls, "open_draft_form")
        intro = (
            "법인·직급·재직형태 등 조건으로 멤버가 자동 관리되는 **동적 DL**은 "
            "기안(결재)으로 접수받아 등록해 드립니다.\n\n"
        )
        return intro + (_draft_block(draft) if draft else "")

    if scenario_id == 9:
        wiki = _tool(tool_calls, "get_wiki_link")
        draft = _tool(tool_calls, "open_draft_form")
        parts = [
            "DL open API는 아래 위키에 스펙과 신청 방법이 정리되어 있습니다.",
            f"\n▸ 위키: {wiki['url'] if wiki else ''}",
            "\n▸ API URL\n  - 개발: http://dldev.navercorp.com/api/\n  - 운영: https://dl.navercorp.com/api/",
            "\n▸ 제공 기능: 메일링그룹 생성/수정/삭제, 정보 조회, 멤버 등록/조회/삭제, "
            "특정 사번 소속 DL 조회, 메일주소·이름 중복 확인, Groups 연동",
            "\n**운영 적용을 위한 ACL 등록**은 기안(결재)으로 신청해 주셔야 합니다.\n",
        ]
        if draft:
            parts.append(_draft_block(draft))
        return "\n".join(parts)

    # 10 — 분류 불가 / 기타
    mail = _tool(tool_calls, "send_mail")
    sent = f"\n\n▸ 전달 완료: {mail['to']} (메시지 ID {mail['message_id']})" if mail else ""
    reason = (
        "문의 내용만으로는 처리 유형을 확신하기 어려워"
        if abstained
        else "자동 분류 대상에 해당하지 않아"
    )
    return (
        f"{reason} **DL 시스템 관리자**에게 문의를 전달했습니다.\n"
        "담당자가 내용을 확인한 뒤 회신드리겠습니다." + sent
    )
