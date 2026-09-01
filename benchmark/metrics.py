"""평가 지표 정의.

외부 의존성 없이 직접 구현했다. 각 지표가 무엇을 재는지 발표에서 설명할 수
있어야 하므로 계산식을 숨기지 않는다.

지표 계층
---------
1. 분류 정확도 / macro-F1 / 혼동행렬   — 라우터가 시나리오를 맞혔는가
2. 액션 정확도 (3-way)                 — 시나리오를 틀려도 처리 방식은 맞았는가
3. 도구 호출 정확도                    — 옳은 도구를, 옳은 인자로 불렀는가
4. End-to-End 성공률                   — 위 셋이 모두 성립하고 응답까지 정상인가
5. 신뢰도 보정(ECE) / risk-coverage    — "확신 없으면 사람에게" 가 작동하는가
6. 지연시간 p50/p95, 토큰              — 4B 로컬 모델의 실사용 가능성
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Sequence

from dl_agent.schema import BY_ID, SCENARIOS

SCENARIO_IDS = [s.id for s in SCENARIOS]


# --- 1. 분류 ---------------------------------------------------------------


def confusion_matrix(gold: Sequence[int], pred: Sequence[int]) -> dict[int, dict[int, int]]:
    m = {g: {p: 0 for p in SCENARIO_IDS} for g in SCENARIO_IDS}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def per_class_prf(gold: Sequence[int], pred: Sequence[int]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for c in SCENARIO_IDS:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return out


def classification_report(gold: Sequence[int], pred: Sequence[int]) -> dict[str, Any]:
    prf = per_class_prf(gold, pred)
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else 0.0
    return {
        "n": len(gold),
        "accuracy": accuracy,
        "macro_f1": mean(v["f1"] for v in prf.values()),
        "macro_precision": mean(v["precision"] for v in prf.values()),
        "macro_recall": mean(v["recall"] for v in prf.values()),
        "per_class": prf,
        "confusion": confusion_matrix(gold, pred),
    }


# --- 2. 액션 ---------------------------------------------------------------


def action_of(scenario_id: int) -> str:
    return BY_ID[scenario_id].action.value


def action_report(gold: Sequence[int], pred: Sequence[int]) -> dict[str, Any]:
    ga = [action_of(g) for g in gold]
    pa = [action_of(p) for p in pred]
    acc = sum(1 for g, p in zip(ga, pa) if g == p) / len(ga) if ga else 0.0
    by_action: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for g, p in zip(ga, pa):
        by_action[g]["total"] += 1
        by_action[g]["correct"] += int(g == p)
    return {"accuracy": acc, "by_action": dict(by_action)}


# --- 3. 도구 호출 -----------------------------------------------------------

DRAFT_TYPE = {1: "dl_rename", 8: "dynamic_dl", 9: "dl_api_acl"}
WIKI_TOPIC = {5: "sender_permission", 9: "rest_api"}
MAIL_TO = {
    6: "DL_SystemManager@navercorp.com",
    7: "works_cs@navercorp.com",
    10: "DL_SystemManager@navercorp.com",
}


def expected_tools(scenario_id: int, slots: dict[str, Any]) -> set[str]:
    """정답 시나리오와 문의에 실제로 들어있는 정보로부터 기대 도구 집합을 만든다."""
    expected: set[str] = set()
    if scenario_id in DRAFT_TYPE:
        expected.add("open_draft_form")
    if scenario_id in WIKI_TOPIC:
        expected.add("get_wiki_link")
    if scenario_id in MAIL_TO:
        expected.add("send_mail")
    if scenario_id == 4 and slots.get("dl_codes"):
        expected.add("dl_api_get")
    return expected


def expected_args(scenario_id: int) -> dict[str, str]:
    args: dict[str, str] = {}
    if scenario_id in DRAFT_TYPE:
        args["request_type"] = DRAFT_TYPE[scenario_id]
    if scenario_id in MAIL_TO:
        args["to"] = MAIL_TO[scenario_id]
    return args


def tool_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """도구 집합 exact-match 와 micro P/R/F1, 인자 정확도."""
    exact = 0
    tp = fp = fn = 0
    arg_ok = arg_total = 0
    n = 0
    for rec in records:
        n += 1
        want = set(rec["expected_tools"])
        got = set(rec["actual_tools"])
        exact += int(want == got)
        tp += len(want & got)
        fp += len(got - want)
        fn += len(want - got)

        for key, value in rec["expected_args"].items():
            arg_total += 1
            arg_ok += int(rec["actual_args"].get(key) == value)

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_set_match": exact / n if n else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "arg_accuracy": arg_ok / arg_total if arg_total else 1.0,
        "arg_checked": arg_total,
    }


# --- 4. End-to-End ----------------------------------------------------------


def e2e_success(rec: dict[str, Any]) -> bool:
    """분류·도구·인자·응답이 모두 성립해야 성공으로 센다."""
    if rec["gold"] != rec["pred"]:
        return False
    if set(rec["expected_tools"]) != set(rec["actual_tools"]):
        return False
    for key, value in rec["expected_args"].items():
        if rec["actual_args"].get(key) != value:
            return False
    return bool(rec.get("response"))


# --- 5. 신뢰도 --------------------------------------------------------------


def ece(records: Sequence[dict[str, Any]], bins: int = 10) -> float:
    """Expected Calibration Error — 모델이 말한 확신도가 실제 정답률과 맞는가."""
    if not records:
        return 0.0
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for rec in records:
        conf = float(rec.get("confidence") or 0.0)
        idx = min(bins - 1, int(conf * bins))
        buckets[idx].append((conf, rec["gold"] == rec["pred"]))
    total = len(records)
    err = 0.0
    for items in buckets.values():
        acc = mean(1.0 if ok else 0.0 for _, ok in items)
        avg_conf = mean(c for c, _ in items)
        err += (len(items) / total) * abs(acc - avg_conf)
    return err


def abstention_report(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """기권이 실제로 무엇을 막았는지 센다.

    분류 정확도만 보면 기권은 손해로 보인다(정답 5를 10으로 바꾸면 여전히 오답).
    하지만 운영에서 중요한 건 **틀린 안내를 자동으로 발송한 비율**이다. 기권한
    건은 관리자에게 넘어가 사람이 처리하므로 잘못된 자동 응답이 나가지 않는다.
    """
    n = len(records)
    if not n:
        return {}
    abstained = [r for r in records if r.get("abstained")]
    auto = [r for r in records if not r.get("abstained")]
    harmful = [r for r in auto if r["gold"] != r["pred"]]
    return {
        "abstain_rate": len(abstained) / n,
        "auto_accuracy": (sum(1 for r in auto if r["gold"] == r["pred"]) / len(auto)) if auto else 0.0,
        # 자동 처리했는데 틀린 비율 — 운영상 가장 비싼 실패
        "harmful_error_rate": len(harmful) / n,
        # 기권했지만 원래 예측이 정답이었던 비율 — 기권의 기회비용
        "wasted_abstention_rate": (
            sum(1 for r in abstained if r.get("raw_pred") == r["gold"]) / n
        ),
    }


def risk_coverage(
    records: Sequence[dict[str, Any]], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """임계값을 올리며 (자동 처리 비율, 자동 처리분의 정확도)를 그린다.

    기권한 건은 사람(관리자)에게 넘어가므로 '틀린 답을 자동 발송하는 위험'은
    사라진다. 어느 임계값에서 얼마나 커버리지를 포기하면 되는지를 보여준다.
    """
    out = []
    for t in thresholds:
        covered = [r for r in records if float(r.get("confidence") or 0.0) >= t]
        acc = (
            sum(1 for r in covered if r["gold"] == r["pred"]) / len(covered) if covered else 0.0
        )
        out.append(
            {
                "threshold": t,
                "coverage": len(covered) / len(records) if records else 0.0,
                "accuracy_on_covered": acc,
                "n_covered": len(covered),
            }
        )
    return out


# --- 6. 성능 ----------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def latency_report(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    lat = [float(r.get("latency_ms") or 0.0) for r in records]
    tot = [float(r.get("total_ms") or 0.0) for r in records]
    return {
        "llm_p50_ms": percentile(lat, 0.5),
        "llm_p95_ms": percentile(lat, 0.95),
        "llm_mean_ms": mean(lat) if lat else 0.0,
        "e2e_p50_ms": percentile(tot, 0.5),
        "e2e_p95_ms": percentile(tot, 0.95),
    }


def by_difficulty(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec.get("difficulty", "unknown")].append(rec)
    return {
        name: {
            "n": len(rows),
            "accuracy": sum(1 for r in rows if r["gold"] == r["pred"]) / len(rows),
            "e2e": sum(1 for r in rows if e2e_success(r)) / len(rows),
        }
        for name, rows in sorted(groups.items())
    }
