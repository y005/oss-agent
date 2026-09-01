"""최종 결과 심층 분석 — 문서에 넣을 숫자를 뽑는다.

    python benchmark/analyze.py

포함 내용
  1. Wilson 95% 신뢰구간 — n이 작을 때 차이를 과대해석하지 않기 위해
  2. val→test 하락의 분해 (난이도 구성 차이 vs 그 외)
  3. 테스트셋 혼동행렬과 오분류 유형
  4. 기권(abstention)이 실제로 막은 오류
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(tag: str) -> dict:
    return json.loads((RESULTS / f"{tag}.json").read_text("utf-8"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 구간. 정규근사보다 작은 n에서 안정적이다."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def section(title: str) -> None:
    print(f"\n{'─' * 66}\n {title}\n{'─' * 66}")


def main() -> None:
    section("1. 테스트셋 정확도와 95% 신뢰구간 (Wilson)")
    tags = [
        ("Qwen3-4B v2 τ=0.7 (제안)", "test-final"),
        ("Qwen3-4B v2 τ=0.0", "test-final-tau00"),
        ("qwen2.5:7b v2 (참조)", "test-ollama-qwen25-7b"),
        ("규칙 베이스라인", "test-keyword"),
    ]
    for name, tag in tags:
        try:
            d = load(tag)
        except FileNotFoundError:
            continue
        n = d["meta"]["n"]
        acc = d["classification"]["accuracy"]
        lo, hi = wilson(round(acc * n), n)
        print(f"  {name:26s} {acc:.3f}  [{lo:.3f}, {hi:.3f}]  n={n}")
    print(
        "\n  → 구간이 겹치면 순위 차이를 단정할 수 없다. n=100에서 5%p 차이는\n"
        "    통계적으로 유의하지 않다."
    )

    section("2. val 0.900 → test 0.730 하락의 분해")
    val = load("val-v2-k0-inline")
    test = load("test-final-tau00")
    vd = val["by_difficulty"]
    td = test["by_difficulty"]

    print("  난이도별 정확도")
    print(f"    {'':8s}{'val':>14s}{'test':>14s}")
    for k in ("easy", "medium", "hard"):
        print(
            f"    {k:8s}{vd[k]['accuracy']:>8.3f} (n={vd[k]['n']:2d}){td[k]['accuracy']:>8.3f} (n={td[k]['n']:2d})"
        )

    # val 성능을 test의 난이도 구성으로 재가중
    total_test = sum(v["n"] for v in td.values())
    reweighted = sum(vd[k]["accuracy"] * td[k]["n"] for k in td) / total_test
    val_acc = val["classification"]["accuracy"]
    test_acc = test["classification"]["accuracy"]

    print(f"\n  val 실측                        {val_acc:.3f}")
    print(f"  val을 test 난이도 구성으로 재가중  {reweighted:.3f}   (구성 차이 기여 {val_acc - reweighted:+.3f})")
    print(f"  test 실측                       {test_acc:.3f}   (나머지 갭 {reweighted - test_acc:+.3f})")
    print(
        "\n  → 하락의 대부분은 난이도 구성이 아니라 '검증셋에 맞춘 프롬프트'와\n"
        "    val의 작은 표본(n=40)에서 온다."
    )

    section("3. 테스트셋 오분류 유형 (상위)")
    pairs = Counter(
        (r["gold"], r["pred"]) for r in test["records"] if r["gold"] != r["pred"]
    )
    from dl_agent.schema import BY_ID

    for (g, p), cnt in pairs.most_common(8):
        print(f"  {cnt:2d}건  {g:2d} {BY_ID[g].name[:20]:20s} → {p:2d} {BY_ID[p].name[:20]}")

    to_ten = sum(c for (g, p), c in pairs.items() if p == 10 and g != 10)
    from_ten = sum(c for (g, p), c in pairs.items() if g == 10 and p != 10)
    total_err = sum(pairs.values())
    print(f"\n  전체 오분류 {total_err}건 중")
    print(f"    실제 업무 문의를 10(기타)으로 흘림 : {to_ten}건 ({to_ten / total_err:.0%})")
    print(f"    무관한 문의를 업무 시나리오로 오인 : {from_ten}건 ({from_ten / total_err:.0%})")

    section("4. 기권(τ=0.7)이 바꾼 것 — 테스트셋")
    a0 = load("test-final-tau00")
    a7 = load("test-final")
    for name, d in (("τ=0.0 (기권 없음)", a0), ("τ=0.7 (제안)", a7)):
        ab = d["abstention"]
        print(
            f"  {name:18s} 분류acc {d['classification']['accuracy']:.3f}  "
            f"기권율 {ab['abstain_rate']:.3f}  "
            f"자동처리분 정확도 {ab['auto_accuracy']:.3f}  "
            f"위험오류율 {ab['harmful_error_rate']:.3f}"
        )
    print(
        "\n  → 기권은 분류 정확도를 크게 올리지 않는다. 대신 '틀린 안내를 자동으로\n"
        "    내보낸 비율'(위험 오류율)을 낮추고, 그만큼을 사람에게 넘긴다."
    )

    section("5. 시나리오별 F1 (테스트셋, τ=0.7)")
    per_class = load("test-final")["classification"]["per_class"]
    for cid_key, v in sorted(per_class.items(), key=lambda kv: int(kv[0])):
        cid = int(cid_key)  # JSON 왕복 후 키가 문자열이 된다
        bar = "█" * round(v["f1"] * 20)
        print(f"  {cid:2d} {BY_ID[cid].name[:22]:22s} F1={v['f1']:.2f} {bar}")


if __name__ == "__main__":
    main()
