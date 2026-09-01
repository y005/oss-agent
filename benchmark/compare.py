"""results/*.json 을 한 표로 모아 비교한다.

    python benchmark/compare.py          # 전부
    python benchmark/compare.py val      # 접두사 필터
    python benchmark/compare.py --md     # 문서에 붙일 마크다운 표
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

COLUMNS = [
    ("설정", 26),
    ("n", 4),
    ("분류acc", 8),
    ("macroF1", 8),
    ("액션acc", 8),
    ("도구set", 8),
    ("E2E", 7),
    ("ECE", 7),
    ("p50ms", 7),
]


def describe(meta: dict) -> str:
    if meta["system"] == "keyword":
        return "규칙 베이스라인"
    model = meta["model"]
    bits = [model, f"k={meta['few_shot_k']}"]
    if meta.get("few_shot_style") and meta["few_shot_k"]:
        bits.append(meta["few_shot_style"])
    if meta.get("prompt_version"):
        bits.append(meta["prompt_version"])
    if meta.get("abstain_threshold"):
        bits.append(f"τ={meta['abstain_threshold']}")
    return " ".join(bits)


def rows(prefix: str) -> list[tuple]:
    out = []
    for path in sorted(RESULTS.glob(f"{prefix}*.json")):
        d = json.loads(path.read_text("utf-8"))
        if "classification" not in d:
            continue
        m, c = d["meta"], d["classification"]
        out.append(
            (
                describe(m),
                m["n"],
                c["accuracy"],
                c["macro_f1"],
                d["action"]["accuracy"],
                d["tools"]["exact_set_match"],
                d["e2e_success_rate"],
                d["calibration_ece"],
                d["latency"]["llm_p50_ms"],
            )
        )
    return sorted(out, key=lambda r: -r[2])


def main() -> None:
    args = [a for a in sys.argv[1:]]
    as_md = "--md" in args
    prefix = next((a for a in args if not a.startswith("--")), "")

    data = rows(prefix)
    if not data:
        raise SystemExit(f"results/{prefix}*.json 결과가 없습니다.")

    if as_md:
        print("| " + " | ".join(name for name, _ in COLUMNS) + " |")
        print("|" + "|".join("---" for _ in COLUMNS) + "|")
        for r in data:
            print(
                f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | "
                f"{r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.0f} |"
            )
        return

    print("".join(f"{name:<{w}}" for name, w in COLUMNS))
    print("-" * sum(w for _, w in COLUMNS))
    for r in data:
        print(
            f"{r[0]:<26}{r[1]:<4}{r[2]:<8.3f}{r[3]:<8.3f}{r[4]:<8.3f}"
            f"{r[5]:<8.3f}{r[6]:<7.3f}{r[7]:<7.3f}{r[8]:<7.0f}"
        )


if __name__ == "__main__":
    main()
