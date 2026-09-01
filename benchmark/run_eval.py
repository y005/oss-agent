"""벤치마크 실행기.

에이전트 **전체 그래프**를 문항마다 한 번씩 돌리고, 분류·라우팅·도구 호출·
응답까지 한 번에 채점한다. 분류만 따로 재면 "분류는 맞았는데 엉뚱한 도구를
불렀다" 같은 실패가 안 보이기 때문이다.

사용 예
-------
    # 검증셋으로 few-shot 개수 고르기
    python benchmark/run_eval.py --split val --few-shot 0  --tag val-zeroshot
    python benchmark/run_eval.py --split val --few-shot 20 --tag val-fewshot20

    # 규칙 기반 베이스라인
    python benchmark/run_eval.py --split val --system keyword --tag val-keyword

    # 최종 테스트 (한 번만)
    python benchmark/run_eval.py --split test --few-shot 20 --threshold 0.5 --tag test-final
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import metrics
from baselines import KeywordClient

from dl_agent.config import settings
from dl_agent.graph import build_graph
from dl_agent.llm import LlamaClient
from dl_agent.schema import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def load_split(split: str) -> list[dict[str, Any]]:
    path = settings.data_dir / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} 가 없습니다. 먼저 `python benchmark/split.py` 를 실행하세요.")
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def make_client(system: str, base_url: str | None, model: str | None):
    if system == "keyword":
        return KeywordClient()
    if system == "ollama":
        # ollama는 GBNF 문법을 지원하지 않아 문법 강제를 끄고 텍스트 파싱으로 폴백한다.
        return LlamaClient(
            base_url=base_url or "http://127.0.0.1:11434/v1",
            model=model or "qwen2.5:7b",
            use_grammar=False,
        )
    return LlamaClient(base_url=base_url, model=model)


def flatten_args(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """채점용으로 도구 인자를 평평하게 편다."""
    flat: dict[str, Any] = {}
    for call in tool_calls:
        for key, value in call["args"].items():
            flat.setdefault(key, value)
    return flat


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_split(args.split)
    client = make_client(args.system, args.base_url, args.model)

    if hasattr(client, "healthy") and not client.healthy():
        raise SystemExit(
            f"LLM 백엔드에 연결할 수 없습니다: {getattr(client, 'base_url', '?')}\n"
            "  ./scripts/serve_llm.sh 로 llama-server 를 먼저 띄우세요."
        )

    # 임계값은 config가 아니라 실행 인자로 주입해 실험마다 독립적으로 만든다.
    object.__setattr__(settings, "abstain_threshold", args.threshold)
    graph = build_graph(
        client=client,
        few_shot_k=args.few_shot,
        few_shot_style=args.few_shot_style,
        prompt_version=args.prompt,
    )

    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        started = time.perf_counter()
        state = graph.invoke({"inquiry": row["text"]})
        total_ms = (time.perf_counter() - started) * 1000

        cls = state.get("classification") or {}
        tool_calls = state.get("tool_calls") or []
        slots = state.get("slots") or {}
        gold = row["scenario"]

        rec = {
            "id": row["id"],
            "difficulty": row["difficulty"],
            "text": row["text"],
            "gold": gold,
            "pred": state.get("scenario", 10),
            # 기권 적용 전의 원래 예측 — 기권의 기회비용을 재는 데 쓴다
            "raw_pred": next(
                (s_.id for s_ in SCENARIOS if s_.label == cls.get("label")), 10
            ),
            "confidence": cls.get("confidence", 0.0),
            "probs": cls.get("probs", {}),
            "abstained": cls.get("abstained", False),
            "latency_ms": cls.get("latency_ms", 0.0),
            "total_ms": total_ms,
            "expected_tools": sorted(metrics.expected_tools(gold, slots)),
            "actual_tools": sorted({c["name"] for c in tool_calls}),
            "expected_args": metrics.expected_args(gold),
            "actual_args": flatten_args(tool_calls),
            "response": state.get("response", ""),
            "trace": state.get("trace", []),
        }
        rec["e2e_ok"] = metrics.e2e_success(rec)
        records.append(rec)

        mark = "O" if rec["gold"] == rec["pred"] else "X"
        print(
            f"[{i:3d}/{len(rows)}] {mark} {row['id']} gold={gold:2d} pred={rec['pred']:2d} "
            f"conf={rec['confidence']:.2f} {rec['latency_ms']:6.0f}ms",
            flush=True,
        )

    gold = [r["gold"] for r in records]
    pred = [r["pred"] for r in records]

    report = {
        "meta": {
            "split": args.split,
            "n": len(records),
            "system": args.system,
            "model": getattr(client, "model", args.system),
            "backend": getattr(client, "base_url", args.system),
            "few_shot_k": args.few_shot,
            "few_shot_style": args.few_shot_style,
            "prompt_version": args.prompt,
            "abstain_threshold": args.threshold,
            "grammar_constrained": getattr(client, "use_grammar", False),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "classification": metrics.classification_report(gold, pred),
        "action": metrics.action_report(gold, pred),
        "tools": metrics.tool_report(records),
        "e2e_success_rate": sum(r["e2e_ok"] for r in records) / len(records),
        "parse_failure_rate": sum(1 for r in records if not r.get("probs") and r["confidence"] == 0.0)
        / len(records),
        "calibration_ece": metrics.ece(records),
        "abstention": metrics.abstention_report(records),
        "risk_coverage": metrics.risk_coverage(records, THRESHOLDS),
        "latency": metrics.latency_report(records),
        "by_difficulty": metrics.by_difficulty(records),
        "records": records,
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    meta = report["meta"]
    cls = report["classification"]
    print("\n" + "=" * 68)
    print(f" {meta['system']} / {meta['model']}  ·  {meta['split']} 셋 {meta['n']}문항")
    print(f" few-shot k={meta['few_shot_k']}  기권 임계값={meta['abstain_threshold']}")
    print("=" * 68)
    print(f"분류 정확도        {cls['accuracy']:.3f}")
    print(f"분류 macro-F1      {cls['macro_f1']:.3f}")
    print(f"액션 정확도(3-way) {report['action']['accuracy']:.3f}")
    print(f"도구 집합 정확도   {report['tools']['exact_set_match']:.3f}  (micro-F1 {report['tools']['micro_f1']:.3f})")
    print(f"도구 인자 정확도   {report['tools']['arg_accuracy']:.3f}  (검사 {report['tools']['arg_checked']}건)")
    print(f"End-to-End 성공률  {report['e2e_success_rate']:.3f}")
    print(f"출력 파싱 실패율   {report['parse_failure_rate']:.3f}")
    print(f"신뢰도 보정 ECE    {report['calibration_ece']:.3f}")
    ab = report["abstention"]
    print(
        f"기권율             {ab['abstain_rate']:.3f}  "
        f"(자동처리분 정확도 {ab['auto_accuracy']:.3f})"
    )
    print(
        f"위험 오류율        {ab['harmful_error_rate']:.3f}  "
        f"← 자동 처리했는데 틀린 비율 (운영상 가장 비싼 실패)"
    )
    lat = report["latency"]
    print(f"LLM 지연 p50/p95   {lat['llm_p50_ms']:.0f} / {lat['llm_p95_ms']:.0f} ms")
    print(f"E2E 지연 p50/p95   {lat['e2e_p50_ms']:.0f} / {lat['e2e_p95_ms']:.0f} ms")

    print("\n난이도별")
    for name, v in report["by_difficulty"].items():
        print(f"  {name:7s} n={v['n']:3d}  분류 {v['accuracy']:.3f}  E2E {v['e2e']:.3f}")

    print("\n시나리오별 (P / R / F1 / n)")
    from dl_agent.schema import BY_ID

    for cid, v in cls["per_class"].items():
        print(
            f"  {cid:2d} {BY_ID[cid].name[:22]:22s} "
            f"{v['precision']:.2f} / {v['recall']:.2f} / {v['f1']:.2f} / {v['support']:3d}"
        )

    print("\n혼동행렬 (행=정답, 열=예측)")
    header = "     " + "".join(f"{c:4d}" for c in metrics.SCENARIO_IDS)
    print(header)
    for g in metrics.SCENARIO_IDS:
        cells = "".join(
            f"{cls['confusion'][g][p]:4d}" if cls["confusion"][g][p] else "   ."
            for p in metrics.SCENARIO_IDS
        )
        print(f"  {g:2d} {cells}")

    print("\nrisk-coverage (임계값 / 자동처리 비율 / 자동처리분 정확도)")
    for row in report["risk_coverage"]:
        print(
            f"  τ={row['threshold']:.2f}  coverage={row['coverage']:.2f}  "
            f"acc={row['accuracy_on_covered']:.3f}  (n={row['n_covered']})"
        )

    wrong = [r for r in report["records"] if r["gold"] != r["pred"]]
    if wrong:
        print(f"\n오분류 {len(wrong)}건")
        for r in wrong[:20]:
            print(f"  {r['id']} [{r['difficulty']}] {r['gold']}→{r['pred']} conf={r['confidence']:.2f}  {r['text'][:44]}")


def main() -> None:
    p = argparse.ArgumentParser(description="DL 문의 에이전트 벤치마크")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--system", default="llama", choices=["llama", "ollama", "keyword"])
    p.add_argument("--few-shot", type=int, default=None, help="프롬프트에 넣을 예시 수 (기본: 설정값)")
    p.add_argument(
        "--few-shot-style",
        default="inline",
        choices=["inline", "chat"],
        help="예시를 시스템 프롬프트 안에 넣을지(inline), 대화 턴으로 넣을지(chat)",
    )
    p.add_argument(
        "--prompt",
        default="v2",
        choices=["v0", "v1", "v2"],
        help="프롬프트 버전 (v0 레이블만 / v1 +구분기준 / v2 +기타사용제한)",
    )
    p.add_argument("--threshold", type=float, default=0.0, help="기권 임계값")
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--tag", default=None, help="결과 파일 이름")
    args = p.parse_args()

    if args.few_shot is None:
        args.few_shot = settings.few_shot_k

    report = evaluate(args)
    print_report(report)

    tag = args.tag or f"{args.split}-{args.system}-k{args.few_shot}-t{args.threshold}"
    out = settings.results_dir / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
