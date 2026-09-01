"""pool.jsonl → train / val / test 스플릿.

시나리오(10) × 난이도(easy/medium/hard) 이중 층화 추출이다. 시나리오만
맞추고 난이도를 섞지 않으면 test에 쉬운 문항만 몰려 성능이 부풀려진다.

  scenario당 20문항 = easy 8 / medium 7 / hard 5
    train 6 = easy 3 / medium 2 / hard 1   (few-shot 예시 풀 + 프롬프트 개발)
    val   4 = easy 2 / medium 1 / hard 1   (임계값·프롬프트 선택)
    test 10 = easy 3 / medium 4 / hard 3   (최종 1회 측정, 그 외 열람 금지)

시드가 고정되어 있어 몇 번을 돌려도 같은 스플릿이 나온다.

    python benchmark/split.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SEED = 20250902

QUOTA = {
    "train": {"easy": 3, "medium": 2, "hard": 1},
    "val": {"easy": 2, "medium": 1, "hard": 1},
    "test": {"easy": 3, "medium": 4, "hard": 3},
}


def main() -> None:
    pool = [json.loads(line) for line in (DATA / "pool.jsonl").read_text("utf-8").splitlines() if line.strip()]

    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in pool:
        buckets[(row["scenario"], row["difficulty"])].append(row)

    rng = random.Random(SEED)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda r: r["id"])
        rng.shuffle(rows)
        cursor = 0
        for split in ("train", "val", "test"):
            n = QUOTA[split][key[1]]
            take = rows[cursor : cursor + n]
            if len(take) < n:
                raise SystemExit(f"{key} 문항 부족: {len(rows)}개, {sum(QUOTA[s][key[1]] for s in QUOTA)}개 필요")
            splits[split].extend(take)
            cursor += n
        if cursor != len(rows):
            raise SystemExit(f"{key}: 사용되지 않은 문항 {len(rows) - cursor}개")

    for split, rows in splits.items():
        rows.sort(key=lambda r: r["id"])
        path = DATA / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        dist = defaultdict(int)
        for r in rows:
            dist[r["difficulty"]] += 1
        print(f"{split:5s} {len(rows):3d}문항  {dict(sorted(dist.items()))}  → {path.name}")

    ids = [r["id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids)) == len(pool), "스플릿 간 중복/누락"
    print(f"\n총 {len(pool)}문항, 스플릿 간 중복 없음 확인")


if __name__ == "__main__":
    main()
