"""Remove all CMI-RewardBench Music Arena UUIDs from our Music Arena pool.

Paper anchor: Section 3 (Training data paragraph) + §A.D (post-cutoff probe).

CMI-RewardBench~\\cite{ma2026cmirewardbench} releases 1,340 ``battle_uuid``s
drawn from Music Arena as its test split. To make every TuneJury cell on
the CMI-RewardBench Music Arena split a fair out-of-distribution reading,
we remove all 1,340 from our Music Arena pool (train + val + held-out
test). All 1,340 match the raw pool; 131 land in the
internal Music Arena test split, which then shrinks from 205 to 74 pairs
(Section 3 Table footnote).

Usage
-----
$ python data/filter_music_arena_bench_clean.py \\
      --music-arena-root      /path/to/music_arena_pool \\
      --cmi-bench-uuid-list   /path/to/cmi_rewardbench/music_arena_uuids.txt \\
      --out-pool              data/music_arena_bench_clean.json \\
      --out-report            data/filter_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_uuid_list(path: Path) -> set[str]:
    with path.open() as fp:
        return {line.strip() for line in fp if line.strip()}


def _load_pool(music_arena_root: Path) -> list[dict]:
    """Loads the Music Arena pool as a list of battle records.

    Each record must have a ``battle_uuid`` field and a ``vote`` field
    (``A``, ``B``, ``TIE``, ``BOTH_BAD``). We retain only decisive
    (A or B) votes; TIE / BOTH_BAD are dropped before any filtering.
    """
    candidates = [
        music_arena_root / "battles.json",
        music_arena_root / "music_arena_pool.json",
    ]
    for cand in candidates:
        if cand.exists():
            with cand.open() as fp:
                return json.load(fp)
    # HuggingFace format fallback
    if (music_arena_root / "dataset_info.json").exists():
        from datasets import load_dataset

        ds = load_dataset(str(music_arena_root), split="train")
        return list(ds)
    raise FileNotFoundError(
        f"No battles.json or HuggingFace dataset found under {music_arena_root}"
    )


def filter_pool(
    pool: list[dict], bench_uuids: set[str]
) -> tuple[list[dict], dict]:
    decisive = [b for b in pool if b.get("vote") in ("A", "B")]
    matched = [b for b in decisive if b["battle_uuid"] in bench_uuids]
    kept = [b for b in decisive if b["battle_uuid"] not in bench_uuids]
    report = {
        "raw_decisive": len(decisive),
        "bench_uuid_pool_size": len(bench_uuids),
        "matched": len(matched),
        "match_rate": len(matched) / max(len(bench_uuids), 1),
        "kept_after_filter": len(kept),
    }
    return kept, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--music-arena-root", type=Path, required=True)
    ap.add_argument("--cmi-bench-uuid-list", type=Path, required=True)
    ap.add_argument("--out-pool", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, default=None)
    args = ap.parse_args()

    bench_uuids = _read_uuid_list(args.cmi_bench_uuid_list)
    pool = _load_pool(args.music_arena_root)
    kept, report = filter_pool(pool, bench_uuids)

    args.out_pool.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pool.open("w") as fp:
        json.dump(kept, fp, indent=2)

    if args.out_report:
        with args.out_report.open("w") as fp:
            json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {len(kept)} bench-clean battles to {args.out_pool}")


if __name__ == "__main__":
    main()
