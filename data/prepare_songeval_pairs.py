"""Synthesize SongEval pairwise battles from its 5-axis aesthetic ratings.

Paper anchor: Section 3 (Training data paragraph).

SongEval~\\cite{yao2025songeval} includes 2,399 songs with per-annotator
ratings on 5 aesthetic axes (Coherence / Musicality / Memorability /
Clarity / Naturalness). We synthesize pairwise battles by drawing
random song pairs whose *mean* rating across the 5 axes differs by
at least ``--mean-gap`` (default 0.5), labeling the higher-rated side
as the preferred. The resulting set is 3,760 pairs (paper Table 1).

The released HuggingFace dataset `ASLP-lab/SongEval` includes audio +
gender + per-annotator ratings, but not lyrics/prompts (those are
not released alongside audio; see Section 3). Our text branch therefore
receives a 512-d zero vector at training for every SongEval pair.

Usage
-----
$ python data/prepare_songeval_pairs.py \\
      --songeval-root /path/to/songeval \\
      --out-pairs    data/splits/songeval_pairs.json \\
      --mean-gap     0.5 \\
      --seed         42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


AXES = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]


def _mean_rating_per_song(songeval_root: Path) -> dict[str, float]:
    """Returns {song_id: mean rating across 5 axes, averaged over annotators}."""
    annotations_path = songeval_root / "annotations.json"
    if not annotations_path.exists():
        # HuggingFace dataset format: load via datasets library if installed.
        from datasets import load_dataset

        ds = load_dataset(str(songeval_root), split="train")
        per_song = {}
        for row in ds:
            song_id = row["song_id"] if "song_id" in row else row["id"]
            ratings = row["annotation"]  # list of {annotator, Coherence, ...}
            axis_means = []
            for axis in AXES:
                vals = [r[axis] for r in ratings if axis in r]
                if vals:
                    axis_means.append(float(np.mean(vals)))
            if axis_means:
                per_song[song_id] = float(np.mean(axis_means))
        return per_song

    with annotations_path.open() as fp:
        rows = json.load(fp)
    per_song = {}
    for row in rows:
        song_id = row.get("song_id") or row.get("id")
        ratings = row["annotation"]
        axis_means = []
        for axis in AXES:
            vals = [r[axis] for r in ratings if axis in r]
            if vals:
                axis_means.append(float(np.mean(vals)))
        if axis_means:
            per_song[song_id] = float(np.mean(axis_means))
    return per_song


def synthesize(
    songeval_root: Path,
    mean_gap: float,
    seed: int,
    target_count: int | None,
) -> list[dict]:
    means = _mean_rating_per_song(songeval_root)
    song_ids = sorted(means.keys())
    rng = random.Random(seed)

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    max_attempts = 100 * (target_count or len(song_ids) ** 2)

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(song_ids, 2)
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        gap = abs(means[a] - means[b])
        if gap < mean_gap:
            continue
        seen.add(key)
        winner, loser = (a, b) if means[a] > means[b] else (b, a)
        pairs.append(
            {
                "winner": winner,
                "loser": loser,
                "mean_winner": means[winner],
                "mean_loser": means[loser],
                "mean_gap": gap,
            }
        )
        if target_count is not None and len(pairs) >= target_count:
            break

    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--songeval-root", type=Path, required=True)
    ap.add_argument("--out-pairs", type=Path, required=True)
    ap.add_argument("--mean-gap", type=float, default=0.5)
    ap.add_argument(
        "--target-count",
        type=int,
        default=3760,
        help="Stop after this many qualifying pairs (paper: 3,760).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = synthesize(args.songeval_root, args.mean_gap, args.seed, args.target_count)
    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pairs.open("w") as fp:
        json.dump(pairs, fp, indent=2)
    print(f"wrote {len(pairs)} pairs to {args.out_pairs}")


if __name__ == "__main__":
    main()
