"""Build the pair-id split files used by tunejury.train / batch_ablations.

Reads the four per-dataset random-split JSON files in ``data/splits/`` and
emits, in the same directory:

* ``train.txt`` / ``val.txt`` / ``test.txt`` — the full four-dataset pools.
* ``test_music_arena.txt`` / ``test_musicprefs.txt`` / ``test_aime.txt`` /
  ``test_songeval.txt`` — per-dataset held-out test splits.
* ``train_no_MA.txt`` / ``train_no_MP.txt`` / ``train_no_AIME.txt`` /
  ``train_no_SE.txt`` — single-dataset leave-out training pools.
* ``train_no_SE_MA.txt`` / ``train_no_MP_MA.txt`` — two-dataset leave-out
  training pools.

The four JSON inputs match the paper's reported splits:

* ``MusicArena-random_split_bench_clean.json`` (Music Arena, bench-clean):
  train 571 / valid 54 / test 74.
* ``MusicPrefs-random_split.json``: train 2012 / valid 251 / test 252.
* ``AIME-random_split.json``: train 12480 / valid 1560 / test 1560.
* ``SongEval-balanced_split.json``: train 2491 / valid 246 / test 249.

Combined: train 17554 / valid 2111 / test 2135 (matches Section 3).

Usage
-----
$ python -m data.build_splits

The script is idempotent and overwrites the .txt outputs every run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLIT_FILES = {
    "MA":   "MusicArena-random_split_bench_clean.json",
    "MP":   "MusicPrefs-random_split.json",
    "AIME": "AIME-random_split.json",
    "SE":   "SongEval-balanced_split.json",
}

# Per-dataset test-split filename (matches the paper's per-dataset Tables).
PER_DATASET_TEST = {
    "MA":   "test_music_arena.txt",
    "MP":   "test_musicprefs.txt",
    "AIME": "test_aime.txt",
    "SE":   "test_songeval.txt",
}

# Leave-out training pools (Section 4 design-space ablations + Appendix D).
LEAVE_OUT = {
    "train_no_MA.txt":    ["MP", "AIME", "SE"],
    "train_no_MP.txt":    ["MA", "AIME", "SE"],
    "train_no_AIME.txt":  ["MA", "MP", "SE"],
    "train_no_SE.txt":    ["MA", "MP", "AIME"],
    "train_no_SE_MA.txt": ["MP", "AIME"],
    "train_no_MP_MA.txt": ["AIME", "SE"],
}


def _load_split(splits_dir: Path, key: str) -> dict[str, list[str]]:
    fname = SPLIT_FILES[key]
    return json.loads((splits_dir / fname).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "splits",
        help="Directory containing the four per-dataset JSON splits.",
    )
    args = ap.parse_args()

    splits_dir: Path = args.splits_dir
    splits = {k: _load_split(splits_dir, k) for k in SPLIT_FILES}

    def write(name: str, ids: list[str]) -> None:
        (splits_dir / name).write_text("\n".join(ids))
        print(f"  {name}: {len(ids)} ids")

    print(f"Writing split files to {splits_dir}/")

    # Full four-dataset pools.
    for partition, fname in [("train", "train.txt"), ("valid", "val.txt"), ("test", "test.txt")]:
        ids = [u for k in SPLIT_FILES for u in splits[k][partition]]
        write(fname, ids)

    # Per-dataset test splits.
    for k, fname in PER_DATASET_TEST.items():
        write(fname, splits[k]["test"])

    # Leave-out training pools.
    for fname, keys in LEAVE_OUT.items():
        ids = [u for k in keys for u in splits[k]["train"]]
        write(fname, ids)


if __name__ == "__main__":
    main()
