"""Full leave-one-out train-by-test accuracy matrix (paper Table 3).

Evaluates the released checkpoint and the four leave-one-out variants on
every per-dataset held-out test split (non-tie pairs), producing the
5 x 4 matrix in paper Section 4.1 Table 3. The All column of the paper
table is the pair-weighted average of the four splits and is covered by
``eval/full_test_gain.py``.

Expected (paper Table 3):

    mix     AIME(1560)  MP(206)  MA(20)  SE(249)
    Full    0.674       0.718    0.800   0.908
    -AIME   0.625       0.689    0.650   0.920
    -MP     0.672       0.689    0.700   0.912
    -MA     0.673       0.704    0.750   0.908
    -SE     0.686       0.718    0.750   0.815

Usage
-----
$ python -m eval.loo_matrix \
      --checkpoint checkpoints/tunejury.pt \
      --leave-out-checkpoints checkpoints/tunejury_leave_AIME.pt \
                              checkpoints/tunejury_leave_MP.pt \
                              checkpoints/tunejury_leave_MA.pt \
                              checkpoints/tunejury_leave_SE.pt \
      --features-dir data/processed_features \
      --splits-dir data/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.internal_per_dataset import _evaluate  # noqa: E402

SPLIT_FILES = {
    "AIME": "test_aime.txt",
    "MP": "test_musicprefs.txt",
    "MA": "test_music_arena.txt",
    "SE": "test_songeval.txt",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--leave-out-checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = [("Full", args.checkpoint)]
    for cp in args.leave_out_checkpoints:
        tag = cp.stem.replace("tunejury_leave_", "-")
        rows.append((tag, cp))

    header = "mix     " + "  ".join(f"{k:>10s}" for k in SPLIT_FILES)
    print(header)
    for name, cp in rows:
        cells = []
        for split, fname in SPLIT_FILES.items():
            acc, n = _evaluate(cp, args.splits_dir / fname, args.features_dir)
            cells.append(f"{acc:.3f} (n={n})")
        print(f"{name:7s} " + "  ".join(f"{c:>10s}" for c in cells))


if __name__ == "__main__":
    main()
