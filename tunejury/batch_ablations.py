"""Train the six leave-one-dataset-out + two leave-two-out ablation checkpoints.

Paper anchor: Section 4.2 (Training-mix breadth) + Appendix D (Table training_mix).

Outputs six checkpoints under ``--out-dir``:

* ``tunejury_leave_MA.pt``     (drop Music Arena)
* ``tunejury_leave_MP.pt``     (drop MusicPrefs)
* ``tunejury_leave_AIME.pt``   (drop AIME)
* ``tunejury_leave_SE.pt``     (drop SongEval)
* ``tunejury_leave_SE_MA.pt``  (drop SongEval + Music Arena)
* ``tunejury_leave_MP_MA.pt``  (drop MusicPrefs + Music Arena)

All variants share the same hyperparameters as the released checkpoint
(AdamW lr 1e-4 / wd 1e-3 / batch 32, early stopping patience 30).

Usage
-----
$ python -m tunejury.batch_ablations \\
      --features-dir data/processed_features \\
      --splits-dir   data/splits \\
      --out-dir      checkpoints/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    ("leave_MA", "train_no_MA.txt"),
    ("leave_MP", "train_no_MP.txt"),
    ("leave_AIME", "train_no_AIME.txt"),
    ("leave_SE", "train_no_SE.txt"),
    ("leave_SE_MA", "train_no_SE_MA.txt"),
    ("leave_MP_MA", "train_no_MP_MA.txt"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-ids", type=Path, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_ids = args.val_ids or args.splits_dir / "val.txt"

    for tag, train_file in ABLATIONS:
        out_ckpt = args.out_dir / f"tunejury_{tag}.pt"
        if args.skip_existing and out_ckpt.exists():
            print(f"[{tag}] exists, skipping.")
            continue
        train_ids = args.splits_dir / train_file
        if not train_ids.exists():
            print(f"[{tag}] WARNING: {train_ids} not found, skipping.")
            continue
        cmd = [
            sys.executable,
            "-m",
            "tunejury.train",
            "--features-dir",
            str(args.features_dir),
            "--train-ids",
            str(train_ids),
            "--val-ids",
            str(val_ids),
            "--out",
            str(out_ckpt),
        ]
        print(f"[{tag}] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print(f"done. checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
