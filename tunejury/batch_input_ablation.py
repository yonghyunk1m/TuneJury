"""Train the seven input-modality ablation variants.

Paper anchor: Section 4.1 (Input ablation) + Appendix C (Table feature_modality).

Seven variants:

| ID  | Input scope                                | Trainable input dim |
|-----|--------------------------------------------|---------------------|
| A1  | CLAP audio only                            | 512                 |
| A2  | MERT audio only                            | 1024                |
| A3  | CLAP text only                             | 512                 |
| A4  | CLAP audio + MERT audio                    | 1536                |
| A5  | CLAP audio + CLAP text                     | 1024                |
| A6  | MERT audio + CLAP text                     | 1536                |
| A7  | CLAP audio + MERT audio + CLAP text (full) | 2048 (released)     |

Each variant uses identical hyperparameters as the released checkpoint
(AdamW lr 1e-4 / wd 1e-3 / batch 32). Single-seed reproducer; absolute
accuracies differ from the released checkpoint by ~0.01 due to seed
variance, but the between-variant ordering is stable.

Usage
-----
$ python -m tunejury.batch_input_ablation \\
      --features-dir data/processed_features \\
      --splits-dir   data/splits \\
      --out-dir      checkpoints/input_ablation/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = [
    ("A1_clap_audio_only", "--no-mert --no-text"),
    ("A2_mert_only",       "--no-clap-audio --no-text"),
    ("A3_text_only",       "--no-clap-audio --no-mert"),
    ("A4_clap_audio+mert", "--no-text"),
    ("A5_clap_audio+text", "--no-mert"),
    ("A6_mert+text",       "--no-clap-audio"),
    ("A7_full",            ""),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = args.splits_dir / "train.txt"
    val_ids = args.splits_dir / "val.txt"

    for tag, flags in VARIANTS:
        out_ckpt = args.out_dir / f"{tag}.pt"
        if args.skip_existing and out_ckpt.exists():
            print(f"[{tag}] exists, skipping.")
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
        cmd.extend(flags.split())
        print(f"[{tag}] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print(f"done. checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
