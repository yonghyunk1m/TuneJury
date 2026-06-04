"""Batch CMI-RewardBench evaluation across all training-mix ablations.

Paper anchor: Appendix D (Table training_mix).

Sweeps the released TuneJury + 6 leave-out variants across the four
CMI-RewardBench test splits (PAM SRCC / MusicEval SRCC / CMI-Pref
pairwise accuracy / Music Arena pairwise accuracy). Wraps
``eval.cmi_rewardbench`` for each checkpoint and writes a single CSV.

Usage
-----
$ python -m eval.cmi_rewardbench_sweep \\
      --checkpoints checkpoints/tunejury.pt \\
                    checkpoints/tunejury_leave_AIME.pt \\
                    checkpoints/tunejury_leave_MP.pt \\
                    checkpoints/tunejury_leave_MA.pt \\
                    checkpoints/tunejury_leave_SE.pt \\
                    checkpoints/tunejury_leave_SE_MA.pt \\
                    checkpoints/tunejury_leave_MP_MA.pt \\
      --bench-root /path/to/CMI-RewardBench \\
      --out-csv    results/training_mix.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _label_from_ckpt(ckpt: Path) -> str:
    stem = ckpt.stem
    if stem == "tunejury":
        return "Primary released (4-dataset)"
    if "leave_" in stem:
        tag = stem.split("leave_", 1)[1]
        parts = tag.upper().split("_")
        if len(parts) > 1:
            return f"leave-({'+'.join(parts)})-out"
        return f"leave-{parts[0]}-out"
    return stem


def _run_one(ckpt: Path, bench_root: Path, protocol: str) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "eval.cmi_rewardbench",
        "--checkpoint",
        str(ckpt),
        "--bench-root",
        str(bench_root),
        "--prompt-protocol",
        protocol,
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--bench-root", type=Path, required=True)
    ap.add_argument("--prompt-protocol", default="with", choices=["with", "empty"])
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    print(f"{'Model':<35} {'PAM':>8} {'MusicEval':>10} {'CMI-Pref':>10} {'MA bench':>10}")
    for ckpt in args.checkpoints:
        label = _label_from_ckpt(ckpt)
        res = _run_one(ckpt, args.bench_root, args.prompt_protocol)
        print(
            f"{label:<35} {res['PAM']:>8.4f} {res['MusicEval']:>10.4f} "
            f"{res['CMI-Pref']:>10.4f} {res['MA bench']:>10.4f}"
        )
        rows.append({"model": label, **res})

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w") as fp:
            fp.write("model,PAM,MusicEval,CMI-Pref,MA_bench\n")
            for r in rows:
                fp.write(
                    f"{r['model']},{r['PAM']:.4f},{r['MusicEval']:.4f},"
                    f"{r['CMI-Pref']:.4f},{r['MA bench']:.4f}\n"
                )
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
