"""Reproduce the §A.D post-cutoff anchor-calibration sweep.

Requires:
  - the released TuneJury checkpoint at `checkpoints/tunejury.pt`
  - the music-ranknet MusicArena feature cache at
    `<MR_ROOT>/data/processed/MusicArena/features/{uuid}.pt`
  - the v2 increment split file `MusicArena_v2_increment.json`
  - the post-cutoff CSV labels `ma_postcut_scored.csv`

Outputs:
  - prints a K-vs-accuracy table (paper Table `tab:ood_scaling` in Appendix D)
  - writes `results/ood_scaling_results.json`
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .fit import AnchorCalibrator

# Allow `from tunejury.model import TuneJury` either from a checkout or from
# the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUES = [0, 3, 10, 30, 100, 250]
SEEDS = [0, 1, 2, 3, 4]
# In-distribution reference system pinned at beta=0 for identifiability
# (anchors the score scale to its training-time meaning). Results are stable
# across in-distribution anchor choices; see the anchor-robustness check.
ANCHOR_SYSTEM = "sonauto-v2-2"


def load_side_scores(
    model: TuneJury, feat_dir: Path, uuids: list[str]
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    model.eval()
    with torch.no_grad():
        for u in uuids:
            d = torch.load(
                feat_dir / f"{u}.pt", map_location=DEVICE, weights_only=False
            )
            te = d["text_emb"].to(DEVICE).float()
            x_a = torch.cat(
                [d["clap_a"].to(DEVICE).float(), d["mert_a"].to(DEVICE).float(), te]
            ).unsqueeze(0)
            x_b = torch.cat(
                [d["clap_b"].to(DEVICE).float(), d["mert_b"].to(DEVICE).float(), te]
            ).unsqueeze(0)
            sa = float(model(x_a).item())
            sb = float(model(x_b).item())
            out[u] = (sa, sb)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tunejury.pt",
        help="released TuneJury head weights",
    )
    parser.add_argument(
        "--feat-dir",
        required=True,
        help="MusicArena feature cache directory (per-pair .pt files)",
    )
    parser.add_argument(
        "--increment-split",
        required=True,
        help="MusicArena_v2_increment.json (post-cutoff UUIDs)",
    )
    parser.add_argument(
        "--label-csv",
        required=True,
        help="ma_postcut_scored.csv (CSV labels per UUID)",
    )
    parser.add_argument("--out", default="results/ood_scaling_results.json")
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    increment = json.loads(Path(args.increment_split).read_text())
    csv_rows = list(csv.DictReader(open(args.label_csv)))
    csv_label = {
        r["battle_id"]: 1 if r["preference"] == "A" else 0 for r in csv_rows
    }
    csv_system = {
        r["battle_id"]: (r["system_a"], r["system_b"]) for r in csv_rows
    }
    universe = [
        u
        for u in increment["new_uuids"]
        if u in csv_label and (feat_dir / f"{u}.pt").exists()
    ]
    print(f"universe size: {len(universe)}")

    model = TuneJury(input_dim=2048)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("caching side scores for universe...")
    side = load_side_scores(model, feat_dir, universe)

    results: dict[str, dict[int, list[float]]] = {"anchor": defaultdict(list)}
    for K in K_VALUES:
        for seed in SEEDS:
            rng = random.Random(seed)
            shuf = universe.copy()
            rng.shuffle(shuf)
            train_uuids = shuf[:299]
            test_uuids = shuf[299:598]

            calibrator = AnchorCalibrator(l2=1.0)
            if K > 0:
                anchor_records = [
                    (*side[u], *csv_system[u], csv_label[u])
                    for u in train_uuids[:K]
                ]
                calibrator.fit(anchor_records, anchor_system=ANCHOR_SYSTEM)

            correct = 0
            for u in test_uuids:
                sa, sb = side[u]
                sys_a, sys_b = csv_system[u]
                pred = (
                    1
                    if calibrator.correct(sa, sys_a)
                    > calibrator.correct(sb, sys_b)
                    else 0
                )
                if pred == csv_label[u]:
                    correct += 1
            acc = correct / len(test_uuids)
            results["anchor"][K].append(acc)
            print(f"K={K:>3d}  seed={seed}  acc={acc:.4f}")

    out = {k: dict(v) for k, v in results.items()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\nSUMMARY (mean ± std, 5 seeds):")
    print(f"{'K':>5s}  {'anchor-calibration':>22s}")
    for K in K_VALUES:
        a = np.array(results["anchor"][K])
        print(f"{K:>5d}  {a.mean()*100:5.2f} ± {a.std()*100:4.2f}")


if __name__ == "__main__":
    main()
