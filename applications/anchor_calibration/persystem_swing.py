"""Per-system swing of cross-month anchor calibration (Figure `ood_scaling`(b)).

Quantifies why Feb-Mar anchor-calibration fits do not transfer to April: fit
per-system Bradley-Terry offsets on the full Feb-Mar pool (in-distribution
anchor pinned at beta=0), apply them to April, and report each system's change
in pairwise agreement (calibrated minus raw) on the April pairs it appears in.
Positive and negative swings cancel, so overall April agreement is unchanged.

The figure caption quotes the min/max of these per-system swings. Run:

$ python -m applications.anchor_calibration.persystem_swing \
    --checkpoint checkpoints/tunejury.pt \
    --feat-dir   <MR>/data/processed/MusicArena/features \
    --label-csv  data/labels/ma_postcut_scored.csv \
    --april-feat-dir <april>/features --april-meta-dir <april>/json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury  # noqa: E402

from .fit import AnchorCalibrator
from .cross_month_eval import ANCHOR_SYSTEM, DEVICE, side_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/tunejury.pt")
    ap.add_argument("--feat-dir", required=True, help="Feb-Mar feature cache")
    ap.add_argument("--label-csv", required=True, help="Feb-Mar CSV labels")
    ap.add_argument("--april-feat-dir", required=True)
    ap.add_argument("--april-meta-dir", required=True)
    ap.add_argument("--min-pairs", type=int, default=10,
                    help="Only report systems with at least this many April pairs.")
    args = ap.parse_args()

    model = TuneJury(input_dim=2048)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Feb-Mar: fit beta on the full pool (in-distribution anchor pinned at 0).
    fm_rows = list(csv.DictReader(open(args.label_csv)))
    fm_label = {r["battle_id"]: 1 if r["preference"] == "A" else 0 for r in fm_rows}
    fm_system = {r["battle_id"]: (r["system_a"], r["system_b"]) for r in fm_rows}
    fm_side = side_scores(model, Path(args.feat_dir), list(fm_label))
    records = [(*fm_side[u], *fm_system[u], fm_label[u]) for u in fm_side]
    cal = AnchorCalibrator(l2=1.0).fit(records, anchor_system=ANCHOR_SYSTEM)

    # April: per-system agreement, raw vs calibrated.
    apr_feat = Path(args.april_feat_dir)
    apr_meta = Path(args.april_meta_dir)
    apr_uuids = sorted(p.stem for p in apr_feat.glob("*.pt"))
    apr_side = side_scores(model, apr_feat, apr_uuids)
    raw = defaultdict(lambda: [0, 0])
    cal_ = defaultdict(lambda: [0, 0])
    for u in apr_side:
        sa, sb = apr_side[u]
        d = torch.load(apr_feat / f"{u}.pt", map_location="cpu", weights_only=False)
        y = 1 if d["winner"] == "model_a" else 0
        meta = json.loads((apr_meta / f"{u}.json").read_text())
        sys_a, sys_b = meta["model_a"], meta["model_b"]
        rp = 1 if sa > sb else 0
        cp = 1 if cal.correct(sa, sys_a) > cal.correct(sb, sys_b) else 0
        for s in (sys_a, sys_b):
            raw[s][0] += rp == y
            raw[s][1] += 1
            cal_[s][0] += cp == y
            cal_[s][1] += 1

    swing = {s: (cal_[s][0] / cal_[s][1] - raw[s][0] / raw[s][1]) * 100
             for s in raw if raw[s][1] >= args.min_pairs}
    vals = sorted(swing.values())
    print(f"per-system swing range: {min(vals):+.1f} to {max(vals):+.1f} pp (net cancels)")
    for s, v in sorted(swing.items(), key=lambda kv: -kv[1]):
        print(f"  {v:+5.1f} pp  {s}")


if __name__ == "__main__":
    main()
