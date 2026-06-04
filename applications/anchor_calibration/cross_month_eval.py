"""Cross-month generalization eval (Paper Appendix §A.D, Table `cross_month`).

Fit per-system beta on a 'previous month' slice, evaluate on a held-out
later-month slice. Also runs the within-later-month sanity probe (50/50
split) to bound how much within-month memorization vs. cross-month
generalization is happening.

Paper §A.D finding: April raw agreement is already ~64% (K=0: 64.0),
materially higher than the 0.551 raw on the Feb-Mar 598-pair anchor
universe. Anchor calibration fit on Feb-Mar does NOT transfer to April
at small K (K=30 regresses ~2.9pp below raw to 61.1 +/- 3.5); the
cross-month curve only recovers to baseline (64.2 +/- 1.2) at K=598.
The within-April 50/50 sanity probe is flat across K (raw 63.4,
K=200 63.3, n_test=199). Per-system biases are month-specific.

Inputs:
  --checkpoint        released TuneJury head weights
  --feat-dir          MusicArena feature cache (Feb-Mar pairs, 598 anchor-eligible)
  --label-csv         Feb-Mar CSV labels
  --april-feat-dir    April feature cache (397 decisive pairs)
  --april-meta-dir    April companion JSONs with model_a/model_b
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury  # noqa: E402

from .fit import AnchorCalibrator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# In-distribution reference system pinned at beta=0 for identifiability
# (matches run_experiment.py). Unseen post-cutoff systems carry no fitted
# offset and receive no correction (beta=0 by default).
ANCHOR_SYSTEM = "sonauto-v2-2"


def side_scores(model: TuneJury, feat_dir: Path, uuids: list[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    model.eval()
    with torch.no_grad():
        for u in uuids:
            pt = feat_dir / f"{u}.pt"
            if not pt.exists():
                continue
            d = torch.load(pt, map_location=DEVICE, weights_only=False)
            te = d["text_emb"].to(DEVICE).float()
            f_a = torch.cat(
                [d["clap_a"].to(DEVICE).float(), d["mert_a"].to(DEVICE).float(), te]
            ).unsqueeze(0)
            f_b = torch.cat(
                [d["clap_b"].to(DEVICE).float(), d["mert_b"].to(DEVICE).float(), te]
            ).unsqueeze(0)
            out[u] = (float(model(f_a).item()), float(model(f_b).item()))
    return out


def eval_accuracy(side: dict, label: dict, systems: dict, calibrator: AnchorCalibrator) -> float:
    n_ok = 0
    for u, (sa, sb) in side.items():
        sys_a, sys_b = systems[u]
        pred = 1 if calibrator.correct(sa, sys_a) > calibrator.correct(sb, sys_b) else 0
        if pred == label[u]:
            n_ok += 1
    return n_ok / len(side)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tunejury.pt")
    parser.add_argument("--feat-dir", required=True, help="Feb-Mar feature cache")
    parser.add_argument("--label-csv", required=True, help="Feb-Mar CSV labels")
    parser.add_argument("--april-feat-dir", required=True)
    parser.add_argument("--april-meta-dir", required=True)
    parser.add_argument("--out", default="results/cross_month.json")
    args = parser.parse_args()

    model = TuneJury(input_dim=2048)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Feb-Mar scores + labels
    fm_rows = list(csv.DictReader(open(args.label_csv)))
    fm_label = {r["battle_id"]: 1 if r["preference"] == "A" else 0 for r in fm_rows}
    fm_system = {r["battle_id"]: (r["system_a"], r["system_b"]) for r in fm_rows}
    fm_uuids = list(fm_label.keys())
    fm_side = side_scores(model, Path(args.feat_dir), fm_uuids)
    fm_uuids = list(fm_side.keys())
    print(f"Feb-Mar scored: {len(fm_uuids)}")

    # April scores + labels (label and system from companion JSON)
    apr_feat_dir = Path(args.april_feat_dir)
    apr_meta_dir = Path(args.april_meta_dir)
    apr_uuids = sorted(p.stem for p in apr_feat_dir.glob("*.pt"))
    apr_side = side_scores(model, apr_feat_dir, apr_uuids)
    apr_uuids = list(apr_side.keys())
    apr_label, apr_system = {}, {}
    for u in apr_uuids:
        d = torch.load(apr_feat_dir / f"{u}.pt", map_location="cpu", weights_only=False)
        apr_label[u] = 1 if d["winner"] == "model_a" else 0
        meta = json.loads((apr_meta_dir / f"{u}.json").read_text())
        apr_system[u] = (meta["model_a"], meta["model_b"])
    print(f"April scored: {len(apr_uuids)}")

    K_TOTAL = [0, 30, 100, 200, 300, 598]
    SEEDS = list(range(10))

    out = {"cross_month": defaultdict(list), "within_april": defaultdict(list)}

    # Cross-month: fit on Feb-Mar, eval on April
    print("\n=== Cross-month: Feb-Mar fit -> April eval ===")
    for K in K_TOTAL:
        for seed in SEEDS:
            rng = random.Random(seed)
            pool = fm_uuids.copy()
            rng.shuffle(pool)
            anchors = pool[:K] if K > 0 else []
            anchor_records = [
                (*fm_side[u], *fm_system[u], fm_label[u]) for u in anchors
            ]
            cal = AnchorCalibrator(l2=1.0).fit(anchor_records, anchor_system=ANCHOR_SYSTEM)
            acc = eval_accuracy(apr_side, apr_label, apr_system, cal)
            out["cross_month"][K].append(acc)
        accs = out["cross_month"][K]
        print(f"  K={K:>3d}  mean {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # Within-April: 50/50 sanity
    print("\n=== Within-April: 50/50 sanity ===")
    for K in [0, 30, 100, 200]:
        for seed in SEEDS:
            rng = random.Random(seed)
            pool = apr_uuids.copy()
            rng.shuffle(pool)
            half = len(pool) // 2
            train, test = pool[:half], pool[half:]
            anchors = train[:K] if K > 0 else []
            anchor_records = [
                (*apr_side[u], *apr_system[u], apr_label[u]) for u in anchors
            ]
            cal = AnchorCalibrator(l2=1.0).fit(anchor_records, anchor_system=ANCHOR_SYSTEM)
            test_side = {u: apr_side[u] for u in test}
            test_label = {u: apr_label[u] for u in test}
            test_system = {u: apr_system[u] for u in test}
            acc = eval_accuracy(test_side, test_label, test_system, cal)
            out["within_april"][K].append(acc)
        accs = out["within_april"][K]
        print(f"  K={K:>3d}  mean {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({k: dict(v) for k, v in out.items()}, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
