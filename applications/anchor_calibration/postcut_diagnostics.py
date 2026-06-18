"""Post-cutoff failure decomposition (Paper Appendix §A.D).

Three diagnostics on the released TuneJury over the held-out post-cutoff
Music Arena slice:
  (i)  Per-system asymmetry (binomial test, miss-rate / over-rate)
  (ii) Encoder-space outlier check (CLAP / MERT cosine to in-distribution centroid)
  (iii) Confidence calibration on |Delta_TJ|

Inputs:
  --checkpoint        released TuneJury head weights
  --feat-dir          MusicArena feature cache (per-pair .pt files)
  --label-csv         post-cutoff CSV labels (battle_id, preference, system_a, system_b)
  --training-split    canonical MA training split json (for "unseen system" probe)

The CSV is the same labeled post-cutoff slice used by run_experiment.py
(see anchor_calibration/README.md). Outputs go to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_features(feat_dir: Path, uuid: str):
    d = torch.load(feat_dir / f"{uuid}.pt", map_location=DEVICE, weights_only=False)
    te = d["text_emb"].to(DEVICE).float()
    f_a = torch.cat(
        [d["clap_a"].to(DEVICE).float(), d["mert_a"].to(DEVICE).float(), te]
    ).unsqueeze(0)
    f_b = torch.cat(
        [d["clap_b"].to(DEVICE).float(), d["mert_b"].to(DEVICE).float(), te]
    ).unsqueeze(0)
    return d, f_a, f_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tunejury.pt")
    parser.add_argument("--feat-dir", required=True)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--training-split", required=True,
                        help="training UUIDs JSON for the 'unseen system' probe")
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    csv_rows = list(csv.DictReader(open(args.label_csv)))
    csv_label = {r["battle_id"]: 1 if r["preference"] == "A" else 0 for r in csv_rows}
    csv_system = {r["battle_id"]: (r["system_a"], r["system_b"]) for r in csv_rows}

    # Load TJ
    model = TuneJury(input_dim=2048)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Score every pair
    side, margins, clap_per_sys, mert_per_sys = {}, [], defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for r in csv_rows:
            u = r["battle_id"]
            pt = feat_dir / f"{u}.pt"
            if not pt.exists():
                continue
            d, f_a, f_b = load_features(feat_dir, u)
            sa, sb = model(f_a).item(), model(f_b).item()
            side[u] = (sa, sb)
            margins.append(abs(sa - sb))
            clap_per_sys[r["system_a"]].append(d["clap_a"].cpu().numpy())
            clap_per_sys[r["system_b"]].append(d["clap_b"].cpu().numpy())
            mert_per_sys[r["system_a"]].append(d["mert_a"].cpu().numpy())
            mert_per_sys[r["system_b"]].append(d["mert_b"].cpu().numpy())
    margins = np.array(margins)
    print(f"scored {len(side)} pairs", flush=True)

    # ===== (i) Per-system asymmetry =====
    print("\n=== (i) Per-system asymmetry (binomial Wilson, null = 50%) ===")
    print(f'{"system":30s} {"n":>5s} {"miss%":>7s}  {"95% CI":>14s}  {"over%":>7s}  {"95% CI":>14s}')
    all_sys = set(csv_system[u][0] for u in side) | set(csv_system[u][1] for u in side)
    for s in sorted(all_sys):
        won = [u for u in side
               if (csv_system[u][0] == s and csv_label[u] == 1)
               or (csv_system[u][1] == s and csv_label[u] == 0)]
        lost = [u for u in side
                if (csv_system[u][0] == s and csv_label[u] == 0)
                or (csv_system[u][1] == s and csv_label[u] == 1)]
        if len(won) < 20 or len(lost) < 20:
            continue
        # miss: TJ predicted opposite of human-decided win
        miss = sum(
            1 for u in won
            if (csv_system[u][0] == s and side[u][0] < side[u][1])
            or (csv_system[u][1] == s and side[u][1] < side[u][0])
        )
        over = sum(
            1 for u in lost
            if (csv_system[u][0] == s and side[u][0] > side[u][1])
            or (csv_system[u][1] == s and side[u][1] > side[u][0])
        )
        bt_m = binomtest(miss, len(won), p=0.5)
        bt_o = binomtest(over, len(lost), p=0.5)
        ci_m = bt_m.proportion_ci(method="wilson")
        ci_o = bt_o.proportion_ci(method="wilson")
        print(
            f'{s:30s} {len(won) + len(lost):>5d}'
            f'  {miss/len(won)*100:5.1f}%'
            f'  [{ci_m.low*100:4.1f},{ci_m.high*100:4.1f}]'
            f'  {over/len(lost)*100:5.1f}%'
            f'  [{ci_o.low*100:4.1f},{ci_o.high*100:4.1f}]'
        )

    # ===== (ii) Encoder-space outlier check =====
    print("\n=== (ii) Encoder-space cosine to in-distribution centroid ===")
    train_uuids = json.loads(Path(args.training_split).read_text())
    train_uuids = train_uuids.get("train", train_uuids)
    if isinstance(train_uuids, dict):
        train_uuids = list(train_uuids.keys())
    train_set = set(train_uuids)
    sys_in_dist = [s for s in clap_per_sys
                   if any(u for u in side
                          if (csv_system[u][0] == s or csv_system[u][1] == s)
                          and u in train_set)]
    sys_in_dist = sys_in_dist or list(clap_per_sys.keys())
    clap_centroid = np.mean(
        [np.mean(np.stack(clap_per_sys[s]), 0) for s in sys_in_dist], 0
    )
    mert_centroid = np.mean(
        [np.mean(np.stack(mert_per_sys[s]), 0) for s in sys_in_dist], 0
    )

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    print(f'{"system":30s} {"CLAP cos":>10s} {"MERT cos":>10s}')
    for s in sorted(clap_per_sys, key=lambda k: cos(np.mean(np.stack(clap_per_sys[k]), 0), clap_centroid)):
        clap_mean = np.mean(np.stack(clap_per_sys[s]), 0)
        mert_mean = np.mean(np.stack(mert_per_sys[s]), 0)
        print(f'{s:30s}  {cos(clap_mean, clap_centroid):8.4f}  {cos(mert_mean, mert_centroid):8.4f}')

    # ===== (iii) Confidence calibration =====
    print("\n=== (iii) |Delta_TJ| calibration (binomial Wilson, null = 50%) ===")
    correct = np.array([
        (1 if ((side[u][0] > side[u][1]) and csv_label[u] == 1)
         or ((side[u][0] < side[u][1]) and csv_label[u] == 0)
         else 0)
        for u in side
    ])
    qs = [0.0, 0.1, 0.3, 0.5, 0.8, 1.2, 2.0, 4.0]
    print(f'{"|margin| bin":>14s}  {"n":>5s}  {"acc":>7s}  {"95% CI":>14s}')
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (margins >= lo) & (margins < hi)
        if mask.sum() == 0:
            continue
        k = int(correct[mask].sum())
        n = int(mask.sum())
        bt = binomtest(k, n, p=0.5)
        ci = bt.proportion_ci(method="wilson")
        print(f'  [{lo:.2f},{hi:.2f})  {n:5d}  {k/n:.4f}  [{ci.low:.3f},{ci.high:.3f}]')

    # Final headline numbers used in the paper paragraph
    low = correct[margins < 0.5].mean()
    high = correct[margins > 1.2].mean()
    print(f"\nHeadline: |Delta|<0.5 -> acc={low:.4f}; |Delta|>1.2 -> acc={high:.4f}")


if __name__ == "__main__":
    main()
