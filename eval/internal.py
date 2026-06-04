"""Reproduce the internal pairwise accuracy + ECE numbers (Section 3 / 4.1).

Headline numbers:
    - 2,035-pair non-tie held-out test set (aggregated across MA = Music Arena, MP = MusicPrefs, AIME, SE = SongEval)
    - 0.7086 pairwise accuracy
    - ECE 0.0339 (over 10 margin-binned reliability bins)
    - Per-dataset: AIME 0.674 (n=1,560), MP 0.718 (n=206),
      MA 0.800 (n=20, bench-disjoint), SE 0.908 (n=249)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunejury import TuneJury, TuneJuryDataset


def _input_block(batch: dict, side: str, zero_text: bool = False) -> torch.Tensor:
    # Training-time concatenation order: [clap_audio, mert_audio, text_emb].
    # The released checkpoint expects this exact 2048-d input; any reordering
    # silently produces wrong scores (overall accuracy drops to ~0.60).
    # zero_text=True replaces the 512-d CLAP text block with a zero vector,
    # the empty-prompt "A-only" inference protocol (Section 3 / 4.2).
    text = torch.zeros_like(batch["text"]) if zero_text else batch["text"]
    return torch.cat(
        [batch[f"clap_{side}"], batch[f"mert_{side}"], text],
        dim=-1,
    )


def _ece(margins: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error matching paper §IV.A.

    Bins each pair by the absolute predicted margin |s(A) - s(B)| into 10
    equal-count bins (sorted-index split; last bin absorbs the remainder).
    Within each bin, confidence = sigmoid(|margin|) = P(predicted winner
    wins); accuracy = fraction of bin pairs where the predicted side
    matches ground truth. ECE = bin-size-weighted mean absolute gap.
    """
    abs_margins = np.abs(margins)
    sorted_idx = np.argsort(abs_margins)
    n = len(abs_margins)
    bin_size = n // n_bins
    ece = 0.0
    for i in range(n_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size if i < n_bins - 1 else n
        idx = sorted_idx[lo:hi]
        if len(idx) == 0:
            continue
        confidence = torch.sigmoid(torch.from_numpy(abs_margins[idx])).numpy().mean()
        accuracy = correct[idx].mean()
        ece += (len(idx) / n) * abs(confidence - accuracy)
    return float(ece)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ids = Path(args.test_ids).read_text().strip().splitlines()
    ds = TuneJuryDataset(args.features_dir, test_ids, "test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    head = TuneJury(input_dim=2048).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(state if "state_dict" not in state else state["state_dict"])

    a_only = getattr(args, "a_only_zero_text", False)
    margins, targets, flags = [], [], []
    margins_a = []  # A-only (zero CLAP text) protocol, populated only when a_only
    for batch in loader:
        x_a = _input_block(batch, "a").to(device)
        x_b = _input_block(batch, "b").to(device)
        m = head(x_a, x_b).view(-1).cpu().numpy()
        margins.append(m)
        if a_only:
            xa0 = _input_block(batch, "a", zero_text=True).to(device)
            xb0 = _input_block(batch, "b", zero_text=True).to(device)
            margins_a.append(head(xa0, xb0).view(-1).cpu().numpy())
        targets.append(batch["target"].view(-1).numpy())
        flags.append(batch["flag"].view(-1).numpy())

    margins = np.concatenate(margins)
    targets = np.concatenate(targets)
    flags = np.concatenate(flags)

    non_tie = targets != 0.5
    preds = (margins[non_tie] > 0).astype(float)
    correct = (preds == targets[non_tie]).astype(float)

    out = {
        "n_total": int(len(margins)),
        "n_non_tie": int(non_tie.sum()),
        "accuracy": float(correct.mean()),
        "ece": _ece(np.abs(margins[non_tie]), correct, n_bins=10),
    }

    if a_only:
        margins_a = np.concatenate(margins_a)
        preds_a = (margins_a[non_tie] > 0).astype(float)
        correct_a = (preds_a == targets[non_tie]).astype(float)
        # McNemar on the discordant pairs (no continuity correction, matching the paper).
        ta_win = int(np.sum((correct == 1) & (correct_a == 0)))
        a_win = int(np.sum((correct == 0) & (correct_a == 1)))
        n_disagree = ta_win + a_win
        chi2 = ((ta_win - a_win) ** 2) / n_disagree if n_disagree else 0.0
        # p-value for chi-square with 1 df: P(X > chi2) = erfc(sqrt(chi2 / 2)).
        p_value = math.erfc(math.sqrt(chi2 / 2)) if n_disagree else 1.0
        out["a_only_accuracy"] = float(correct_a.mean())
        out["mcnemar_disagree"] = {
            "n": n_disagree,
            "TA_correct": ta_win,
            "Aonly_correct": a_win,
            "chi2": round(chi2, 2),
            "p_value": round(p_value, 2),
        }

    print(json.dumps(out, indent=2))

    # Optional: emit 10-bin reliability data for figures/make_calibration_figure.py
    if getattr(args, "out_bins_json", None):
        abs_m = np.abs(margins[non_tie])
        sorted_idx = np.argsort(abs_m)
        n = len(abs_m)
        n_bins = 10
        bin_size = n // n_bins
        bins = []
        for i in range(n_bins):
            lo = i * bin_size
            hi = (i + 1) * bin_size if i < n_bins - 1 else n
            idx = sorted_idx[lo:hi]
            if len(idx) == 0:
                continue
            margin_lo = float(abs_m[idx].min())
            margin_hi = float(abs_m[idx].max())
            mean_margin = float(abs_m[idx].mean())
            mean_conf = float(torch.sigmoid(torch.from_numpy(abs_m[idx])).numpy().mean())
            wr = float(correct[idx].mean())
            bins.append({
                "bin": i, "margin_lo": margin_lo, "margin_hi": margin_hi,
                "n": int(len(idx)), "mean_abs_margin": mean_margin,
                "mean_confidence": mean_conf, "empirical_winrate": wr,
            })
        bins_json = {"overall": {
            "n_total": out["n_non_tie"],
            "accuracy_overall": out["accuracy"],
            "expected_calibration_error": out["ece"],
            "bins": bins,
        }}
        out_path = Path(args.out_bins_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(bins_json, indent=2))
        print(f"wrote bins JSON → {out_path}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--test-ids", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-bins-json", default=None,
                   help="if set, emit per-bin reliability data for figures/make_calibration_figure.py")
    p.add_argument("--a-only-zero-text", action="store_true",
                   help="also run the A-only (zero CLAP text vector) protocol and report the "
                        "T+A vs A-only McNemar disagreement (Section 4.2 text-branch contribution)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
