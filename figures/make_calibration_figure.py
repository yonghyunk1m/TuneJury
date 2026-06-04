"""Reliability diagram + win rate vs predicted margin (Appendix A, Figure
``calibration``).

Paper anchor: §A.A reliability + Table ``calibration_bins``.

Reads the bins JSON emitted by ``eval/internal.py --out-bins-json`` and
produces a two-panel figure: (a) reliability diagram (decile bins), (b)
empirical win rate vs absolute predicted margin with the
$p_{95}$ / max training-margin lines for extrapolation context.

Usage
-----
$ python -m eval.internal \\
      --checkpoint checkpoints/tunejury.pt \\
      --features-dir data/processed_features \\
      --test-ids data/splits/test.txt \\
      --out-bins-json results/calibration_bins.json

$ python figures/make_calibration_figure.py \\
      --bins-json results/calibration_bins.json \\
      --out       figures/fig_calibration.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_PRIMARY = "#3d5a80"  # deep blue, matches Figure 5 anchor color family
COLOR_GUIDE = "#888888"    # neutral gray for identity / threshold lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bins-json", type=Path, required=True,
                    help="output of `python -m eval.internal --out-bins-json`")
    ap.add_argument("--out", type=Path, required=True,
                    help="output PDF path (paper uses figures/fig_calibration.pdf)")
    args = ap.parse_args()

    d = json.loads(args.bins_json.read_text())
    bins = d["overall"]["bins"]
    confs = np.array([b["mean_confidence"] for b in bins])
    wins = np.array([b["empirical_winrate"] for b in bins])
    margins = np.array([b["mean_abs_margin"] for b in bins])
    ns = np.array([b["n"] for b in bins])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    # ---- Panel (a): Reliability diagram ----
    ax1.plot([0.5, 1.0], [0.5, 1.0], "--", color=COLOR_GUIDE, lw=1.2, alpha=0.7,
             label="Perfect calibration")
    sizes = np.clip(ns / ns.max() * 80, 18, 80)
    ax1.scatter(confs, wins, s=sizes, alpha=0.85, color=COLOR_PRIMARY,
                edgecolor="white", linewidth=0.6, zorder=3,
                label="TuneJury (decile bins)")
    for c, w in zip(confs, wins):
        ax1.plot([c, c], [min(c, w), max(c, w)], "-", color=COLOR_PRIMARY,
                 alpha=0.30, lw=1.0)
    ax1.set_xlabel(r"Mean predicted confidence $\mathrm{sigmoid}(|s(A){-}s(B)|)$")
    ax1.set_ylabel("Empirical win rate")
    ax1.set_xlim(0.48, 1.02)
    ax1.set_ylim(0.48, 1.02)
    ax1.set_title("(a) Reliability diagram")
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ---- Panel (b): Win-rate vs. predicted margin ----
    # Same scatter style as panel (a) (circle, size scaled by n, primary blue)
    # so the two panels of this figure share a visual convention. No connecting
    # line: each dot is a distinct decile bin, and the trend is visible from
    # the 10 sorted points without implying linear interpolation across bins.
    sizes_b = np.clip(ns / ns.max() * 80, 18, 80)
    ax2.scatter(margins, wins, s=sizes_b, alpha=0.85, color=COLOR_PRIMARY,
                edgecolor="white", linewidth=0.6, zorder=3,
                label="TuneJury (decile bins)")
    ax2.set_xlabel(r"Mean $|s(A) - s(B)|$")
    ax2.set_ylabel("Empirical win rate")
    ax2.set_xlim(0, margins.max() * 1.05)
    ax2.set_ylim(0.48, 1.02)
    ax2.set_title("(b) Win rate vs. predicted margin")
    ax2.grid(alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
