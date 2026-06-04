"""Anchor calibration K-sweep + April cross-month transfer (Figure
``ood_scaling``, Appendix D).

Two-panel figure:
- (a) Feb-Mar 50/50 split, $n_\\text{test}{=}299$, 5 seeds: anchor (A) vs
       retrain (R) K-sweep with the data-efficiency callout (anchor at
       smallest K matching retrain saturation).
- (b) Cross-month Feb-Mar $\\beta_s$ fits applied to April 2026
       ($n_\\text{test}{=}397$, 10 seeds), with a within-April 50/50 sanity
       probe overlaid.

Usage
-----
$ # Step 1: anchor K-sweep
$ python -m applications.anchor_calibration.run_experiment \\
      --checkpoint checkpoints/tunejury.pt \\
      --feat-dir /path/to/musicarena/features \\
      --increment-split data/splits/MusicArena_v2_increment.json \\
      --label-csv data/labels/ma_postcut_scored.csv \\
      --out results/ood_repro.json

$ # Step 2: retrain K-sweep (Phase 2)
$ python -m applications.anchor_calibration.retrain_ksweep \\
      --feat-dir /path/to/musicarena/features \\
      --bench-split data/splits/MusicArena-random_split_bench_clean.json \\
      --increment-split data/splits/MusicArena_v2_increment.json \\
      --label-csv data/labels/ma_postcut_scored.csv \\
      --out results/ood_retrain.json

$ # Step 3: cross-month
$ python -m applications.anchor_calibration.cross_month_eval \\
      --checkpoint checkpoints/tunejury.pt \\
      --feat-dir /path/to/musicarena/features \\
      --label-csv data/labels/ma_postcut_scored.csv \\
      --april-feat-dir /path/to/april/features \\
      --april-meta-dir /path/to/april/json \\
      --out results/cross_month.json

$ # Step 4: render figure
$ python figures/make_ood_scaling_figure.py \\
      --anchor-json     results/ood_repro.json \\
      --retrain-json    results/ood_retrain.json \\
      --cross-month-json results/cross_month.json \\
      --out             figures/ood_scaling.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CEILING = 0.72  # CMI-RewardBench MA in-distribution
COLOR_R = "#e07a5f"  # retraining (orange-red)
COLOR_A = "#3d5a80"  # anchor (deep blue)
COLOR_X = "#2a9d8f"  # cross-month (green-teal)
COLOR_W = "#666666"  # within-April (gray)
YLIM = (0.48, 0.76)


def mean_ci(values_by_k):
    ks = sorted(int(k) for k in values_by_k.keys())
    means, lo, hi = [], [], []
    for k in ks:
        arr = np.array(values_by_k[str(k)])
        m = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        means.append(m)
        lo.append(m - 1.96 * se)
        hi.append(m + 1.96 * se)
    return np.array(ks), np.array(means), np.array(lo), np.array(hi)


def add_yaxis_break(ax):
    """Bottom-of-y axis break indicator (//): y starts above 0."""
    d = 0.010
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1.1)
    y0 = 0.005
    ax.plot([-d, +d], [y0 - d, y0 + d], **kwargs)
    ax.plot([-d, +d], [y0 - d + 0.018, y0 + d + 0.018], **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-json", type=Path, required=True,
                    help="anchor K-sweep JSON from run_experiment.py")
    ap.add_argument("--retrain-json", type=Path, default=None,
                    help="retrain K-sweep JSON from retrain_ksweep.py "
                         "(if omitted, panel (a) shows anchor only)")
    ap.add_argument("--cross-month-json", type=Path, default=None,
                    help="cross-month JSON from cross_month_eval.py "
                         "(if omitted, only panel (a) is drawn)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output PDF path (paper uses figures/ood_scaling.pdf)")
    args = ap.parse_args()

    anchor = json.loads(args.anchor_json.read_text())
    retrain = json.loads(args.retrain_json.read_text()) if args.retrain_json and args.retrain_json.exists() else None
    cross = json.loads(args.cross_month_json.read_text()) if args.cross_month_json and args.cross_month_json.exists() else None

    if cross is None:
        fig, ax_a = plt.subplots(1, 1, figsize=(9.0, 4.0))
        ax_b = None
    else:
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(12.5, 4.0), gridspec_kw={"wspace": 0.18},
        )

    # ===========================================================
    # Panel (a): Feb-Mar 50/50 split
    # ===========================================================
    ks_a, m_a, lo_a, hi_a = mean_ci(anchor["anchor"])
    x_a = np.arange(len(ks_a))
    ax_a.fill_between(x_a, lo_a, hi_a, color=COLOR_A, alpha=0.18)
    ax_a.plot(x_a, m_a, "-s", color=COLOR_A, lw=2.5, ms=8, mfc=COLOR_A, mec="white",
              label=r"Anchor calibration (fit $\beta_s$ on $K$ pairs)")

    if retrain is not None:
        ks_r, m_r, lo_r, hi_r = mean_ci(retrain["retrain"])
        x_r = np.array([list(ks_a).index(int(k)) for k in ks_r])
        ax_a.fill_between(x_r, lo_r, hi_r, color=COLOR_R, alpha=0.18)
        ax_a.plot(x_r, m_r, "-o", color=COLOR_R, lw=2.5, ms=8, mfc=COLOR_R, mec="white",
                  label=r"Retraining (bench-clean $\cup$ $K$ added pairs)")

        r_sat = m_r[-1]
        # Compare at the displayed precision (0.1pp): a 57.0% vs 57.0% tie counts as
        # a match rather than failing on a sub-noise (~1e-7) float difference.
        i_match = next((i for i, am in enumerate(m_a) if round(am, 3) >= round(r_sat, 3) and ks_a[i] > 0), None)
        if i_match is not None:
            k_anchor_match = int(ks_a[i_match])
            k_retrain_sat = int(ks_r[-1])
            ratio = max(1, round(k_retrain_sat / max(k_anchor_match, 1)))
            ax_a.plot([i_match], [m_a[i_match]], "s", ms=14, mfc="none",
                      mec=COLOR_A, mew=2.0, zorder=5)
            ax_a.plot([x_r[-1]], [r_sat], "o", ms=14, mfc="none",
                      mec=COLOR_R, mew=2.0, zorder=5)
            from matplotlib.patches import FancyArrowPatch
            arc = FancyArrowPatch((i_match, m_a[i_match]), (x_r[-1], r_sat),
                                  connectionstyle="arc3,rad=-0.6",
                                  arrowstyle="-", color="black", lw=1.1, zorder=4)
            ax_a.add_patch(arc)
            label_x = (i_match + x_r[-1]) / 2
            label_y = max(m_a) + 0.060
            tag = f"~{ratio}× data efficient" if ratio >= 2 else "matches retraining"
            ax_a.text(label_x, label_y, tag,
                      ha="center", va="center", fontsize=10.5, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8),
                      zorder=6)

    ax_a.axhline(CEILING, ls="--", color="dimgray", lw=1.2)
    ax_a.text(0.02, CEILING + 0.005,
              f"in-distribution ceiling on CMI-RewardBench MA ({CEILING:.2f})",
              color="dimgray", fontsize=8.5, ha="left", va="bottom",
              transform=ax_a.get_yaxis_transform())
    ax_a.set_xticks(np.arange(len(ks_a)))
    ax_a.set_xticklabels([str(int(k)) for k in ks_a])
    ax_a.set_xlabel(r"$K$ calibration pairs (Feb-Mar slice)")
    ax_a.set_ylabel("Pairwise agreement on held-out test")
    ax_a.set_ylim(*YLIM)
    ax_a.set_title(r"(a) Feb-Mar 50/50 split, $n_{\rm test}{=}299$, 5 seeds", fontsize=11)
    ax_a.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax_a.grid(True, alpha=0.3)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    add_yaxis_break(ax_a)

    # ===========================================================
    # Panel (b): April cross-month
    # ===========================================================
    if cross is not None and ax_b is not None:
        cross_key = "cross_month" if "cross_month" in cross else "cross_month_febmar_to_april"
        ks_x_all, m_x_all, lo_x_all, hi_x_all = mean_ci(cross[cross_key])
        keep = ks_x_all <= 598
        ks_x = ks_x_all[keep]; m_x = m_x_all[keep]
        lo_x = lo_x_all[keep]; hi_x = hi_x_all[keep]
        x_x = np.arange(len(ks_x))

        ax_b.fill_between(x_x, lo_x, hi_x, color=COLOR_X, alpha=0.18)
        ax_b.plot(x_x, m_x, "-^", color=COLOR_X, lw=2.5, ms=8, mfc=COLOR_X, mec="white",
                  label=r"Cross-month: Feb-Mar $\beta_s\to$ April")

        if "within_april" in cross:
            ks_w, m_w, lo_w, hi_w = mean_ci(cross["within_april"])
            idx_lookup = {int(k): i for i, k in enumerate(ks_x)}
            w_idx, w_vals, w_lo, w_hi = [], [], [], []
            for j, k in enumerate(ks_w):
                if int(k) in idx_lookup:
                    w_idx.append(idx_lookup[int(k)])
                    w_vals.append(m_w[j])
                    w_lo.append(lo_w[j])
                    w_hi.append(hi_w[j])
            if w_idx:
                ax_b.fill_between(w_idx, w_lo, w_hi, color=COLOR_W, alpha=0.15)
                ax_b.plot(w_idx, w_vals, "--D", color=COLOR_W, lw=1.8, ms=7,
                          mfc=COLOR_W, mec="white",
                          label=r"Within-April 50/50 sanity")

        # Reference line repeats on (b) for visual continuity; label is
        # carried by panel (a) only (and the caption) to avoid duplication.
        ax_b.axhline(CEILING, ls="--", color="dimgray", lw=1.2)
        ax_b.set_xticks(np.arange(len(ks_x)))
        ax_b.set_xticklabels([str(int(k)) for k in ks_x])
        ax_b.set_xlabel(r"$K_{\rm total}$ calibration pairs (Feb-Mar)")
        ax_b.set_ylim(*YLIM)
        ax_b.set_title(r"(b) Cross-month: Feb-Mar $\to$ April, $n_{\rm test}{=}397$, 10 seeds", fontsize=11)
        ax_b.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
        ax_b.grid(True, alpha=0.3)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        add_yaxis_break(ax_b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
