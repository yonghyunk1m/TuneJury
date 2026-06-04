"""Three-panel probe figure: vocal + external SingMOS + popularity.

Paper anchor: Figure ``probes`` + Appendix §A.F.

(a) Internal vocal probe — Music Arena clips grouped by lyrics-field
    non-empty (prompt-level proxy for vocal-request intent).
(b) External SingMOS-Pro per-system scatter (n=141 singing-voice generation
    systems; track-level SRCC +0.19, per-system SRCC +0.44; Tang et al. 2025).
(c) Popularity probe — FMA-Large reward vs listens (track-level SRCC +0.285,
    n=106,401), log-scale x-axis with raw-listens tick labels.

Usage
-----
$ python figures/make_probes_figure.py \\
      --vocal-json      applications/probes/results/vocal_probe/summary.json \\
      --singmos-json    applications/probes/results/singmos_pro/summary.json \\
      --popularity-json applications/probes/results/amateur_pro/summary.json \\
      --out             figures/probes.pdf

The popularity panel uses a raw scatter overlay when
``applications/probes/results/amateur_pro/raw_fma.json`` exists (produced by
``popularity_probe.py``; contains 106,401 (listens, reward) pairs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocal-json", default="applications/probes/results/vocal_probe/summary.json")
    ap.add_argument("--singmos-json", default="applications/probes/results/singmos_pro/summary.json")
    ap.add_argument("--popularity-json", default="applications/probes/results/amateur_pro/summary.json")
    ap.add_argument("--popularity-raw", default="applications/probes/results/amateur_pro/raw_fma.json",
                    help="Optional raw (listens, reward) pairs for scatter overlay")
    ap.add_argument("--out", default="figures/probes.pdf")
    args = ap.parse_args()

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    vocal = json.loads(Path(args.vocal_json).read_text())
    sm = json.loads(Path(args.singmos_json).read_text())
    pop = json.loads(Path(args.popularity_json).read_text())

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.5))

    # ===== Panel (a) Internal vocal probe =====
    ax = axes[0]
    vocal_scores = vocal["vocal_requested_scores"]
    instr_scores = vocal["instrumental_scores"]
    parts = ax.violinplot([instr_scores, vocal_scores], positions=[1, 2],
                          showmedians=True, widths=0.85)
    for pc, color in zip(parts["bodies"], ["#3b82f6", "#f59e0b"]):
        pc.set_facecolor(color); pc.set_edgecolor("black")
        pc.set_linewidth(0.5); pc.set_alpha(0.75)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.4)
    for k in ("cmaxes", "cmins", "cbars"):
        if k in parts:
            parts[k].set_color("black"); parts[k].set_linewidth(0.6); parts[k].set_alpha(0.7)
    means_lr = [float(np.mean(instr_scores)), float(np.mean(vocal_scores))]
    ax.scatter([1, 2], means_lr, marker="D", s=42,
               facecolor="white", edgecolor="black", linewidth=0.8, zorder=5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Instr.\n(n={len(instr_scores)})", f"Vocal\n(n={len(vocal_scores)})"])
    ax.set_ylabel("TuneJury reward")
    ax.set_title(f"Internal vocal probe (gap +{vocal['gap']:.3f})")
    ax.grid(True, alpha=0.25, axis="y")

    # ===== Panel (b) External SingMOS-Pro per-system scatter =====
    ax = axes[1]
    per_sys = sm.get("per_system", sm)
    tj_means = np.array(per_sys["tj_means"])
    mos_means = np.array(per_sys["mos_means"])
    ax.scatter(mos_means, tj_means, s=18, c="#8b5cf6", alpha=0.65,
               edgecolor="black", linewidth=0.3)
    if len(tj_means) > 2:
        z = np.polyfit(mos_means, tj_means, 1)
        xs = np.linspace(mos_means.min(), mos_means.max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="black", linestyle="--", linewidth=0.9, alpha=0.6,
                label="linear fit (visual)")
    rho = per_sys["spearman"]
    n = per_sys["n_systems"]
    ax.set_xlabel("Per-system mean MOS (SingMOS-Pro human raters)")
    ax.set_ylabel("Per-system mean TuneJury reward")
    ax.set_title(f"External singing-voice MOS (SRCC = {rho:+.3f}, n = {n})")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # ===== Panel (c) Popularity probe =====
    ax = axes[2]
    raw_path = Path(args.popularity_raw)
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        listens = np.array(raw["listens"])
        rewards = np.array(raw["rewards"])
        listens_p = listens + 1
        ax.scatter(listens_p, rewards, s=3, c="#6b7280", alpha=0.04,
                   edgecolor="none", zorder=1, rasterized=True)
        # Decile means by log_listens
        log_listens = np.log10(listens_p)
        deciles_arr = np.percentile(log_listens, np.arange(0, 101, 10))
        decile_x_centers = []
        decile_y_means = []
        for i in range(10):
            mask = (log_listens >= deciles_arr[i]) & (log_listens <= deciles_arr[i + 1])
            if mask.any():
                decile_x_centers.append(float(np.median(listens_p[mask])))
                decile_y_means.append(float(np.mean(rewards[mask])))
        ax.plot(decile_x_centers, decile_y_means, color="#10b981", linewidth=1.5, zorder=3)
        ax.scatter(decile_x_centers, decile_y_means, marker="D", s=42,
                   facecolor="white", edgecolor="black", linewidth=0.8, zorder=5,
                   label="Decile means")
        ax.set_xscale("log")
        ax.set_xlim(10, 100000)
        ax.set_xticks([10, 100, 1000, 10000, 100000])
        ax.set_xticklabels(["10", "100", "1K", "10K", "100K"])
        ax.set_xlabel("Listens (FMA-Large)")
        ax.set_ylabel("TuneJury reward")
        ax.set_ylim(-4.0, 3.5)
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    else:
        # Fallback: decile-only line (no raw data)
        deciles = pop["deciles"]
        xs = np.arange(1, 11)
        means = [d["mean_reward"] for d in deciles]
        ax.plot(xs, means, color="#10b981", linewidth=1.5, zorder=3)
        ax.scatter(xs, means, marker="D", s=42,
                   facecolor="white", edgecolor="black", linewidth=0.8, zorder=5)
        ax.set_xlabel("Listens decile (FMA-Large)")
        ax.set_ylabel("Mean TuneJury reward")
    ax.set_title(f"Popularity probe (track-level SRCC = {pop['spearman']:+.3f}, n = {pop.get('n_tracks', 106401):,})")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
