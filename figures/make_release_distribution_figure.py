"""Release reward distribution per dataset (Figure release_distribution).

Paper anchor: Figure ``release_distribution`` + Table ``release_distribution``,
Appendix §A.I (Released Artifacts and License Interplay).

Shows the per-dataset TuneJury reward distribution across the seven
released score collections (MTG-Jamendo, SDD, MidiCaps, MTAT, MusicCaps,
FMA-Large, OpenMIC), all scored end-to-end on full audio with the
empty-prompt protocol.

Plot conventions:
  - Violin bodies: kernel density per dataset, clipped at [-4, +3.5] for clarity
  - Black bars inside each violin: medians (showmedians)
  - White diamonds: means
  - Dotted gray line: silence baseline (Appendix sanity table)
  - Shaded gray band: white-noise baseline range across amplitudes

Usage
-----
$ python figures/make_release_distribution_figure.py \\
      --scores-root /path/to/released_scores \\
      --out         figures/release_distribution.pdf

The released ``scores-root`` includes seven CSVs, one per dataset, each
with a ``reward_score`` column.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    "SDD",
    "MTG-Jamendo",
    "MidiCaps",
    "MusicCaps",
    "MTAT",
    "FMA-Large",
    "OpenMIC",
]

# Sanity baselines from Appendix sanity table (released checkpoint, zero-vector empty prompt).
SILENCE = -1.05
WHITE_NOISE_LO = -4.75  # 0 dBFS
WHITE_NOISE_HI = -3.93  # -60 dBFS
CLIP_LO = -4.0
CLIP_HI = 3.5


def _load_scores(csv_path: Path) -> np.ndarray:
    out = []
    with csv_path.open() as fp:
        for row in csv.DictReader(fp):
            out.append(float(row["reward_score"]))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mpl.rcParams.update(
        {"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )

    per_ds: dict[str, np.ndarray] = {}
    for ds in DATASETS:
        stem = ds.lower().replace("-", "_")
        # Prefer the post-bugfix rescored CSV when present (May 2026 MERT/empty-prompt fix).
        csv_path = args.scores_root / f"{stem}_scores_rescored.csv"
        if not csv_path.exists():
            csv_path = args.scores_root / f"{stem}_scores.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found")
            continue
        per_ds[ds] = _load_scores(csv_path)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))

    # Clipped data for plot (keep full data for stats)
    clipped = [np.clip(per_ds[ds], CLIP_LO, CLIP_HI) for ds in per_ds]
    means = [per_ds[ds].mean() for ds in per_ds]

    # Baseline shading first (behind violins) + foreground silence line on top of violins
    ax.axhspan(WHITE_NOISE_LO, WHITE_NOISE_HI, color="gray", alpha=0.15, zorder=0)
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.25, linewidth=0.6, zorder=1)
    # Silence line drawn ABOVE violins so it stays visible where violin bodies overlap it.
    # Dashed style is conventional for reference baselines; pattern matches legend automatically.
    ax.axhline(y=SILENCE, color="#b35858", linestyle="--", alpha=0.9, linewidth=1.3, zorder=4)

    positions = list(range(1, len(per_ds) + 1))
    parts = ax.violinplot(
        clipped,
        positions=positions,
        showmedians=True,
        widths=0.85,
    )
    palette = ["#3b82f6", "#1e3a8a", "#10b981", "#ef4444", "#f59e0b", "#6b7280", "#a855f7"]
    for pc, color in zip(parts["bodies"], palette):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
        pc.set_alpha(0.75)
    # Median bars styled black
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.4)
    for k in ("cmaxes", "cmins", "cbars"):
        if k in parts:
            parts[k].set_color("black")
            parts[k].set_linewidth(0.6)
            parts[k].set_alpha(0.7)

    # White diamonds for means
    ax.scatter(
        positions,
        means,
        marker="D",
        s=42,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{ds}\n(n={len(per_ds[ds]):,})" for ds in per_ds], fontsize=8.5)
    ax.set_ylabel("Released TuneJury reward (empty-prompt)")
    ax.set_ylim(CLIP_LO - 0.1, CLIP_HI + 0.1)
    ax.grid(True, alpha=0.25, axis="y")

    # Legend for baselines (small, lower-right)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elems = [
        Line2D([0], [0], color="#b35858", linestyle="--", linewidth=1.3,
               label=f"Silence ({SILENCE:+.2f})"),
        Patch(facecolor="gray", alpha=0.15, label=f"White noise band ({WHITE_NOISE_LO:+.2f} to {WHITE_NOISE_HI:+.2f})"),
        Line2D([0], [0], marker="D", color="black", markerfacecolor="white",
               markersize=6, linestyle="None", label="Mean"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=7.5, framealpha=0.92,
              borderpad=0.4, handletextpad=0.5)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
