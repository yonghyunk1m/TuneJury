"""Per-system reward ranking visualization (Figure per_system_test).

Paper anchor: Figure ``per_system_test`` + Appendix §A.F.

Three-panel scatter: per-system mean TuneJury reward (x) vs per-system
human win rate (y), for AIME / MusicPrefs / Music Arena held-out test
splits. Marker shape encodes model type (circle = open-weights,
triangle = proprietary, square = real audio). Color encodes vocal
capability (orange = vocal-capable, blue = instrumental-only, gray =
real audio).

Usage
-----
$ python figures/make_per_system_figure.py \\
      --aime-json       applications/probes/results/per_system/aime.json \\
      --musicprefs-json applications/probes/results/per_system/musicprefs.json \\
      --music-arena-json applications/probes/results/per_system/music_arena.json \\
      --out             figures/per_system_test.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SHAPES = {"open-weights": "o", "proprietary": "^", "real": "s"}
COLORS = {"vocal": "#f59e0b", "instrumental": "#3b82f6", "real": "gray"}


def _scatter_panel(ax, data: dict, title: str, label_rho: float | None) -> None:
    for sys_name, row in data.items():
        ax.scatter(
            row["reward_mean"],
            row["win_rate"],
            s=80,
            marker=SHAPES.get(row.get("type", "open-weights"), "o"),
            c=COLORS.get(row.get("vocal_cap", "instrumental"), "#3b82f6"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        ax.annotate(
            sys_name,
            (row["reward_mean"], row["win_rate"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    if label_rho is not None:
        ax.text(
            0.05,
            0.95,
            rf"$\rho = {label_rho:+.3f}$",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
        )
    ax.set_xlabel("Mean TuneJury reward")
    ax.set_ylabel("Per-system win rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aime-json", type=Path, required=True)
    ap.add_argument("--musicprefs-json", type=Path, required=True)
    ap.add_argument("--music-arena-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mpl.rcParams.update(
        {"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )

    aime = json.loads(args.aime_json.read_text())
    mp = json.loads(args.musicprefs_json.read_text())
    ma = json.loads(args.music_arena_json.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    _scatter_panel(axes[0], aime["per_system"], "AIME (n=13)", aime.get("spearman"))
    _scatter_panel(axes[1], mp["per_system"], "MusicPrefs (n=7)", mp.get("spearman"))
    _scatter_panel(axes[2], ma["per_system"], "Music Arena (n=4 with non-zero rate)", ma.get("spearman"))

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
