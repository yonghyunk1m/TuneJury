"""TuneJury architecture diagram (Figure 1).

Paper anchor: Figure ``tunejury_arch`` in §3.

Renders the architecture diagram showing the 2,048-d concatenated input
[CLAP audio (512) + MERT audio (1024) + CLAP text (512)] feeding a
4-layer MLP head with the pairwise logistic loss
``L = -log sigma(s(A) - s(B))``.

This script is documentation-grade; the figure in the paper is a
hand-tuned TikZ rendering at ``neurips2026-paper/figures/tunejury_arch.pdf``.
For a quick reproduction, run this script and inspect the output.

Usage
-----
$ python figures/make_tunejury_arch.py --out figures/tunejury_arch.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, label, color, text_color="black"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.8,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=text_color, fontsize=8)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=0.8,
            color="black",
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # Frozen encoders (left)
    _box(ax, 0.2, 3.2, 1.6, 0.6, "Audio (10s, 16 kHz)", "#e5e7eb")
    _box(ax, 0.2, 2.0, 1.6, 0.6, "Audio (10s, 24 kHz)", "#e5e7eb")
    _box(ax, 0.2, 0.8, 1.6, 0.6, "Text prompt", "#e5e7eb")

    _box(ax, 2.2, 3.2, 1.8, 0.6, "LAION-CLAP audio", "#cbd5e1", text_color="black")
    _box(ax, 2.2, 2.0, 1.8, 0.6, "MERT-v1-330M (audio)", "#cbd5e1")
    _box(ax, 2.2, 0.8, 1.8, 0.6, "LAION-CLAP text", "#cbd5e1")

    _box(ax, 4.4, 3.2, 0.6, 0.6, "512", "#94a3b8")
    _box(ax, 4.4, 2.0, 0.6, 0.6, "1024", "#94a3b8")
    _box(ax, 4.4, 0.8, 0.6, 0.6, "512", "#94a3b8")

    # Concat
    _box(ax, 5.4, 1.8, 1.0, 1.2, "concat\n[2048]", "#fbbf24")

    # MLP head (trainable)
    _box(ax, 6.8, 1.8, 1.6, 1.2, "MLP head\n[1024,512,256,128]\n~2.8M params", "#f87171")

    # Output
    _box(ax, 8.8, 2.2, 1.0, 0.4, "scalar s(x)", "#10b981", text_color="white")

    # Loss
    _box(ax, 7.4, 0.2, 2.4, 0.4, "L = -log σ(s(A) - s(B))", "#a3e635")

    # Arrows
    for y in [3.5, 2.3, 1.1]:
        _arrow(ax, 1.8, y, 2.2, y)
        _arrow(ax, 4.0, y, 4.4, y)
        _arrow(ax, 5.0, y, 5.4, y - (y - 2.4) * 0.3)
    _arrow(ax, 6.4, 2.4, 6.8, 2.4)
    _arrow(ax, 8.4, 2.4, 8.8, 2.4)
    _arrow(ax, 9.3, 2.2, 8.6, 0.6)

    ax.set_title("TuneJury (CLAP+MERT instantiation): frozen encoders + 2.8M-param MLP head", fontsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
