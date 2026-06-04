"""Generate the 3-panel Mode 1 BoN N-sweep figure for the TuneJury paper.

Panels (left to right):
  (a) Reward vs N           — TuneJury reward at each N (Top-1 of N)
  (b) CLAP score vs N       — CLAP text-audio cosine at each N
  (c) FAD-CLAP vs N         — Frechet Audio Distance against SDD-706 (CLAP space)

x-axis: N in {1, 2, 4, 8, 16, 32}, log2 spaced
4 colored lines per panel (4 backbones).

Data sources:
  - Reward (N=1..32 from tab:apps_mode1_bon_full Reward column)
  - FAD-CLAP + CLAP-score: per-backbone fad_clap_results.csv files written
    by the Mode 1 BoN scoring pipeline (applications/mode1_bon/) — N=1,2,4,8,16
    in <results-root>/<backbone>_bon100/eval_proper/fad_clap_results.csv,
    N=32 in <results-root>/<backbone>_bon100/eval_n32/fad_clap_results.csv

Usage
-----
$ python figures/make_mode1_bon_figure.py \\
      --results-root /path/to/mode1_bon_results \\
      --out          figures/mode1_bon_sweep.pdf
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BACKBONES = [
    # (display_name, dir_stem, color) — order matches Table 17
    # (tab:apps_mode1_bon_full, canonical reference per §A.G):
    # MusicGen variants together (AR transformers, size ascending), then
    # AudioLDM2-music (latent diffusion), then ACE-Step v1.5 Turbo Continuous
    # (continuous-latent DiT). The ACE-Step label uses the §5 first-mention
    # canonical form including "v1.5" for the standalone legend.
    ("MusicGen-medium",                 "musicgen_medium_bon100", "#3b82f6"),  # blue
    ("MusicGen-large",                  "musicgen_large_bon100",  "#1e3a8a"),  # dark blue
    ("AudioLDM2-music",                 "audioldm2_music_bon100", "#f59e0b"),  # amber
    ("ACE-Step v1.5 Turbo Continuous",  "acestep_bon100",         "#10b981"),  # green
]

# Reward at N=1,2,4,8,16,32 (from appendix Table 9 = tab:bon_saturation_4bb)
REWARD = {
    "MusicGen-medium":                  [+0.3145, +0.5698, +0.8214, +0.9992, +1.1882, +1.3321],
    "MusicGen-large":                   [+0.3853, +0.7189, +0.8403, +1.0534, +1.2500, +1.3695],
    "AudioLDM2-music":                  [-0.8153, -0.4322, -0.0500, +0.2414, +0.3652, +0.4248],
    "ACE-Step v1.5 Turbo Continuous":   [-0.7511, +0.0801, +0.6076, +0.8514, +1.0643, +1.2055],
}
REWARD_N = [1, 2, 4, 8, 16, 32]

# FAD-CLAP + CLAP-text. FAD from fad_clap_results.csv; CLAP score from clap_direct_*.json
# (the latter matches Table 18 / paper-canonical "CLAP score = direct text-audio cosine").
def load_eval(results_root: Path, stem: str) -> dict[int, dict]:
    import json
    f_main = results_root / stem / "eval_proper" / "fad_clap_results.csv"
    f_n32  = results_root / stem / "eval_n32"   / "fad_clap_results.csv"
    out = {}
    for f in (f_main, f_n32):
        with f.open() as fp:
            for row in csv.DictReader(fp):
                n = int(row["N"])
                out[n] = {"fad_clap": float(row["FD-CLAP"])}
    # CLAP-text scores from clap_direct JSONs
    clap_proper = results_root / stem / "clap_direct_proper.json"
    clap_n32    = results_root / stem / "clap_direct_N32.json"
    for f in (clap_proper, clap_n32):
        if f.exists():
            data = json.loads(f.read_text())
            for n_str, v in data.items():
                n = int(n_str)
                if n in out:
                    out[n]["clap_text"] = float(v)
    return out


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--results-root", type=Path, required=True,
                help="root dir containing per-backbone <name>_bon100/eval_{proper,n32}/fad_clap_results.csv")
ap.add_argument("--out", type=Path, required=True,
                help="output PDF path (paper uses figures/mode1_bon_sweep.pdf)")
args = ap.parse_args()

EVAL = {name: load_eval(args.results_root, stem) for name, stem, _color in BACKBONES}
EVAL_N = [1, 2, 4, 8, 16, 32]

# Build figure
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
ax_r, ax_clap, ax_fad = axes

for name, _stem, color in BACKBONES:
    rewards = REWARD[name]   # N=1,2,4,8,16,32
    ax_r.plot(REWARD_N, rewards, "-o", color=color, label=name, markersize=5, linewidth=1.6)

    clap = [EVAL[name][n]["clap_text"] for n in EVAL_N]
    ax_clap.plot(EVAL_N, clap, "-o", color=color, label=name, markersize=5, linewidth=1.6)

    fad = [EVAL[name][n]["fad_clap"] for n in EVAL_N]
    ax_fad.plot(EVAL_N, fad, "-o", color=color, label=name, markersize=5, linewidth=1.6)

# Common x-axis: log2
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(EVAL_N)
    ax.set_xticklabels([str(n) for n in EVAL_N])
    ax.set_xlabel("$N$ (best-of-$N$)")
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)

# Note: y-labels are rotated 90 deg CCW by matplotlib.
# Use $\rightarrow$ to appear as visual UP-arrow after rotation,
# and $\leftarrow$ to appear as visual DOWN-arrow after rotation,
# so the on-plot direction of improvement matches the in-label arrow.
ax_r.set_ylabel("Reward $\\rightarrow$")
ax_r.set_title("(a) TuneJury reward")
ax_clap.set_ylabel("CLAP score $\\rightarrow$")
ax_clap.set_title("(b) CLAP score")
ax_fad.set_ylabel("FAD-CLAP $\\leftarrow$")
ax_fad.set_title("(c) FAD-CLAP against SDD-706")

# Legend across the bottom
handles, labels = ax_r.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.05), frameon=False)

plt.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, bbox_inches="tight", dpi=200)
print(f"wrote {args.out}")
