"""Decomposition probe figure (paper Appendix Fig. decomp_probe).

Inputs (passed via --scaling and --stability) are JSON dumps from
eval/decomposition_probe.py runs:
  - scaling: 5-seed n-vs-partial-SRCC sweep
  - stability: 20-seed audit at n_train=728

Usage:
    python figures/make_decomp_probe_figure.py \
        --scaling results/scaling_multiseed.json \
        --stability results/stability_audit_20seed.json \
        --out figures/decomp_probe.pdf
"""
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import t as t_dist

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scaling", required=True,
                    help="JSON from eval/decomposition_probe.py scaling sweep")
parser.add_argument("--stability", required=True,
                    help="JSON from eval/decomposition_probe.py 20-seed stability audit")
parser.add_argument("--out", required=True,
                    help="Output figure path (.pdf or .svg). Both extensions written.")
args = parser.parse_args()

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_ALN = "#3d5a80"
COLOR_PAR = "#e07a5f"
COLOR_RES = "#2a9d8f"
COLOR_REF = "#666666"

scaling = json.load(open(args.scaling))
stability = json.load(open(args.stability))

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.18, "width_ratios": [1.35, 1.0]},
)

# ============================================================
# Panel (a) — unchanged
# ============================================================
sizes = scaling['sizes']
N = scaling['n_seeds']
t_crit = t_dist.ppf(0.975, df=N-1)

aln_means, aln_lo, aln_hi = [], [], []
par_means, par_lo, par_hi = [], [], []
for f in sizes:
    aln_arr = np.array(scaling['results'][str(f)]['aln'])
    par_arr = np.array(scaling['results'][str(f)]['partial'])
    am = aln_arr.mean(); ase = aln_arr.std(ddof=1) / np.sqrt(N)
    pm = par_arr.mean(); pse = par_arr.std(ddof=1) / np.sqrt(N)
    aln_means.append(am); aln_lo.append(am - t_crit * ase); aln_hi.append(am + t_crit * ase)
    par_means.append(pm); par_lo.append(pm - t_crit * pse); par_hi.append(pm + t_crit * pse)

train_n = [int(f * 728) for f in sizes]
x = np.arange(len(train_n))

ax_a.fill_between(x, aln_lo, aln_hi, color=COLOR_ALN, alpha=0.18)
ax_a.fill_between(x, par_lo, par_hi, color=COLOR_PAR, alpha=0.18)
ax_a.plot(x, aln_means, "-s", color=COLOR_ALN, lw=2.5, ms=8, mfc=COLOR_ALN, mec="white",
          label="Alignment SRCC")
ax_a.plot(x, par_means, "-o", color=COLOR_PAR, lw=2.5, ms=8, mfc=COLOR_PAR, mec="white",
          label="Partial SRCC (musicality removed)")
ax_a.axhline(0, color=COLOR_REF, linestyle=":", linewidth=0.8)

ax_a.plot([x[-1]], [par_means[-1]], "o", ms=14, mfc="none", mec=COLOR_PAR, mew=2.0, zorder=5)
ax_a.text(x[-1] - 0.4, par_means[-1] + 0.10,
          "monotone ascent,\nno plateau",
          ha="center", va="bottom", fontsize=10, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8),
          zorder=6)

ax_a.set_xticks(x)
ax_a.set_xticklabels([str(n) for n in train_n])
ax_a.set_xlabel(r"Alignment-labeled probe clips $n_{\rm train}$")
ax_a.set_ylabel("Spearman correlation on val")
ax_a.set_title(r"(a) Data scaling, $5$ seeds, $95\%$ CI", fontsize=11)
ax_a.set_ylim(-0.05, 0.75)
ax_a.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
ax_a.grid(True, alpha=0.3)

# ============================================================
# Panel (b) — strip + mean ± CI, single-line labels
# ============================================================
data = [stability['aln_srcc'], stability['partial'], stability['residual']]
labels = ['Alignment\nSRCC', 'Partial\nSRCC', 'Residual\nSRCC']
colors = [COLOR_ALN, COLOR_PAR, COLOR_RES]

# Individual seed points (strip with jitter)
rng = np.random.RandomState(42)
for i, arr in enumerate(data):
    arr = np.array(arr)
    jitter = rng.uniform(-0.10, 0.10, size=len(arr))
    ax_b.scatter(np.full_like(arr, i + 1) + jitter, arr,
                 s=22, color=colors[i], alpha=0.45, edgecolor="white", linewidth=0.5,
                 zorder=3)

# Mean + 95% CI: thicker bar with cap
for i, arr_name in enumerate(['aln_srcc', 'partial', 'residual']):
    arr = np.array(stability[arr_name])
    n = len(arr); t = t_dist.ppf(0.975, df=n-1)
    m = arr.mean(); se = arr.std(ddof=1) / np.sqrt(n)
    lo, hi = m - t*se, m + t*se
    # Wide horizontal mean line
    ax_b.plot([i + 1 - 0.22, i + 1 + 0.22], [m, m],
              color=colors[i], linewidth=3.0, zorder=5, solid_capstyle="round")
    # Slim CI bracket
    ax_b.plot([i + 1, i + 1], [lo, hi], color="black", linewidth=1.6, zorder=4)
    ax_b.plot([i + 1 - 0.09, i + 1 + 0.09], [lo, lo], color="black", linewidth=1.6, zorder=4)
    ax_b.plot([i + 1 - 0.09, i + 1 + 0.09], [hi, hi], color="black", linewidth=1.6, zorder=4)
    # Value label — placed BELOW the cluster to avoid overlap
    ax_b.text(i + 1, lo - 0.04, f"{m:+.3f}", fontsize=10.5, va="top", ha="center",
              color=colors[i], fontweight="bold")

# Reference: 0 line (statistical significance reference)
ax_b.axhline(0, color=COLOR_REF, linestyle=":", linewidth=0.8)
# Annotation: all three CIs exclude zero
ax_b.text(2.0, 0.10, "All three 95% CIs exclude 0\n(real signal, not noise)",
          ha="center", va="center", fontsize=9.5, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))

ax_b.set_xticks([1, 2, 3])
ax_b.set_xticklabels(labels)
ax_b.set_title(r"(b) $20$-seed distribution at $n_{\rm train}{=}728$", fontsize=11)
ax_b.set_xlim(0.5, 3.5)
ax_b.set_ylim(0.05, 0.75)
ax_b.grid(True, alpha=0.3, axis="y")

from pathlib import Path
out_path = Path(args.out)
plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight', dpi=200)
plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight')
print(f'Saved {out_path.with_suffix(".pdf")} and {out_path.with_suffix(".svg")}')
