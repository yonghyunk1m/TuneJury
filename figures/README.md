# Paper figure scripts

Every quantitative figure in the paper has a source script here.

| Figure | Script | Inputs |
|---|---|---|
| Figure 2 (architecture, §3) | `make_tunejury_arch.py` | none (diagram-only) |
| Figure 3 (Mode 1 BoN sweep, §5.1) | `make_mode1_bon_figure.py` | per-backbone `eval_proper/fad_clap_results.csv` + `eval_n32/fad_clap_results.csv` (reads inline `REWARD` table from paper §A.G Table `apps_mode1_bon_full`) |
| Calibration reliability (§A.A) | `make_calibration_figure.py` | `eval/internal.py --out-bins-csv` |
| OOD scaling + cross-month (§A.D) | `make_ood_scaling_figure.py` | `applications/anchor_calibration/run_experiment.py` + `cross_month_eval.py` CSV outputs |
| Per-system reward ranking (§A.F) | `make_per_system_figure.py` | `applications/probes/per_system_ranking.py` per-dataset JSON outputs |
| Vocal + popularity probes (§A.F) | `make_probes_figure.py` | `applications/probes/vocal_discrimination.py` + `popularity_probe.py` JSON outputs |
| Release distribution (§A.I) | `make_release_distribution_figure.py` | one CSV per dataset with a `reward` column |

Figure 1 (modes pipeline) is a TikZ diagram embedded directly in
`sections/05_applications.tex` and is not reproduced here.

All scripts read inputs that are emitted by the upstream eval/probe
steps documented in `docs/reproducing.md`. End-to-end: run the eval
step, then the figure script.

```bash
# Example: Figure 3 (Mode 1 BoN sweep)
cd applications/mode1_bon  # see its README for the full sweep
# ... runs scoring + FAD-CLAP/MERT + CLAP-score for N=1..32
cd ../..
python figures/make_mode1_bon_figure.py
# Writes figures/mode1_bon_sweep.pdf
```

All figure scripts use matplotlib (`pip install matplotlib`) and no
other plotting libraries.
