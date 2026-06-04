# Anchor-Calibrated TuneJury

Lightweight per-system bias correction for deploying TuneJury under generator drift.
Reproduces the post-cutoff recovery experiment in Appendix D of the paper
(Figure `fig:ood_scaling`, Table `tab:ood_scaling`).

## Why

Released TuneJury reaches `0.7194` pairwise agreement on the in-distribution
CMI-RewardBench Music Arena split, but drops to `0.5369` on the 2026-02 / 2026-03
post-cutoff Music Arena slice. Diagnosis (Appendix D, `app:postcut_decomposition`):

- 4 of 11 post-cutoff systems (ACE-Step v1.5 Turbo Continuous, Lyria 3-30s,
  Lyria 3 Pro preview, Sonauto v3 preview) appear 0 times in training.
- MusicGen-medium sits 0.18 cosine units further from the in-distribution
  CLAP centroid than the median trained system (MERT cosines are uniform
  across systems, isolating CLAP as the drifting encoder).
- Confidence calibration is only partially informative on the post-cutoff slice
  (51.3% at `|Δ|<0.5` to 62.0% at `|Δ|>1.2`, a +10.7 pp gradient but well below
  the 0.7086 in-distribution test accuracy).

## Method

Freeze the released TuneJury. Treat its score `r(x)` as the offset in a
Bradley–Terry model with a single per-system scalar bias `β_s`:

    P(a wins) = sigmoid( (r(a) - β_{system(a)}) - (r(b) - β_{system(b)}) )

Fit `{β_s}` by ℓ₂-regularized maximum-likelihood logistic regression on `K`
anchor preference pairs from the new distribution (L-BFGS, λ=1.0, < 1 s of CPU).
One in-distribution system is held at `β=0` for identifiability, anchoring
the score scale to its training-time meaning, analogous to anchor-item
calibration in item-response theory (Wright & Stone, *Best Test Design*,
1979), where common anchor items link calibrations across separate test
forms.

## Result on the post-cutoff slice (5 seeds, n_test = 299)

| K | Retraining (R) | Anchor calibration (A) | Gap A − R |
|---:|---:|---:|---:|
| 0   | **55.6 ± 1.7** | 55.1 ± 2.0 | −0.5 |
| 3   | 54.8 ± 2.0 | **56.6 ± 2.5** | +1.8 |
| 10  | 54.1 ± 2.3 | **57.6 ± 3.2** | +3.5 |
| 30  | 56.1 ± 2.4 | **57.7 ± 2.7** | +1.6 |
| 100 | 55.6 ± 2.0 | **60.1 ± 3.3** | +4.5 |
| 250 | 57.0 ± 3.5 | **61.2 ± 1.0** | +4.2 |

Anchor calibration reaches retraining's K=250 accuracy at K=10: ~25× data
efficiency in the few-shot regime, with tighter seed variance throughout.

## Files

- `fit.py`: Bradley–Terry per-system β estimation (sklearn-free, L-BFGS).
- `run_experiment.py`: anchor-calibration (A) reproducer of the paper's K-sweep
  (Figure `fig:ood_scaling`, Table `tab:ood_scaling`) against the pre-extracted
  Music Arena v2 increment features.
- `retrain_ksweep.py`: naive-retraining (R) reproducer of the paper's K-sweep
  (Table `tab:ood_scaling`, R row), trained from scratch on bench-clean (571)
  ∪ K added post-cutoff pairs.
- `postcut_diagnostics.py`: three-way failure decomposition (per-system
  asymmetry binomial test; CLAP/MERT cosine to in-distribution centroid;
  |Δ_TJ| calibration bins). Backs Appendix D §"Three diagnostics localize
  the gap" (`app:postcut_decomposition`).
- `cross_month_eval.py`: Feb-Mar anchor fit → April held-out eval, plus
  within-April 50/50 sanity probe (Table `tab:cross_month`, Figure `fig:ood_scaling` Panel B).
- `retrain_forgetting.py`: naive-retraining baseline and the
  "no catastrophic forgetting" probe (four training conditions × three
  test partitions).
- `intrinsic_difficulty.py`: |Δ_TJ| distribution comparison across
  bench-clean / Feb-Mar / April plus April human-side TIE/BOTH_BAD
  fractions (Table `tab:intrinsic_difficulty`). Backs Appendix D
  §"Post-cutoff battles are intrinsically harder".
- `heldout_margin.py`: |Δ_TJ| distribution on the full non-tie 4-dataset
  held-out test (n=2,035). This is the in-distribution baseline row in
  Table `tab:intrinsic_difficulty`; replaces the n=20 bench-clean Music Arena
  decisive subset used in an earlier draft, which was too small to support the
  intrinsic-difficulty comparison robustly.

All scripts share the same feature-cache and split-file layout from
the music-ranknet preprocessing pipeline; see each script header for
exact path conventions.

## Usage (release)

```python
from tunejury.score import Scorer
from applications.anchor_calibration.fit import AnchorCalibrator

scorer = Scorer.from_pretrained("checkpoints/tunejury.pt")
calibrator = AnchorCalibrator(l2=1.0)

# 1) Score each anchor pair's two audios with the frozen TuneJury.
# 2) Pass tuples (score_a, score_b, system_a, system_b, y) where
#    y = 1 if human voters preferred A, else 0 (ties excluded).
# ~100 pairs per newly deployed system recovers ~5 pp at K=100; K=30 ~+2.5 pp.
anchor_records = []
for audio_a, audio_b, prompt, sys_a, sys_b, y in anchor_pairs:
    score_a = scorer.score(audio_a, prompt)
    score_b = scorer.score(audio_b, prompt)
    anchor_records.append((score_a, score_b, sys_a, sys_b, y))

# Pin an in-distribution generator at beta=0 for identifiability (e.g.,
# Sonauto v2 in the paper's Feb-Mar fits). Other systems' biases are
# fit relative to this anchor. Pass the system tag exactly as it appears
# in the system_a / system_b fields of anchor_records.
calibrator.fit(anchor_records, anchor_system="<in-distribution anchor tag>")

# At inference, score both candidates then request the calibrated pairwise prob:
score_a = scorer.score(audio_a, prompt)
score_b = scorer.score(audio_b, prompt)
p_a_wins = calibrator.predict_pairwise(score_a, score_b, system_a, system_b)
```

## Limitations

- Slice supports K ≤ 250 (598 universe size, 50/50 train/test). Whether either
  curve eventually closes the full gap to the in-distribution ceiling requires
  a larger increment.
- Per-system β is an additive intercept; multiplicative scale or higher-order
  drift is not modeled.
- The released checkpoint and encoders are not modified; pure post-hoc
  correction.
