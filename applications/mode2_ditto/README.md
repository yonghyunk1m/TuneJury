# Mode 2: Inference-time latent optimization (DITTO-style)

Reproduces Section 5.2 of the paper (Table 5 top panel).

We route the frozen TuneJury reward back into the sampler in DITTO style on two backbones. For both SAO-small and TangoFlux we run full 8-step rectified-flow sampler backprop with Adam on the noise tensor (5 outer iterations, lr 0.05) against the negative reward. Base weights stay frozen. AudioLDM2-music is omitted: its 50-step denoising trajectory exceeds the full-backprop memory budget on our hardware, and partial late-stage backprop on only the final few steps produced perceptually muted reward lifts.

The asymmetric prompt counts (SAO-small n=30, TangoFlux n=100) reflect a reproducibility constraint of the SAO release at the time of writing (see paper Appendix `app:sao_caveat`).

| Model | Reward | MAD | CLAP score | Win |
|---|---|---|---|---|
| SAO-small (340 M, n=30) baseline | $+0.159$ | $1.070$ | $0.1961$ | -- |
| &nbsp;&nbsp;+ DITTO | $+0.404$ ($+0.245$) | $1.570$ ($+0.500$) | $0.1886$ ($-0.007$) | 19/30 |
| TangoFlux (515 M, n=100) baseline | $-0.978$ | $4.263$ | $0.1501$ | -- |
| &nbsp;&nbsp;+ DITTO | $+0.578$ ($+1.557$) | $2.048$ ($-2.214$) | $0.1933$ ($+0.043$) | 100/100 |

Parenthesized values are the change from the baseline, computed before rounding (matching paper Table 5).

MAD is reported as $-\ln(\mathrm{MAUVE})$ against the SDD-706 reference distribution (Huang et al., 2025); range $[0,\infty)$, lower is closer to the reference. DITTO lifts mean TuneJury reward on both backbones (${+}0.245$ on SAO-small, ${+}1.557$ on TangoFlux), and the lift is larger on the backbone whose baseline reward is lower (TangoFlux from ${-}0.978$ vs.\ SAO-small from ${+}0.159$). The two side metrics split per backbone. On TangoFlux MAD against SDD-706 drops sharply (${-}2.214$) and the CLAP score rises (${+}0.043$): DITTO rescues a low-reward backbone toward audio that is closer to SDD-706 and better aligned with the text prompt, a win-win pattern with no visible reward exploitation. On SAO-small both side metrics regress (MAD ${+}0.500$, CLAP ${-}0.007$): the backbone already produces audio at near-zero reward at baseline (${+}0.159$), and DITTO's TuneJury reward gain (${+}0.245$) comes at the cost of distribution and alignment drift, the classic three-axis reward-exploitation pattern. The explicit Pareto trade-off in Mode 3 (Section 5.3) is a more controlled demonstration of the reward-fidelity tension across a learning-rate sweep.

## SAO-small

```bash
conda activate sao
python sao_small.py \
    --prompts ../../eval/prompts/sdd100.json \
    --checkpoint ../../checkpoints/tunejury.pt \
    --out_dir results/ditto_sao_small \
    --limit 30
```

## TangoFlux

```bash
conda activate tangoflux
python tangoflux_ditto.py \
    --prompts ../../eval/prompts/sdd100.json \
    --checkpoint ../../checkpoints/tunejury.pt \
    --out_dir results/ditto_tangoflux \
    --limit 100
```

## Delta-CLAP recompute

Both backbones write `{pid}_baseline.wav` and `{pid}_ditto.wav` pairs plus a `summary.json` directly in `--out_dir`:

```bash
python compute_clap_delta.py \
    --results_dir  results/ditto_sao_small \
    --json         results/ditto_sao_small/summary.json \
    --prompts_json ../../eval/prompts/sdd100.json \
    --clap_ckpt    ../../checkpoints/music_audioset_epoch_15_esc_90.14.pt \
    --limit 30
```

Swap `results/ditto_sao_small` for `results/ditto_tangoflux` and drop `--limit` (or set `--limit 100`) for the TangoFlux pass.

## MAD recompute

MAD = $-\ln(\mathrm{MAUVE})$ against the SDD-706 reference set. Requires `pip install mauve-text` (not bundled in `environment.yml`).

`mad_compute.py` globs `*.wav` inside `--baseline_dir` and `--after_ft_dir`, so first split the flat output into two sibling directories (one holding the `*_baseline.wav` files, the other the `*_ditto.wav` files):

```bash
mkdir -p results/ditto_sao_small/baseline results/ditto_sao_small/ditto
mv results/ditto_sao_small/*_baseline.wav results/ditto_sao_small/baseline/
mv results/ditto_sao_small/*_ditto.wav    results/ditto_sao_small/ditto/

python ../mode3_expert_iter/mad_compute.py \
    --baseline_dir results/ditto_sao_small/baseline \
    --after_ft_dir results/ditto_sao_small/ditto \
    --sdd706_dir   /path/to/SDD-706/audio_wav_torch \
    --out_json     mad_sao_small.json
```

The same split then MAD pattern works for TangoFlux against `results/ditto_tangoflux/`.

## Notes

- **Differentiable scorer.** DITTO needs a differentiable path through the encoders. `DifferentiableScorer` keeps MERT on-graph while detaching LAION-CLAP audio (numpy interop). The CLAP text branch is cached per prompt.
- **GPU memory.** Full 8-step sampler backprop fits on a single 24 GB GPU for both SAO-small and TangoFlux. AudioLDM2-music full 50-step UNet backprop does not, and partial late-stage backprop (final 5 of 50 steps) produced perceptually muted reward lifts. We therefore omit AudioLDM2-music from the headline Mode 2 result rather than reporting a partial-backprop variant.
