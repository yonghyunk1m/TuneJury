# Probes (Appendix analyses)

Backbone-agnostic, feature-cache-only probes that quantitatively
support the Section 6 self-diagnostic and limitations claims of the
paper. The three pairwise probes operate on cached LAION-CLAP + MERT
features and never call the audio encoders at probe time; the
soundfont probe (Appendix I) does call the encoders because it
generates fresh WAVs from MIDI.

## Scripts

| Script | Paper anchor | Headline |
| --- | --- | --- |
| `per_system_ranking.py` | Appendix F (Per-System Reward Ranking on Held-Out Test Splits) | AIME test (n=1,560, 13 systems): Spearman $\rho=+0.978$. MusicPrefs test (n=252, 7 systems): $\rho=+0.964$. Music Arena test (n=74) too small. |
| `vocal_discrimination.py` | Appendix F (Vocal-vs-instrumental discrimination probe) | Music Arena (n=6,120 clips): vocal-requested $+0.977$ vs. instrumental $+0.536$, gap $+0.441$ (Welch $t=+24.2$, $p<0.001$). Population-level signal; not per-clip vocal skill. |
| `popularity_probe.py` | Appendix F (Amateur-vs-professional probe) | FMA-Large (106K tracks): bottom listens decile $\to$ $-1.418$ reward, top decile $\to$ $+0.081$, gap $\sim 1.50$ reward units. Spearman(log-listens, reward) $=+0.285$, $p<0.001$. |
| `soundfont_sensitivity.py` | Appendix I (MidiCaps soundfont sensitivity) | First 300 MidiCaps tracks re-rendered under FluidR3\_GM (142 MB) vs.\ TimGM6mb (5.7 MB): aggregate means are close (paired $t=+1.73$, $p\approx 0.085$), but track-level rankings are noisy (cross-soundfont Spearman $+0.69$; 33% top-10 overlap). Distribution-level comparisons safe; track-level rankings should be treated as (score, renderer) joint quantities. |
| `singmos_validation.py` | Appendix F (External singing-voice MOS validation) | SingMOS-Pro (Tang et al. 2025, arXiv:2510.01812): per-utterance Spearman $+0.19$ and per-system Spearman $+0.44$ on $n{=}7{,}981$ utterances across $141$ singing-voice generation systems (SVS / SVR / SVC / GT, Chinese and Japanese). |
| `svcc_validation.py` | Appendix F (External singing-voice MOS validation) | SVCC 2025 (Violeta et al. 2025, arXiv:2509.15629): $n{=}48$ real-human recordings, $2$ singers $\times$ $6$ vocal techniques; ANOVA across techniques $F{=}3.82$, $p<0.01$; Mixed Voice highest mean reward ($-0.11$), Pharyngeal lowest ($-0.70$). |

> **Paper-exact reproduction default.** Running `python applications/probes/svcc_validation.py` (no args) loads the committed `results/svcc2025/summary.json` and recomputes the analysis (ANOVA + Welch t) deterministically from the stored per-clip TuneJury scores — paper numbers reproduce **exactly** every run. Same default behavior for `singmos_validation.py`: it reports the canonical per-utterance and per-system Spearman directly from the committed `results/singmos_pro/summary.json`.
>
> **Optional fresh re-inference.** Pass `--reinfer --svcc_root <path>` (SVCC) or run the 4-shard inference followed by `--aggregate` (SingMOS) to regenerate from raw audio. Fresh re-inference may show small variance vs the committed scores: the transformer encoders (LAION-CLAP, MERT) use bf16 attention reductions whose order is not fully deterministic across GPUs, so per-technique SVCC means can drift by up to $\pm 0.02$ and ANOVA $F$ by $\pm 0.3$. Statistical significance ($p < 0.01$ for ANOVA, $p < 0.005$ for Real-vs-SVC Welch t) is robust across observed runs, and SingMOS-Pro Spearman correlations averaged over $n{=}7{,}981$ utterances stay within $\pm 0.005$ SRCC.

## Inputs

### In-repo (always available)

- `data/processed_features/<uuid>.pt`: 24,935 paired feature files
  carrying `clap_a`, `mert_a`, `clap_b`, `mert_b`, `text_emb`,
  `winner`, `source` (one of MusicArena, AIME, MusicPrefs, SongEval).
- `data/splits/*.json` and `data/splits/test_*.txt`: held-out
  test-fold UUID lists per dataset.
- `release_scores/fma_large_scores.csv`: 106K pre-computed per-track
  TuneJury rewards on FMA-Large (used as the fast path by
  `popularity_probe.py`).
- `checkpoints/tunejury.pt`: released head checkpoint.

### External prerequisites (only needed by certain probes)

The in-repo feature cache carries everything the head needs to score,
but several Appendix F aggregations require **per-pair metadata** that
is not in the .pt files (system labels, requested-lyrics text,
play-count statistics). The probes accept those as optional JSON
sidecars; reviewers who reproduce from raw must supply them:

| Probe | Required metadata | Where to get it |
| --- | --- | --- |
| `per_system_ranking.py` | `model_a`, `model_b` per pair | The original Music Arena / AIME / MusicPrefs preference dumps that fed `data/extract_features.py`. The companion [`music-ranknet`](https://github.com/yonghyunk1m/music-ranknet) preprocessing repo emits these as `<DS>/json/<uuid>.json`. |
| `vocal_discrimination.py` | `lyrics` per pair | The same Music Arena preference dump. |
| `popularity_probe.py` | `track_id,listens` CSV (FMA-Large play counts) | Derived from FMA-Large's `fma_metadata/tracks.csv` column `track_listens`. See FMA-Large download at https://github.com/mdeff/fma . |
| `soundfont_sensitivity.py` | First 300 MidiCaps `.mid` files; FluidR3\_GM.sf2 (apt `fluid-soundfont-gm`); TimGM6mb.sf2 (ships with `pretty_midi`). | https://huggingface.co/datasets/amaai-lab/MidiCaps |
| `singmos_validation.py` | SingMOS-Pro dataset (`metadata.json` + `wav/*.wav`) | `git clone https://huggingface.co/datasets/TangRain/SingMOS-Pro` (~5 GB) |
| `svcc_validation.py` | SVCC 2025 dataset (`test_gt/` + `test/` directories) | `huggingface-cli download lestervioleta/svcc2025 --repo-type dataset --local-dir <path>` (~11 GB tar.gz auto-extracts to test_gt + test) |

## Reproducing

```bash
# Per-system ranking (Appendix F): needs system labels.
python applications/probes/per_system_ranking.py \
    --features_dir data/processed_features \
    --splits_dir   data/splits \
    --meta_root    /path/to/music-ranknet/data/processed \
    --checkpoint   checkpoints/tunejury.pt \
    --out_dir      applications/probes/results/per_system

# Vocal vs instrumental (Appendix F): needs Music Arena lyrics field.
python applications/probes/vocal_discrimination.py \
    --features_dir data/processed_features \
    --meta_root    /path/to/music-ranknet/data/processed \
    --checkpoint   checkpoints/tunejury.pt \
    --out_dir      applications/probes/results/vocal_probe

# Popularity probe (Appendix F): fast path using release_scores CSV.
python applications/probes/popularity_probe.py \
    --scores_csv  release_scores/fma_large_scores.csv \
    --listens_csv /path/to/fma_listens.csv \
    --out_dir     applications/probes/results/popularity_probe

# Soundfont sensitivity (Appendix I): renders + scores end-to-end.
python applications/probes/soundfont_sensitivity.py \
    --midi_dir   /path/to/midicaps_midi_first_300 \
    --sf_fluidr3 /usr/share/sounds/sf2/FluidR3_GM.sf2 \
    --sf_timgm6  $(python -c "import pretty_midi, os; print(os.path.join(os.path.dirname(pretty_midi.__file__), 'TimGM6mb.sf2'))") \
    --checkpoint checkpoints/tunejury.pt \
    --out_root   applications/probes/results/soundfont_probe

# External SingMOS-Pro validation (Appendix F):
# (a) 4-way shard inference (one shard per GPU; sequential or parallel)
for s in 0 1 2 3; do
  python applications/probes/singmos_validation.py \
      --checkpoint   checkpoints/tunejury.pt \
      --singmos_root /path/to/SingMOS-Pro \
      --device       cuda:$s \
      --shard $s --n_shards 4 \
      --out_dir      applications/probes/results/singmos_pro
done
# (b) Aggregate shards into summary.json
python applications/probes/singmos_validation.py \
    --aggregate --out_dir applications/probes/results/singmos_pro

# External SVCC 2025 validation (Appendix F): vocal-technique ANOVA on real recordings.
python applications/probes/svcc_validation.py \
    --checkpoint checkpoints/tunejury.pt \
    --svcc_root  /path/to/svcc2025 \
    --out_dir    applications/probes/results/svcc2025
```

Each probe writes a `summary.json` to `--out_dir` and prints the
headline numbers to stdout.
