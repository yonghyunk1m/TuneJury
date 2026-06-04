# Mode 1: Inference-time best-of-$N$ selection

Reproduces Section 5.1 / Figure 3 / Appendix G of the paper.

For each backbone we (i) generate `N` candidates per prompt at the backbone's defaults (only the noise seed differs), (ii) score every candidate with frozen TuneJury, and (iii) keep the top-1. Reward is strictly monotone in `N` on every backbone through `N=32`. The per-doubling gain narrows from `[+0.178, +0.291]` at `N=4→8` to `[+0.060, +0.144]` at `N=16→32`, and AudioLDM2-music saturates earliest with the smallest gain (+0.060 at `N=16→32`) (Appendix G).

## Backbones

| Script                  | Backbone                                       | Conda env  |
|-------------------------|------------------------------------------------|------------|
| `generate_musicgen.py`  | MusicGen-medium / MusicGen-large (audiocraft)  | `musicgen` |
| `generate_audioldm2.py` | AudioLDM2-music (diffusers)                    | `meanaudio`|
| `generate_acestep.py`   | ACE-Step v1.5 Turbo Continuous                 | `ace_step` |
| `generate_sao.py`       | Stable Audio Open-small (stable_audio_tools)   | `sao`      |

The four-backbone sweep in the paper uses MusicGen-medium / MusicGen-large / AudioLDM2-music / ACE-Step Turbo Continuous. SAO-small is used in Mode 2.

## End-to-end on MusicGen-medium

```bash
# Step 1: generate N=32 candidates per prompt
python generate_musicgen.py \
    --prompts ../../eval/prompts/sdd100.json \
    --model facebook/musicgen-medium \
    --out_dir results/musicgen_medium_bon100 \
    --n_candidates 32 \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73

# Step 2: score with TuneJury and select top-1 at N ∈ {1, 2, 4, 8, 16, 32}
python score_and_select.py \
    --prompts ../../eval/prompts/sdd100.json \
    --candidates_dir results/musicgen_medium_bon100 \
    --ckpt ../../checkpoints/tunejury.pt \
    --n_candidates 32 --N 1 2 4 8 16 32 \
    --per_cand_csv results/musicgen_medium_bon100/tunejury_per_candidate.csv \
    --out_csv results/musicgen_medium_bon100/results.csv

# Step 3: aggregate metrics into a single table
python aggregate.py results/musicgen_medium_bon100/results.csv
```

The `results.csv` matches the paper's Figure 3 MusicGen-medium curve and Appendix Table `tab:apps_mode1_bon_full` at every `N ∈ {1, 2, 4, 8, 16, 32}`.

## Prompt set

`../../eval/prompts/sdd100.json` is the 100-item subset of the Song Describer Dataset used in the paper. Each item is prefixed with `"high quality instrumental music, "` to force instrumental output on all four backbones. ACE-Step Turbo Continuous additionally receives an empty lyric input, because it is the only backbone in the sweep with a separate lyric channel.

## MAD (MAUVE Audio Divergence)

To add a third distributional view alongside FAD-CLAP and FAD-MERT (the paper's "distributional metrics disagree across encoders" analysis), compute MAD on MERT-v1-330M embeddings against the SDD-706 reference set:

```bash
python mad_compute.py \
    --bon_results_root /path/to/results \
    --sdd706_dir       /path/to/SDD-706/audio_wav_torch \
    --prompts          ../../eval/prompts/sdd100.json \
    --backbones        musicgen_medium musicgen_large audioldm2_music acestep \
    --out_json         mad_mode1_summary.json
```

Requires `pip install mauve-text`. The script selects N=1 (first seed per prompt) and N=16 (top-1 of 16 by TuneJury score) audio, then runs `mauve.compute_mauve` between each backbone's distribution and the SDD-706 reference distribution.

## MPCD (mode-collapse diagnostic)

To rule out mode collapse as the cause of the `N=32` MAD rise on AudioLDM2-music and ACE-Step Turbo Continuous, compute mean pairwise cosine distance among the 100 top-1 MERT-v1-330M embeddings at each `N`:

```bash
python mpcd_diagnostic.py \
    --backbones musicgen_medium musicgen_large audioldm2_music acestep \
    --results-root results \
    --N 1 8 16 32 \
    --out-csv results/mode1_diversity.csv
```

Paper values (Appendix G, `app:mode1_diversity`): MPCD rises substantially from `N=1` to `N=32` on AudioLDM2-music (`+39%`) and ACE-Step Turbo Continuous (`+66%`), while the two MusicGen variants stay flat (`±10%` of `N=1`). The MAD rise on AudioLDM2-music and ACE-Step Turbo Continuous therefore reflects distributional drift away from SDD-706 rather than narrowing diversity.
