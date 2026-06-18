# Reproducing TuneJury

This document maps every concrete result in the paper (numbers, tables, figures, ablations, probes) to a runnable command in this repository. Sub-READMEs in each `applications/*/` directory carry the operational detail; this file is the single entry point.

## Table of contents

- [0. Prerequisites (stop and read this first)](#0-prerequisites-stop-and-read-this-first)
- [1. Setup](#1-setup)
- [2. Data preparation](#2-data-preparation)
  - [2.5 SDD-706 reference (Mode 1/2/3 prerequisite)](#25-sdd-706-reference-mode-1--mode-2--mode-3-prerequisite)
  - [2.6 Post-cutoff Music Arena slice (Anchor calibration prerequisite)](#26-post-cutoff-music-arena-slice-anchor-calibration-prerequisite)
- [3. Training](#3-training)
  - [3.1 Released checkpoint](#31-released-checkpoint)
  - [3.2 Leave-one-dataset-out ablations](#32-leave-one-dataset-out-ablations)
  - [3.3 Input ablation (Appendix C)](#33-input-ablation-appendix-c)
  - [3.4 MuQ-MuLan encoder swap](#34-muq-mulan-encoder-swap)
- [4. Internal evaluation](#4-internal-evaluation)
  - [4.1 Pairwise accuracy + ECE (§4.1)](#41-pairwise-accuracy--ece-§41)
  - [4.2 Calibration bins + reliability diagram (Appendix A)](#42-calibration-bins--reliability-diagram-appendix-a)
  - [4.3 Per-dataset leave-out (Table `internal_per_dataset`)](#43-per-dataset-leave-out)
- [5. External evaluation (CMI-RewardBench)](#5-external-evaluation-cmi-rewardbench)
  - [5.0 CMI-RewardBench preparation (prerequisite)](#50-cmi-rewardbench-preparation-prerequisite)
  - [5.1 Head-to-head (Table `head_to_head`)](#51-head-to-head-table-head_to_head)
  - [5.2 Training-mix design space (Table `training_mix`)](#52-training-mix-design-space)
  - [5.3 Encoder swap (Table `encoder_swap`)](#53-encoder-swap)
  - [5.4 AIME held-out per-axis (Table `aime_per_axis`)](#54-aime-held-out-per-axis)
  - [5.5 MuQ-Eval head-to-head baseline (Table `head_to_head` row)](#55-muq-eval-head-to-head-baseline)
- [6. Adversarial sanity checks](#6-adversarial-sanity-checks)
- [7. Mode 1: best-of-N selection (§5.1 + Appendix G)](#7-mode-1-best-of-n-selection-§51--appendix-g)
- [8. Mode 2: DITTO-style latent optimization (§5.2)](#8-mode-2-ditto-style-latent-optimization-§52)
- [9. Mode 3: expert-iteration post-training (§5.3 + Appendix H)](#9-mode-3-expert-iteration-post-training-§53--appendix-h)
- [10. Anchor calibration (§6 + Appendix D)](#10-anchor-calibration-§6--appendix-d)
- [11. Behavior probes (Appendix F)](#11-behavior-probes-appendix-f)
- [12. Figures](#12-figures)
- [13. Backbone environments](#13-backbone-environments)

---

## 0. Prerequisites (stop and read this first)

A fresh clone of this repository is **not self-contained**. Before running any of the `python` commands in §1-§13 below, please work through this checklist. The single most common reviewer failure mode is running `tunejury.train` or one of the Mode 1/2/3 scripts against an empty `data/processed_features/` or a missing backbone checkpoint, and then reading the resulting `FileNotFoundError` as a repo bug.

### 0.1 System dependencies

```bash
# Debian / Ubuntu
sudo apt-get install -y ffmpeg libsndfile1 git-lfs
git lfs install
```

`ffmpeg` is required by `torchaudio` and by every Mode 1/2/3 generation script. `libsndfile1` is required by `soundfile`. `git-lfs` is **only** needed if you want to clone the released checkpoint blobs from the GitHub mirror; the canonical download is the Hugging Face Hub copy (see §0.4) and does not require git-lfs.

### 0.2 Conda environments

The TuneJury head, internal eval (§4), CMI-RewardBench eval (§5), anchor calibration (§10), and all probes (§11) run in a single environment:

```bash
conda env create -f environment.yml   # creates env "tunejury"
conda activate tunejury
```

Each Mode 1/2/3 backbone needs its **own** conda environment (their CUDA / `torch` / `diffusers` / `transformers` pins conflict with each other and with `tunejury`). See §0.7 below and §13 for the install commands.

**Container alternative.** A `Dockerfile` at the repo root builds the main `tunejury` environment on top of `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04`. It covers §3-§5, §10, §11 but not the Mode 1/2/3 backbones (which need separate envs per §0.7). Build with `docker build -t tunejury:0.1 .`; run with `docker run --rm -it --gpus '"device=N"' -v $(pwd):/workspace -w /workspace tunejury:0.1 bash`.

### 0.3 Hugging Face login (gated models)

MERT-v1-330M is a **gated** model on Hugging Face. Without a token, the first call into `Scorer.score(...)` will fail with a 401. Request access on the model card, then:

```bash
huggingface-cli login   # paste a read-token from https://huggingface.co/settings/tokens
```

The same token also unlocks MuQ-MuLan-large (encoder-swap probe, §3.4 / §5.3) and the Music Arena / SongEval dataset configs used in §2.

### 0.4 Required external downloads

| What | Where | Size | Required for |
|---|---|---|---|
| `tunejury.pt` + 7 ablation checkpoints | [`checkpoints/README.md`](../checkpoints/README.md) (HF: `TuneJury/tunejury`) | ~50 MB total | Any scoring at all |
| `music_audioset_epoch_15_esc_90.14.pt` (LAION-CLAP-Music) | auto-download from HF on first use | 2.3 GB | Any scoring at all |
| `MERT-v1-330M` weights | auto-download via `transformers` (after HF login, §0.3) | ~1.3 GB | Any scoring at all |
| `data/processed_features/` (~480 MB extracted; ~280 GB raw upstream audio) | not redistributable; extract locally per §2 | 480 MB extracted | `tunejury.train` (§3), internal eval (§4) |
| CMI-RewardBench manifest | <https://github.com/Haiwen-Xia/CMI-RewardBench> | ~2 GB | §5 head-to-head |
| SDD-706 reference audio (MTG-Jamendo subset) | <https://github.com/mulab-mir/song-describer-dataset> | ~3.5 GB | Mode 1 BoN (§7), Mode 3 MAD (§9), all FAD-CLAP / FAD-MERT against SDD-706 |
| Mode 1 backbones (MusicGen-{medium,large}, AudioLDM2-music, ACE-Step Turbo Continuous) | auto-download on first use via `audiocraft` / `diffusers` / `acestep` | ~10 GB total | §7 |
| Mode 2 backbones (SAO-small, TangoFlux) | auto-download via `stable_audio_tools` / TangoFlux release | ~5 GB total | §8 |
| Mode 3 backbone (FluxAudio-S, 120 M DiT) | MeanAudio release: download `fluxaudio_s_full.pth` from `huggingface.co/AndreasXi/MeanAudio` and use the MeanAudio training framework (`github.com/xiquan-li/MeanAudio`) for Step 4 (fine-tune). | ~500 MB ckpt | §9 expert iter |

### 0.4a Backbone licenses (Modes 1–3)

The Mode 1–3 generation backbones are external dependencies, downloaded from their upstream sources (table above) and not redistributed in this repository. Their upstream licenses:

| Backbone | Mode | License |
|---|---|---|
| MusicGen-medium / -large | 1 | weights CC-BY-NC 4.0, code MIT (`facebookresearch/audiocraft`) |
| AudioLDM2-music | 1 | CC-BY-NC-SA 4.0 |
| ACE-Step (Turbo Continuous) | 1 | Apache 2.0 |
| Stable Audio Open small (SAO-small) | 2 | Stability AI Community License (non-commercial; commercial use requires a separate Stability license) |
| TangoFlux | 2 | Stability AI Community License (non-commercial research) |
| FluxAudio-S / MeanAudio | 3 | MIT (Sony Research Inc.) |

### 0.5 Compute

- **GPU.** We ran every experiment on a single RTX A5000 24 GB. Mode 2 TangoFlux DITTO (§8) and Mode 1 MusicGen-large need gradient checkpointing + fp16 to fit; the lighter pipelines (TuneJury head §3, internal eval §4, CMI-RewardBench eval §5, anchor calibration §10, probes §11, Mode 3 expert iter §9) leave most of the 24 GB unused. Smaller-memory GPUs may work but are untested.
### 0.6 Numeric reproducibility

All paper numbers were computed with torch 2.4.0, torchaudio 2.4.0, transformers 4.44.0, and laion_clap 1.1.6 on an NVIDIA RTX A5000. The pinned environment in `environment.yml` (torch 2.7.1, transformers 4.57.6) runs every script but shifts the frozen-encoder numerics. Measured deviations on the canonical Mode 2/3 audio:

| Quantity | Published (torch 2.4.0) | Fresh run (torch 2.7.1) |
| --- | --- | --- |
| Mode 2 TangoFlux ΔCLAP | +0.043 | +0.027 |
| Mode 3 ΔCLAP (LR 1e-6 / 5e-6 / 1e-5) | +0.019 / +0.027 / +0.023 | +0.006 / +0.013 / +0.008 |
| Mode 3 baseline mean reward | -0.262 | -0.214 |
| Mode 3 ΔRwd, Win (LR 1e-5) | +0.416, 75/100 | +0.429, 76/100 |

Signs, orderings, and win counts are essentially unchanged. For bit-exact reproduction of the published numbers, install the torch 2.4.0 stack above. laion_clap 1.1.6 and 1.1.7 are functionally identical (1.1.7 only adds a `weights_only=False` flag to `torch.load`).

### 0.7 Per-Mode environment table

| Mode | Conda env | Needed for |
|---|---|---|
| Training TuneJury (the head) | `meanaudio` (Mode 3 driver repo) **or** `tunejury` | `tunejury/train.py` |
| Mode 1 · MusicGen-medium/large | `musicgen` | `applications/mode1_bon/generate_musicgen.py` |
| Mode 1 · ACE-Step Turbo Continuous | `ace_step` | `applications/mode1_bon/generate_acestep.py` |
| Mode 1 · AudioLDM2-music | `meanaudio` (shares the `diffusers` pin) | `applications/mode1_bon/generate_audioldm2.py` |
| Mode 2 · SAO-small | `sao` | `applications/mode2_ditto/sao_small.py` |
| Mode 2 · TangoFlux | `tangoflux` | `applications/mode2_ditto/tangoflux_ditto.py` |
| Mode 3 · FluxAudio-S expert iter | `meanaudio` (from `github.com/xiquan-li/MeanAudio`) | `applications/mode3_expert_iter/run.py` |
| TuneJury scoring + internal eval + CMI-RewardBench + anchor calibration + probes + figures | `tunejury` | everything else |

Once all six items above are in place, the §1 quick start and the §3-§11 reproducing pipeline should run end-to-end on a single 24 GB+ GPU (we used RTX A5000). If any step fails before that, please re-check this section before opening an issue.

---

## 1. Setup

```bash
git clone https://github.com/yonghyunk1m/TuneJury
cd TuneJury
conda env create -f environment.yml
conda activate tunejury
```

The seven TuneJury head checkpoints (`tunejury.pt` and the leave-out variants, ~11 MB each) ship under `checkpoints/`. The LAION-CLAP-Music checkpoint (`music_audioset_epoch_15_esc_90.14.pt`, ~2.2 GB) is **auto-downloaded** by `Scorer.from_pretrained` on first use; alternatively, pre-fetch it manually with the `wget` command in `checkpoints/README.md`. MERT-v1-330M is auto-downloaded from Hugging Face on first use.

**System prerequisites:** `apt-get install -y ffmpeg libsndfile1` (or the macOS / conda equivalents). A CUDA 12.x-compatible GPU is required for training and evaluation. Expect ~5 GB of Hugging Face-cached weights (LAION-CLAP ~2.2 GB + MERT ~1.3 GB + ancillary).

---

## 2. Data preparation

**On a fresh clone `data/processed_features/` is empty.** The four upstream label sources carry incompatible redistribution clauses (CC0, CC-BY, CC-BY 4.0, CC-BY-NC-SA 4.0), so we do not mirror their audio or our extracted features from this repo. To reproduce the paper's headline internal-eval numbers (test pairwise 0.7086, ECE 0.0339 on 2,035 non-tie held-out pairs; per-dataset 0.674 / 0.718 / 0.800 / 0.908) you must download the raw audio and run feature extraction yourself.

If you start training or evaluation against an empty `data/processed_features/`, `tunejury.dataset.TuneJuryDataset` will raise a `FileNotFoundError` that points back to this section rather than silently returning 0 pairs.

### 2.0 Sources, URLs, and licenses

| Source | Download URL | License | Pairs |
|---|---|---|---|
| Music Arena~\cite{kim2025musicarena} | `https://huggingface.co/datasets/music-arena/music-arena-dataset` (configs `2025_07` ... `2026_01`) | CC-BY 4.0 | 2,039 |
| MusicPrefs~\cite{huang2025musicprefs} | `https://huggingface.co/datasets/i-need-sleep/musicprefs` | no formal license (open-source) | 2,515 |
| AIME~\cite{grotschla2025aime} | `https://huggingface.co/datasets/disco-eth/AIME-survey` (labels) + `https://huggingface.co/datasets/disco-eth/AIME` (audio) | CC-BY 4.0 | 15,600 |
| SongEval~\cite{yao2025songeval} | `https://huggingface.co/datasets/ASLP-lab/SongEval` | CC-BY-NC-SA 4.0 | 2,399 songs → 3,760 synthesized pairs |

Total raw audio footprint after download: ~280 GB across the four sources (Music Arena ~18 GB, MusicPrefs ~12 GB, AIME ~210 GB, SongEval ~40 GB). Total extracted feature footprint: ~480 MB (each `.pt` blob is ~20 KB; ~21.8 K pairs after the bench-clean filter in §2.2).

**Pre-extracted feature bundle (planned).** Music Arena and AIME (both CC-BY 4.0) permit redistribution of derived features with attribution. MusicPrefs carries no formally declared license, so redistributing its derived features would require explicit permission from its authors. SongEval's CC-BY-NC-SA 4.0 share-alike clause means a SongEval bundle would have to be released separately under matching share-alike terms. The plan is to host a Music Arena $+$ AIME feature bundle on Hugging Face at `TuneJury/tune-jury-features`, with the MusicPrefs and SongEval portions handled separately under their respective constraints. Until those are published (`applications/anchor_calibration/README.md` will carry the live links once available), reproduce by following the end-to-end extraction in §2.0a below. The bundles, when released, will skip the ~280 GB raw-audio download step; they do not change the rest of the pipeline.

### 2.0a End-to-end regeneration pipeline

```bash
# Step 0: prerequisites. LAION-CLAP-Music checkpoint, ~6 GB free GPU memory
ls checkpoints/music_audioset_epoch_15_esc_90.14.pt   # must exist (see §1)

# Step 1: download raw audio for all four sources (URLs above).
#   Suggested layout (matches downstream scripts):
#     /path/to/audio/MusicArena/<battle_uuid>_a.wav
#     /path/to/audio/MusicArena/<battle_uuid>_b.wav
#     /path/to/audio/MusicPrefs/<pair_id>_a.wav, _b.wav
#     /path/to/audio/AIME/<pair_id>_a.wav, _b.wav
#     /path/to/audio/SongEval/<song_id>_a.wav, _b.wav

# Step 2 (SongEval only): synthesize pairwise labels
python data/prepare_songeval_pairs.py \
    --songeval-root /path/to/songeval \
    --out-pairs data/splits/songeval_pairs.json \
    --mean-gap 0.5

# Step 3 (optional, for item-level disjointness with CMI-RewardBench):
python data/filter_music_arena_bench_clean.py \
    --music-arena-root /path/to/music_arena \
    --cmi-bench-uuid-list /path/to/cmi_rewardbench/music_arena_uuids.txt \
    --out-pool data/music_arena_bench_clean.json

# Step 4: merge the four per-source pair files into one all_pairs.json
#   (each record carries pair_id, source, winner, prompt, a_audio, b_audio).
#   This step is dataset-specific: see each source's README for field names.

# Step 5: feature extraction
python data/extract_features.py \
    --pairs data/splits/all_pairs.json \
    --audio-root /path/to/audio \
    --out-dir data/processed_features \
    --encoders clap mert

# Step 6: validate. The .pt count must be non-zero (and typically ~21.8 K)
python - <<'PY'
from pathlib import Path
n = sum(1 for _ in Path("data/processed_features").glob("*.pt"))
print(f"Found {n} .pt blobs in data/processed_features/")
assert n > 0, "extraction produced 0 files; see docs/reproducing.md §2 and data/processed_features/README.md"
PY

# Step 7: (re)build the 17 derived .txt splits consumed by tunejury.train
python -m data.build_splits
```

### 2.1 SongEval pair synthesis

```bash
python data/prepare_songeval_pairs.py \
    --songeval-root /path/to/songeval \
    --out-pairs data/splits/songeval_pairs.json \
    --mean-gap 0.5
```

`--mean-gap 0.5` reproduces the paper's filter (mean across the 5 aesthetic axes differs by ≥ 0.5; higher-rated side labeled preferred). Yields 3,760 pairs (paper §3 Table 1).

### 2.2 Music Arena bench-clean filtering

CMI-RewardBench releases 1,340 `battle_uuid`s drawn from Music Arena as its test split. We remove all 1,340 from our Music Arena pool (train, val, held-out test) so TuneJury is item-level disjoint from the CMI-RewardBench Music Arena split.

```bash
python data/filter_music_arena_bench_clean.py \
    --music-arena-root /path/to/music_arena \
    --cmi-bench-uuid-list /path/to/cmi_rewardbench/music_arena_uuids.txt \
    --out-pool data/music_arena_bench_clean.json
```

`music_arena_uuids.txt` is the newline-delimited list of 1,340 `battle_uuid`s from the upstream CMI-RewardBench release (see §5.0 for download URLs: https://github.com/Haiwen-Xia/CMI-RewardBench). The repository does **not** ship this file. For reviewers who only want to reproduce the anchor calibration in §10, the downstream split JSON `data/splits/MusicArena-random_split_bench_clean.json` is already committed, so this filtering step is optional for that path.

Expected: all 1,340 `battle_uuid`s found in the raw Music Arena pool; 131 of those land in our internal Music Arena test (which then shrinks to 74 pairs, §3 Table footnote).

### 2.3 Feature extraction

```bash
python data/extract_features.py \
    --pairs data/splits/all_pairs.json \
    --audio-root /path/to/audio \
    --out-dir data/processed_features \
    --encoders clap mert
```

Each clip yields a `.pt` blob with `clap_a` / `clap_b` (512-d each, LAION-CLAP audio), `text_emb` (512-d, CLAP text; all-zero for SongEval), `mert_a` / `mert_b` (1024-d each, MERT time-mean of final layer), plus `flag`, `winner`, `source`, `prompt`, `uuid`. The schema is consumed by `tunejury/dataset.py:TuneJuryDataset.__getitem__`. After extraction, `ls data/processed_features/ | wc -l` should match the number of records in `data/splits/all_pairs.json` (typically ~21.8 K after the bench-clean filter).

### 2.4 Pair-id splits

The repo includes the four per-dataset random-split JSON files under `data/splits/`
(MusicArena bench-clean, MusicPrefs, AIME, SongEval-balanced). To (re)generate the
17 derived `.txt` split files consumed by `tunejury.train` and `tunejury.batch_ablations`:

```bash
python -m data.build_splits
```

Outputs (all under `data/splits/`):

- `train.txt` (17,554), `val.txt` (2,111), `test.txt` (2,135): full four-dataset pools.
- `test_music_arena.txt` (74), `test_musicprefs.txt` (252), `test_aime.txt` (1,560), `test_songeval.txt` (249): per-dataset held-out splits.
- `train_no_MA.txt`, `train_no_MP.txt`, `train_no_AIME.txt`, `train_no_SE.txt`: single-leave-out training pools.
- `train_no_SE_MA.txt`, `train_no_MP_MA.txt`: double-leave-out pools.

The script is idempotent.

### 2.5 SDD-706 reference (Mode 1 / Mode 2 / Mode 3 prerequisite)

Mode 1 BoN (§7), Mode 2 DITTO (§8), and Mode 3 expert iteration (§9) all evaluate against the **Song Describer Dataset (SDD)** 706-clip subset (Manco et al. 2023). The 100-prompt subset used as conditioning text is committed at `eval/prompts/sdd100.json`; the 706-clip reference audio for FAD / MAD distributional metrics is **not** shipped.

To prepare the SDD-706 reference directory:

1. Download SDD: `https://github.com/mulab-mir/song-describer-dataset` (CC-BY-NC-SA; audio sourced from MTG-Jamendo).
2. Filter to the canonical 706 clips (the subset used in the original Stable Audio Open evaluation). The required `track_id` list is derivable from the SDD `descriptions.csv` after dropping rows with `is_valid_subset == False` and de-duplicating to one caption per clip.
3. Resample all clips to 16 kHz mono WAV under a single directory. This is the path passed as `--reference-dir` (Mode 1, §7 Step 3) and `--sdd706_dir` (Mode 1 / Mode 3 MAD, §7 Step 4 and §9).

For Mode 3 the MAD compute script (`applications/mode3_expert_iter/mad_compute.py`) expects pre-rendered `audio_wav_torch` `.pt` files (one per clip, 16 kHz mono, torch tensor); the README in `applications/mode3_expert_iter/` documents the converter.

### 2.6 Post-cutoff Music Arena slice (Anchor calibration prerequisite)

§10 anchor calibration uses a 598-pair post-cutoff Music Arena slice (Feb–Mar 2026) plus a 397-pair April 2026 cross-month slice. These are drawn from `huggingface.co/datasets/music-arena/music-arena-dataset` configs `2026_02`, `2026_03`, and `2026_04`.

The repository ships:

- `data/splits/MusicArena_v2_increment.json`: the 598 Feb-Mar `new_uuids` (small JSON, top-level `{"new_uuids": [<uuid>, ...]}`).
- `data/labels/ma_postcut_scored.csv`: the pairwise vote CSV with columns `battle_id` (UUID), `preference` (`A` or `B`, ties excluded), `system_a`, `system_b` (generator tags).

The post-cutoff features (CLAP/MERT `.pt` blobs, same schema as `data/processed_features/`) are **not** in the git tree on a fresh clone. Re-extract them from the upstream HF audio via `data/extract_features.py` against the Feb-Mar audio pool and pass the output directory as `--feat-dir`. The April cross-month equivalents go under `--april-feat-dir`/`--april-meta-dir`. (A pre-extracted bundle for Music Arena post-cutoff is planned as part of the `TuneJury/tune-jury-features` Hugging Face release described in §2.0; until that is published, local re-extraction is the only option.)

---

## 3. Training

### 3.1 Released checkpoint

```bash
python -m tunejury.train \
    --features-dir data/processed_features \
    --train-ids data/splits/train.txt \
    --val-ids data/splits/val.txt \
    --out checkpoints/tunejury.pt
```

AdamW (lr `1e-4`, wd `1e-3`, batch 32), early stopping on val loss with patience 30.

### 3.2 Leave-one-dataset-out ablations

Six leave-out checkpoints are included pre-trained under `checkpoints/tunejury_leave_*.pt`. To retrain from scratch:

```bash
# Single retrain (example: drop MusicPrefs)
python -m tunejury.train \
    --features-dir data/processed_features \
    --train-ids data/splits/train_no_MP.txt \
    --val-ids data/splits/val.txt \
    --out checkpoints/tunejury_leave_MP.pt

# Batch retrain all 6 leave-out variants in one go
python -m tunejury.batch_ablations \
    --features-dir data/processed_features \
    --splits-dir data/splits \
    --out-dir checkpoints/
```

The batch script trains 6 single-leave-out (MA, MP, AIME, SE) and 2 double-leave-out (SE+MA, MP+MA) variants. Each is a single-seed reproducer.

### 3.3 Input ablation (Appendix C)

Seven input-modality variants from `tab:feature_modality`. Each variant masks a different
subset of the three input blocks (`clap_audio`, `mert`, `text`) via the corresponding
`--no-X` flag on `tunejury.train`; `input_dim` is set dynamically from the remaining blocks
and a sidecar `.json` is written next to each checkpoint with the resolved
`{input_dim, exclude}` so `eval.input_ablation_eval` can reload each variant correctly.

```bash
python -m tunejury.batch_input_ablation \
    --features-dir data/processed_features \
    --splits-dir data/splits \
    --out-dir checkpoints/input_ablation/
```

Outputs seven checkpoints (one .pt and one sidecar .json each):
`A1_clap_audio_only`, `A2_mert_only`, `A3_text_only`, `A4_clap_audio_mert`,
`A5_clap_audio_text`, `A6_mert_text`, `A7_full` (= released architecture).

Then evaluate per variant on the four held-out test splits:

```bash
python -m eval.input_ablation_eval \
    --ckpt-dir checkpoints/input_ablation/ \
    --features-dir data/processed_features \
    --splits-dir data/splits \
    --out-csv results/input_ablation_seed42.csv
```

All runs use `seed=42` (the default in `tunejury.train`). Per the paper's
single-seed retrain caveat, absolute accuracies vary ${\sim}0.01$ across seeds;
the released `results/input_ablation_seed42.csv` is the reproducer reference
for Appendix C Table `feature_modality`.

### 3.4 MuQ-MuLan encoder swap

```bash
python -m tunejury.train \
    --features-dir data/muq_features \
    --train-ids data/splits/train_no_MA.txt \
    --val-ids data/splits/val.txt \
    --out checkpoints/tunejury_muq_leave_MA.pt
```

Use `data/extract_features.py --encoders muq` first to populate `data/muq_features/`.
The encoder choice is read from the sidecar `.json` written next to the checkpoint
(input dimension auto-resolved from the feature stack).

---

## 4. Internal evaluation

### 4.1 Pairwise accuracy + ECE (§4.1)

```bash
python -m eval.internal \
    --checkpoint checkpoints/tunejury.pt \
    --features-dir data/processed_features \
    --test-ids data/splits/test.txt
```

Expected: `accuracy ≈ 0.7086`, `ece ≈ 0.0339`, `n_non_tie = 2035`.

### 4.2 Calibration bins + reliability diagram (Appendix A)

Two steps: first emit the 10-bin reliability JSON, then render the figure.

```bash
# Step 1: bin data (also re-prints accuracy / ECE / n_non_tie from §4.1)
python -m eval.internal \
    --checkpoint checkpoints/tunejury.pt \
    --features-dir data/processed_features \
    --test-ids data/splits/test.txt \
    --out-bins-json results/calibration_bins.json

# Step 2: render Figure `fig:calibration` (Figure 4 in the paper)
python figures/make_calibration_figure.py \
    --bins-json results/calibration_bins.json \
    --out       figures/fig_calibration.pdf
```

Reproduces `tab:calibration_bins` (10 equal-count bins) and `fig:calibration` (two-panel: reliability + win-rate vs predicted margin). The overall accuracy 0.7086 / ECE 0.0339 figures are from §4.1 of the paper.

### 4.3 Per-dataset leave-out (Table `internal_per_dataset`)

```bash
python -m eval.internal_per_dataset \
    --checkpoint checkpoints/tunejury.pt \
    --leave-out-checkpoints checkpoints/tunejury_leave_MA.pt \
                            checkpoints/tunejury_leave_MP.pt \
                            checkpoints/tunejury_leave_AIME.pt \
                            checkpoints/tunejury_leave_SE.pt \
    --features-dir data/processed_features \
    --splits-dir data/splits
```

Expected: in-mix lift of +0.029 (MusicPrefs) to +0.093 (SongEval) across the four sources.

The full train-by-test matrix in paper Table 3 (every leave-out checkpoint on every per-dataset test split, including the off-diagonal cells) is reproduced by:

```bash
python -m eval.loo_matrix \
    --checkpoint checkpoints/tunejury.pt \
    --leave-out-checkpoints checkpoints/tunejury_leave_AIME.pt \
                            checkpoints/tunejury_leave_MP.pt \
                            checkpoints/tunejury_leave_MA.pt \
                            checkpoints/tunejury_leave_SE.pt \
    --features-dir data/processed_features \
    --splits-dir data/splits
```

Expected values are listed in the script docstring; the Full row and the diagonal reproduce the values above, and the paper's All column is the pair-weighted average of the four splits.

### 4.4 Full-test gain'_d (Table `internal_per_dataset`, second gain column)

```bash
python -m eval.full_test_gain \
    --checkpoint checkpoints/tunejury.pt \
    --leave-out-ckpts checkpoints/tunejury_leave_MA.pt \
                      checkpoints/tunejury_leave_MP.pt \
                      checkpoints/tunejury_leave_AIME.pt \
                      checkpoints/tunejury_leave_SE.pt \
    --features-dir data/processed_features \
    --test-ids data/splits/test.txt
```

Expected: full-test gain' of +0.0408 (AIME), +0.0029 (MA), +0.0054 (MP), +0.0029 (SE). AIME's gain' dominates because it accounts for 77% of the 2,035-pair test set.

---

## 5. External evaluation (CMI-RewardBench)

### 5.0 CMI-RewardBench preparation (prerequisite)

CMI-RewardBench (Ma et al. 2026, arXiv:2603.00610) is an *external* benchmark and is **not** shipped with this repository. Every command in §5.1, §5.2, §5.3 takes `--bench-root /path/to/CMI-RewardBench` pointing at a local copy you must prepare first.

**Upstream sources:**

| Artefact | URL |
|---|---|
| Paper | https://arxiv.org/abs/2603.00610 |
| Code | https://github.com/Haiwen-Xia/CMI-RewardBench |
| CMI-Pref (4k human) | https://huggingface.co/datasets/HaiwenXia/cmi-pref |
| CMI-Pref pseudo (110k) | https://huggingface.co/datasets/HaiwenXia/cmi-pref-pseudo |
| License | CC-BY-NC-SA |

The four CMI-RewardBench splits used by §4.2 / Table 4 are:

- **PAM** (500 clips, musicality MOS regression): from Manco et al., redistributed in the CMI-RewardBench release.
- **MusicEval** (413 clips, musicality MOS regression): from Liu et al.
- **CMI-Pref** (500-pair test split, pairwise): from `HaiwenXia/cmi-pref`.
- **Music Arena** (1,340-pair benchmark, pairwise): `battle_uuid`s drawn from the Music Arena dataset; removed from our train/val/test by `data/filter_music_arena_bench_clean.py` (§2.2).

**Required directory layout** (`--bench-root` should point to this directory):

```
<bench-root>/
  manifest.json                # see schema below
  audio/                       # any layout; manifest paths can be relative to <bench-root> or absolute
    pam/...
    musiceval/...
    cmi_pref/...
    music_arena/...
```

**Expected `manifest.json` schema** (consumed by `eval/cmi_rewardbench.py`):

```json
{
  "pam":         [{"audio_path": "audio/pam/0001.wav",        "prompt": "...", "musicality_mos": 4.2}, ...],
  "musiceval":   [{"audio_path": "audio/musiceval/0001.wav",  "prompt": "...", "musicality_mos": 3.8}, ...],
  "cmi_pref":    [{"audio_a": "audio/cmi_pref/0001_a.wav",   "audio_b": "audio/cmi_pref/0001_b.wav",   "prompt": "...", "winner": "a"}, ...],
  "music_arena": [{"audio_a": "audio/music_arena/0001_a.wav","audio_b": "audio/music_arena/0001_b.wav","prompt": "...", "winner": "b"}, ...]
}
```

`audio_path`, `audio_a`, `audio_b` may be absolute paths or relative to `<bench-root>`. `winner` is `"a"` or `"b"`. `musicality_mos` is a float (1–5 scale in the upstream release).

The upstream CMI-RewardBench release does *not* ship a `manifest.json` in this exact four-key layout. Reviewers must build it once from the upstream files (paths and labels are all derivable from the upstream split CSVs / JSONs).

**Sanity check before running §5.1:** `eval/cmi_rewardbench.py` will refuse to run if `--bench-root` is missing or `manifest.json` lacks any of the four keys, and will print this section's URL.

### 5.1 Head-to-head (Table `head_to_head`)

```bash
python -m eval.cmi_rewardbench \
    --checkpoint checkpoints/tunejury.pt \
    --bench-root /path/to/CMI-RewardBench \
    --prompt-protocol with
```

Expected (T+A protocol): `PAM 0.6100`, `MusicEval 0.6687`, `CMI-Pref 0.7140`, `MA bench 0.7194`.
Empty-prompt protocol (`--prompt-protocol empty`): `PAM 0.6731`, `MusicEval 0.6618`, `CMI-Pref 0.7240`, `MA bench 0.7007`.

### 5.2 Training-mix design space (Table `training_mix`)

```bash
python -m eval.cmi_rewardbench_sweep \
    --checkpoints checkpoints/tunejury.pt \
                  checkpoints/tunejury_leave_AIME.pt \
                  checkpoints/tunejury_leave_MP.pt \
                  checkpoints/tunejury_leave_MA.pt \
                  checkpoints/tunejury_leave_SE.pt \
                  checkpoints/tunejury_leave_SE_MA.pt \
                  checkpoints/tunejury_leave_MP_MA.pt \
    --bench-root /path/to/CMI-RewardBench \
    --out-csv results/training_mix.csv
```

### 5.3 Encoder swap (Table `encoder_swap`)

```bash
python -m eval.cmi_rewardbench \
    --checkpoint checkpoints/tunejury_muq_leave_MA.pt \
    --bench-root /path/to/CMI-RewardBench
```

The encoder choice is read from the checkpoint sidecar `.json` (no
`--feature-stack` flag on `eval/cmi_rewardbench.py`).

### 5.4 AIME held-out per-axis (Table `aime_per_axis`)

```bash
python -m eval.aime_per_axis \
    --checkpoint checkpoints/tunejury.pt \
    --aime-test-split /path/to/aime/test_1560.json \
    --features-dir data/processed_features \
    --out-csv results/aime_per_axis.csv
```

### 5.5 MuQ-Eval head-to-head baseline

Concurrent MuQ-Eval (Zhu & Li 2026, arXiv:2603.22677), an MSE MOS regressor over a frozen MuQ-large encoder trained on MusicEval, is the only baseline whose paper does not include CMI-RewardBench numbers. We score its public HuggingFace A1 checkpoint on the four test splits ourselves and report the row in Table `head_to_head`.

```bash
# Paper-exact: reads applications/baselines/results/muqeval_a1/summary.json
python applications/baselines/muqeval_a1.py

# Fresh re-inference from raw audio:
python applications/baselines/muqeval_a1.py \
    --reinfer --bench_root /path/to/CMI-RewardBench/data \
    --device cuda:0
```

### 5.6 Decomposition probe (Section 3 + Figure)

Four-stage probe asking whether TuneJury's features can be decomposed into
musicality + alignment axes. See `eval/decomposition_probe.py` for the full
defining math; the stages run independently or together via `--stage`.

```bash
python -m eval.decomposition_probe \
    --checkpoint     checkpoints/tunejury.pt \
    --cmi-root       /path/to/CMI-RewardBench/data \
    --stage          all \
    --n-seeds-stab   20 \
    --n-seeds-scale  5 \
    --output-json    decomposition_probe.json
```

Expected (paper Section 3 Decomposition probe paragraph):

| Stage | Result |
|---|---|
| (i)  Post-hoc PAM | composite musicality SRCC ${+}0.59$, audio-only ${+}0.67$; delta vs PAM alignment MOS ${-}0.30$ |
| (i)  Post-hoc MusicEval | composite musicality ${+}0.67$, audio-only ${+}0.66$; delta vs alignment MOS ${+}0.02$ |
| (ii) Cross-distribution | PAM→MusicEval alignment SRCC ${+}0.18$; MusicEval→PAM ${-}0.41$ |
| (iii) Stratified (20-seed) | alignment ${+}0.630 \pm 0.047$, partial ${+}0.305 \pm 0.074$, residual ${+}0.444 \pm 0.058$ |
| (iv)  Scaling (5-seed) | partial at $n{=}36$: ${+}0.085$ → $n{=}728$: ${+}0.318$ (monotone, no plateau) |

Expected: `PAM 0.4995`, `MusicEval 0.8089` (in-distribution for MuQ-Eval, italicized in the paper), `CMI-Pref 0.6600`, `MA bench 0.6761`. See [`applications/baselines/README.md`](../applications/baselines/README.md) for the runner walkthrough and the upstream-repo / HuggingFace-checkpoint prerequisites.

---

## 6. Adversarial sanity checks

```bash
# Boundary inputs (Table sanity_boundary)
python -m eval.sanity --check boundary --out-csv results/sanity_boundary.csv

# Graded perturbations: SNR sweep + clip-ratio sweep (Table sanity_perturb)
python -m eval.sanity --check perturb --out-csv results/sanity_perturb.csv

# Length sensitivity (1/3/5/10/15/20/30/45/60/120s + full-track)
python -m eval.sanity --check length --out-csv results/sanity_length.csv

# Per-segment discrimination probe (silence/noise insertion at center 10s slot)
python -m eval.sanity --check segment --out-csv results/sanity_segment.csv

# Temporal-structure sensitivity (time-reversal)
python -m eval.sanity --check temporal --out-csv results/sanity_temporal.csv
```

All five use the same 8-clip Jamendo set, fixed seed for reproducibility.

---

## 7. Mode 1: best-of-N selection (§5.1 + Appendix G)

Full instructions in [`applications/mode1_bon/README.md`](../applications/mode1_bon/README.md).

**Mode 1 prerequisites:**

- §1 Setup completed (the `tunejury` conda env, the TuneJury head checkpoint, the LAION-CLAP-Music checkpoint).
- §2.5 SDD-706 reference directory ready. The 100 conditioning prompts are committed at `eval/prompts/sdd100.json`; the 706-clip reference audio (16 kHz mono WAV) is needed for `--reference-dir` in Step 3 and `--sdd706_dir` in Step 4.
- Each backbone needs its own conda env (see §13): `musicgen`, `meanaudio`, `ace_step`. The four backbones are MusicGen-medium, MusicGen-large, AudioLDM2-music, and ACE-Step v1.5 Turbo Continuous; `N ∈ {1, 2, 4, 8, 16, 32}`.

Per-backbone summary:

```bash
cd applications/mode1_bon

# Step 1: generate N=32 candidates per backbone (seeds 42..73)
python generate_musicgen.py  --prompts ../../eval/prompts/sdd100.json \
    --model facebook/musicgen-medium --out_dir results/musicgen_medium_bon100 \
    --n_candidates 32 --seeds $(seq 42 73)
python generate_musicgen.py  --prompts ../../eval/prompts/sdd100.json \
    --model facebook/musicgen-large  --out_dir results/musicgen_large_bon100 \
    --n_candidates 32 --seeds $(seq 42 73)
python generate_audioldm2.py --prompts ../../eval/prompts/sdd100.json \
    --out_dir results/audioldm2_music_bon100 --n_candidates 32 --seeds $(seq 42 73)
python generate_acestep.py   --prompts ../../eval/prompts/sdd100.json \
    --out_dir results/acestep_bon100 --n_candidates 32 --seeds $(seq 42 73)
# The prompt prefix "high quality instrumental music, " is applied to ALL
# four backbones. ACE-Step v1.5 Turbo Continuous (the only Mode 1 backbone
# with a separate lyric channel) additionally receives an empty lyric input.

# Step 2: score + select top-1 at N ∈ {1,2,4,8,16,32}
# See applications/mode1_bon/README.md for the canonical form.
for bb in musicgen_medium musicgen_large audioldm2_music acestep; do
    python score_and_select.py \
        --prompts ../../eval/prompts/sdd100.json \
        --candidates_dir results/${bb}_bon100 \
        --ckpt ../../checkpoints/tunejury.pt \
        --n_candidates 32 --N 1 2 4 8 16 32 \
        --per_cand_csv results/${bb}_bon100/tunejury_per_candidate.csv \
        --out_csv results/${bb}_bon100/results.csv
done

# Step 3: distributional metrics (FAD-CLAP/MERT + CLAP score) for each top-N
for bb in musicgen_medium musicgen_large audioldm2_music acestep; do
    for n in 1 2 4 8 16 32; do
        python -m eval.distributional \
            --reference-dir /path/to/sdd_706_audio \
            --candidate-dir results/${bb}_bon100/topN_${n} \
            --encoder clap
        python -m eval.distributional \
            --reference-dir /path/to/sdd_706_audio \
            --candidate-dir results/${bb}_bon100/topN_${n} \
            --encoder mert
        python -m eval.clap_score \
            --prompts ../../eval/prompts/sdd100.json \
            --candidates_dir results/${bb}_bon100/topN_${n}
    done
done

# Step 4: MAD (MAUVE Audio Divergence) on MERT for all N
python mad_compute.py \
    --bon_results_root /path/to/results \
    --sdd706_dir /path/to/SDD-706/audio_wav_torch \
    --prompts ../../eval/prompts/sdd100.json \
    --backbones musicgen_medium musicgen_large audioldm2_music acestep

# Step 5: mode-collapse diagnostic (MPCD) at N ∈ {1,8,16,32}
python mpcd_diagnostic.py \
    --backbones musicgen_medium musicgen_large audioldm2_music acestep \
    --results-root results
```

Expected (Section 5.1 + Appendix G):
- Reward strictly monotone in N for all 4 backbones through N=32 (`tab:bon_saturation_4bb`)
- Per-doubling gain narrows from `[+0.178, +0.291]` at N=4→8 to `[+0.060, +0.144]` at N=16→32; AudioLDM2-music saturates earliest (smallest gain +0.060 at N=16→32)
- MAD on MERT: AudioLDM2 peaks at N=8, ACE-Step Turbo Continuous at N=16 (`tab:mad_mode1`)
- MPCD: AudioLDM2 +39%, ACE-Step Turbo Continuous +66% mean pairwise distance growth from N=1→32 (rules out mode collapse, `tab:mode1_diversity`)

---

## 8. Mode 2: DITTO-style latent optimization (§5.2)

Full instructions in [`applications/mode2_ditto/README.md`](../applications/mode2_ditto/README.md).

**Mode 2 prerequisites:**

- §1 Setup completed.
- Two separate conda envs: `sao` (Stable Audio Open small, n=30 prompts) and `tangoflux` (TangoFlux, n=100 prompts). See §13. AudioLDM2 is **omitted** from Mode 2 (memory ceiling; see note below).
- The `sao_small.py` entry-point monkey-patches `stable_audio_tools.generate_diffusion_cond` at import time to enable full-sampler backprop on the DITTO objective; no fork-and-pin is required.
- **Version pin**: paper-reproducible behavior is verified against `stable-audio-tools==0.0.19` (May 2026 snapshot). Later releases exceed 24 GB at $n{=}100$ ($10$\,s $/$ $44.1$\,kHz, default CFG $6$); see paper §A `app:sao_caveat`. Pin via `pip install stable-audio-tools==0.0.19` if reproducing exact numbers.
- Conditioning prompts are committed at `eval/prompts/sdd100.json`.

Headline:

```bash
cd applications/mode2_ditto

# SAO-small
conda activate sao
python sao_small.py \
    --prompts ../../eval/prompts/sdd100.json \
    --checkpoint ../../checkpoints/tunejury.pt \
    --out_dir results/ditto_sao_small --limit 30

# TangoFlux (full sampler backprop)
conda activate tangoflux
python tangoflux_ditto.py \
    --prompts ../../eval/prompts/sdd100.json \
    --checkpoint ../../checkpoints/tunejury.pt \
    --out_dir results/ditto_tangoflux --limit 100

# Compute ΔCLAP for each backbone (see applications/mode2_ditto/README.md)
python compute_clap_delta.py \
    --results_dir  results/ditto_sao_small \
    --json         results/ditto_sao_small/summary.json \
    --prompts_json ../../eval/prompts/sdd100.json \
    --clap_ckpt    ../../checkpoints/music_audioset_epoch_15_esc_90.14.pt \
    --limit 30
# Repeat with --results_dir results/ditto_tangoflux --json results/ditto_tangoflux/summary.json --limit 100
```

AudioLDM2-music is omitted from Mode 2: its 50-step denoising trajectory exceeds the full-backprop memory budget on our hardware, and partial late-stage backprop on only the final few steps produced perceptually muted reward lifts (paper §5.2).

Expected (Table 5, top panel):
- SAO-small (n=30): `+0.159 → +0.404` (Δreward `+0.245`, ΔMAD `+0.500`, ΔCLAP `−0.007`, 19/30)
- TangoFlux (n=100): `−0.978 → +0.578` (Δreward `+1.557`, ΔMAD `−2.214`, ΔCLAP `+0.043`, 100/100)

---

## 9. Mode 3: expert-iteration post-training (§5.3 + Appendix H)

Full instructions in [`applications/mode3_expert_iter/README.md`](../applications/mode3_expert_iter/README.md).

```bash
cd applications/mode3_expert_iter
python run.py \
    --candidates_dir /path/to/mode3_candidates \
    --checkpoint ../../checkpoints/tunejury.pt \
    --out_dir results/mode3_expert_iter \
    --n_candidates 900 --iterations 5000 --lr 1e-5 --batch_size 16
```

Paper Table 5 (bottom panel) reports a three-LR sweep mapping the reward-fidelity Pareto frontier (baseline reward $-0.262$ on FluxAudio-S). MAD is reported as $-\ln(\mathrm{MAUVE})$ per Huang et al. 2025; range $[0, \infty)$, **lower is closer** to the SDD-706 reference distribution, so $\Delta$MAD $> 0$ here indicates increased divergence from reference (a fidelity cost).

| LR | After | $\Delta$Rwd | $\Delta$MAD | $\Delta$CLAP | Win |
|---|---|---|---|---|---|
| $10^{-6}$ (conservative)   | $-0.096$ | $+0.166$ | $+0.293$ | $+0.019$ | 67/100 |
| $5{\times}10^{-6}$         | $+0.107$ | $+0.369$ | $+0.284$ | $+0.027$ | 73/100 |
| $10^{-5}$ (aggressive)     | $+0.154$ | $+0.416$ | $+0.669$ | $+0.023$ | 75/100 |

Multi-round expert iteration at LR $10^{-6}$ (paper Table 5, Appendix H). $\Delta$MAD is cumulative from the FluxAudio-S baseline. R1 matches the single-round LR $10^{-6}$ row above.

| Round | $\Delta$Rwd (cumulative) | $\Delta$MAD (cumulative) |
|---|---|---|
| R1 | $+0.166$ | $+0.293$ |
| R2 | (see paper) | $+0.836$ |
| R3 | (see paper) | $+0.978$ |

The reward-vs-MAD divergence is the classic reward-exploitation signature (paper §5.3). Among the swept rates, $5{\times}10^{-6}$ is the most favorable single-round trade-off: more than 2$\times$ reward lift over $10^{-6}$ at essentially the same MAD cost. We do not claim this is the global Pareto optimum, only the best of the three swept points. Multi-round expert iteration at the conservative $10^{-6}$ LR amplifies both reward gain and MAD drift.

**Mode 3 backbone prerequisites:**

The FluxAudio-S backbone checkpoint and the MeanAudio training framework are external dependencies:

1. Clone MeanAudio: `git clone https://github.com/xiquan-li/MeanAudio`.
2. Follow MeanAudio's README to fetch the FluxAudio-S checkpoint (`fluxaudio_s_full.pth` from `huggingface.co/AndreasXi/MeanAudio`) into the expected local path.
3. `run.py` in this repo handles candidate scoring and Top-K filtering; the actual gradient-update step runs in the MeanAudio repo. `run.py` prints the exact MeanAudio entry-point commands at the end of each invocation.
4. The `--score 4.5` MeanAudio flag corresponds to the paper's "CFG 4.5" knob; `--cfg 2.5` is a secondary CFG-rescale kept at MeanAudio's default at inference time.

Per-round candidate manifests (per-candidate prompt + reward) for the LR sweep are committed at `applications/mode3_expert_iter/mad_unified.json` for audit (paper Table 5 was regenerated against these). SDD-706 reference audio for the MAD metric is the same one prepared in §2.5.

---

## 10. Anchor calibration (§6 + Appendix D)

Full instructions in [`applications/anchor_calibration/README.md`](../applications/anchor_calibration/README.md).

**Anchor calibration prerequisites** (see §2.6 for the full data-preparation recipe):

- §1 Setup completed; `checkpoints/tunejury.pt` resolved (md5 `0524e60900e5ee3c4046e599f613b466`).
- `data/splits/MusicArena_v2_increment.json` and `data/labels/ma_postcut_scored.csv` are committed in the repository.
- Pre-extracted CLAP/MERT features for the 598 Feb-Mar post-cutoff Music Arena pairs and the 397 April cross-month pairs are required; see §2.6 for the extraction recipe. Replace `/path/to/musicarena/features` and `/path/to/april/{features,json}` with your local feature dirs in the commands below.

```bash
# Headline anchor (A) row: K-sweep on the 598-pair Feb-Mar post-cutoff slice,
# 50/50 held-out split, 5 seeds, K ∈ {0, 3, 10, 30, 100, 250}
python -m applications.anchor_calibration.run_experiment \
    --checkpoint checkpoints/tunejury.pt \
    --feat-dir /path/to/musicarena/features \
    --increment-split data/splits/MusicArena_v2_increment.json \
    --label-csv data/labels/ma_postcut_scored.csv \
    --out results/ood_repro.json

# Retrain (R) row: from-scratch head training on bench_clean ∪ K added pairs,
# same 50/50 split and seeds as the anchor row
python -m applications.anchor_calibration.retrain_ksweep \
    --feat-dir /path/to/musicarena/features \
    --bench-split data/splits/MusicArena-random_split_bench_clean.json \
    --increment-split data/splits/MusicArena_v2_increment.json \
    --label-csv data/labels/ma_postcut_scored.csv \
    --out results/ood_retrain.json

# Cross-month transfer: Feb-Mar β_s applied to April 2026 (Table cross_month)
python -m applications.anchor_calibration.cross_month_eval \
    --checkpoint checkpoints/tunejury.pt \
    --feat-dir /path/to/musicarena/features \
    --label-csv data/labels/ma_postcut_scored.csv \
    --april-feat-dir /path/to/april/features \
    --april-meta-dir /path/to/april/json \
    --out results/april_cross_month.json

# Render Figure `fig:ood_scaling` (Figure 5)
python figures/make_ood_scaling_figure.py \
    --anchor-json     results/ood_repro.json \
    --retrain-json    results/ood_retrain.json \
    --cross-month-json results/april_cross_month.json \
    --out             figures/ood_scaling.pdf

# Diagnostics + behavior probes (text-level claims in Appendix D, not in tab:ood_scaling)
python -m applications.anchor_calibration.intrinsic_difficulty \
    --checkpoint     checkpoints/tunejury.pt \
    --feat-dir-fm    /path/to/musicarena/features_feb_mar \
    --feat-dir-apr   /path/to/musicarena/features_april \
    --bench-test     data/splits/MusicArena-random_split_bench_clean.json \
    --increment-uuids data/splits/MusicArena_v2_increment.json
python -m applications.anchor_calibration.postcut_diagnostics \
    --checkpoint     checkpoints/tunejury.pt \
    --feat-dir       /path/to/musicarena/features \
    --training-split data/splits/MusicArena-random_split_bench_clean.json \
    --label-csv      data/labels/ma_postcut_scored.csv
python -m applications.anchor_calibration.heldout_margin \
    --checkpoint   checkpoints/tunejury.pt \
    --features_dir data/processed_features \
    --test_ids     data/splits/test.txt

# "No catastrophic forgetting" probe (§A.D): naive-retraining baseline
# trained from scratch on bench_clean (571) ∪ Feb-Mar (598) ∪ April (397),
# evaluated on three held-out partitions (3 seeds). Backs the "+5 pp
# Feb-Mar, +7 pp April, bench-clean within noise" claim in the paper.
python -m applications.anchor_calibration.retrain_forgetting \
    --feat-dir-fm     /path/to/musicarena/features_feb_mar \
    --feat-dir-apr    /path/to/musicarena/features_april \
    --bench-split     data/splits/MusicArena-random_split_bench_clean.json \
    --increment-split data/splits/MusicArena_v2_increment.json \
    --fm-label-csv    data/labels/ma_postcut_scored.csv
```

Expected (with the released `tunejury.pt`, md5 `0524e60900e5ee3c4046e599f613b466`):

Anchor calibration (A row):
- K=0:   `0.551 ± 0.020`
- K=3:   `0.565 ± 0.025`
- K=10:  `0.570 ± 0.025`
- K=30:  `0.581 ± 0.020` (+3.0 pp over K=0)
- K=100: `0.600 ± 0.026` (+4.9 pp over K=0)
- K=250: `0.610 ± 0.016` (saturation, +5.9 pp over K=0)

Retraining from scratch (R row, bench-clean ∪ K added pairs):
- K=0:   `0.556 ± 0.017`
- K=3:   `0.548 ± 0.020`
- K=10:  `0.541 ± 0.023`
- K=30:  `0.561 ± 0.024`
- K=100: `0.556 ± 0.020`
- K=250: `0.570 ± 0.035`

Data-efficiency: anchor at K=10 (0.570) already matches retraining at K=250 (0.570); anchor at K=30 (0.581) exceeds retraining's K=250 saturation, at least ~25× data-efficiency edge over the swept K-grid.

Feb-Mar β_s on April (cross-month, Table `cross_month`):
- K=0 raw on April: `0.640` (released ckpt already adequate on April without calibration)
- K=30 cross-month: `0.611 ± 0.035` (a ~2.9 pp **regression** vs raw: per-system biases from Feb-Mar do not transfer at small K)
- K=598 cross-month: `0.642 ± 0.012` (saturates near raw)
- Within-April 50/50 sanity (n_test=199, 10 seeds): flat across K (raw 0.634, K=200 0.633)

Per-system breakdown reveals month-over-month drift: Sonauto v3 gains +19 pp, Lyria-3-30s +10 pp, Lyria-3-Pro preview +5.5 pp, MusicGen-medium +3.4 pp, Magenta-RT flat; Elevenlabs loses -5.8 pp, ACE-Step -2.0 pp, Sonauto v2 -0.6 pp. Aggregate effect cancels.

### Verifying the released checkpoint

`md5sum checkpoints/tunejury.pt` should print `0524e60900e5ee3c4046e599f613b466`.

**Paper headline numbers reproduce exactly with the released checkpoint:**
- §1 / §3 headline: 0.7086 pairwise accuracy + ECE 0.0339 on 2,035-pair test (`python -m eval.internal`)
- §4.1 per-dataset: AIME 0.6744 / MP 0.7184 / SE 0.9076 / MA 0.8000, with leave-out gains matching paper to 0.001 (`python -m eval.internal_per_dataset`)
- §A.D heldout-margin in-distribution row: mean $|\Delta r|$ = 1.148, median = 0.728, $<$0.5 = 36%, $<$1.0 = 62% on the same 2,035-pair test (`python -m applications.anchor_calibration.heldout_margin`)
- §11 per-system / vocal / popularity probes: AIME ρ = +0.978, MP ρ = +0.964, vocal $-$ instrumental gap +0.441 (Welch t = +24.2 at n=6,120), FMA-Large bottom-decile −1.413 vs top-decile +0.084 (Spearman +0.285 at n=106,401)

---

## 11. Behavior probes (Appendix F)

Full instructions in [`applications/probes/README.md`](../applications/probes/README.md).

```bash
cd applications/probes

# Per-system reward ranking (Figure per_system_test)
python per_system_ranking.py \
    --features_dir ../../data/processed_features \
    --splits_dir   ../../data/splits \
    --meta_root    /path/to/raw_audio_pool \
    --checkpoint   ../../checkpoints/tunejury.pt \
    --out_dir      results/per_system

# Vocal-vs-instrumental discrimination probe
python vocal_discrimination.py \
    --features_dir ../../data/processed_features \
    --meta_root    /path/to/raw_audio_pool \
    --checkpoint   ../../checkpoints/tunejury.pt \
    --out_dir      results/vocal_probe

# Popularity-stratified probe (FMA-Large). Fast path: reuse the released
# CSV from release_scores/. Re-extract path: pass --features_root + --checkpoint.
python popularity_probe.py \
    --scores_csv ../../release_scores/fma_large_scores_rescored.csv \
    --listens_csv /path/to/fma_metadata/track_listens.csv \
    --out_dir results/amateur_pro

# Soundfont sensitivity (MidiCaps with FluidR3 vs TimGM6mb)
python soundfont_sensitivity.py \
    --midi_dir    /path/to/midicaps/midi \
    --sf_fluidr3  /path/to/FluidR3_GM.sf2 \
    --sf_timgm6   /path/to/TimGM6mb.sf2 \
    --checkpoint  ../../checkpoints/tunejury.pt

# External SingMOS-Pro validation (Appendix F external singing-voice MOS).
# Default: reports per-utterance and per-system Spearman from the committed
# summary.json (paper-exact). --reinfer regenerates from raw audio (~5 GB
# SingMOS-Pro download required).
python singmos_validation.py     # paper-exact reproduction
# python singmos_validation.py --reinfer --singmos_root /path/to/SingMOS-Pro

# External SVCC 2025 vocal-technique validation (Appendix F).
# Default: recomputes ANOVA + Welch t from the committed summary.json.
# --reinfer regenerates from raw audio (~11 GB SVCC 2025 download required).
python svcc_validation.py        # paper-exact reproduction
# python svcc_validation.py --reinfer --svcc_root /path/to/svcc2025
```

Expected (Appendix F + behavior probes):
- AIME (n=1,560, 13 systems): Spearman ρ = +0.978
- MusicPrefs (n=252, 7 systems): ρ = +0.964
- Music Arena vocal-requested vs instrumental: +0.977 vs +0.536 (gap +0.441, Welch t=+24.2, n=6,120)
- FMA-Large bottom decile −1.413 vs top decile +0.084 (gap ~1.50, Spearman +0.285, n=106K)
- MidiCaps soundfont swap: paired t=+1.73 (p~0.085), Spearman = +0.69, 33% top-10 overlap (renderer-dependent)
- SingMOS-Pro (n=7,981, 141 systems): per-utterance ρ = +0.19, per-system ρ = +0.44 (Figure `probes` middle panel)
- SVCC 2025 (n=48, 6 vocal techniques): ANOVA F=3.82 p<0.01; Mixed Voice highest mean reward (-0.11), Pharyngeal lowest (-0.70)

---

## 12. Figures

| Paper figure | Source script | Output | Upstream data |
|---|---|---|---|
| Figure 1 (modes pipeline) | TikZ in `sections/05_applications.tex` | inline LaTeX | – |
| Figure 2 (tunejury architecture) | `figures/make_tunejury_arch.py` | `figures/tunejury_arch.pdf` | – |
| Figure 3 (mode1_bon_sweep) | `figures/make_mode1_bon_figure.py` | `figures/mode1_bon_sweep.pdf` | §7 mode1 outputs |
| Figure 4 (`fig:calibration`) | `figures/make_calibration_figure.py` | `figures/fig_calibration.pdf` | §4.2 bins JSON |
| Figure 5 (`fig:ood_scaling`) | `figures/make_ood_scaling_figure.py` | `figures/ood_scaling.pdf` | §10 anchor + retrain + cross-month JSONs |
| Figure (`fig:per_system_test`) | `figures/make_per_system_figure.py` | `figures/per_system_test.pdf` | §11 per-system outputs |
| Figure (`fig:probes`) | `figures/make_probes_figure.py` | `figures/probes.pdf` | §11 probe outputs |
| Figure (`fig:release_distribution`) | `figures/make_release_distribution_figure.py` | `figures/release_distribution.pdf` | §2 release scoring outputs |

All figure scripts read JSON/CSV produced by the eval/probe/anchor-calibration steps above and emit PDF only (no PNG). Output filenames match the `\includegraphics{...}` references in the paper.

---

## 13. Backbone environments

Each application backbone needs its own conda environment to avoid CUDA/torch conflicts:

| Backbone | Conda env | Install |
|---|---|---|
| MusicGen-medium/large | `musicgen` | `pip install audiocraft` |
| AudioLDM2-music | `meanaudio` | `pip install diffusers` |
| SAO-small | `sao` | `pip install stable_audio_tools` |
| TangoFlux | `tangoflux` | `pip install git+https://github.com/declare-lab/TangoFlux` |
| ACE-Step v1.5 Turbo Continuous | `ace_step` | `pip install git+https://github.com/ace-step/ACE-Step` |
| FluxAudio-S (Mode 3) | `meanaudio` | clone `github.com/xiquan-li/MeanAudio` |
| TuneJury scoring + everything else | `tunejury` | `environment.yml` in this repo |
