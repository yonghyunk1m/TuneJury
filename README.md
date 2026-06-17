<div align="center">

<img src="assets/tunejury_logo.webp" alt="TuneJury logo" width="120" height="120">

# TuneJury

### An Open Metric for Improving Music Generation Preference Alignment

<span style="white-space:nowrap;">[Yonghyun Kim](https://yonghyunk1m.notion.site/)<sup>♯</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Junwon Lee](https://jnwnlee.github.io/)<sup>♭♭</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Haiwen Xia](https://haiwen-xia.github.io/)<sup>♮♮</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Yinghao Ma](https://nicolaus625.github.io/)<sup>♯♯</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Junghyun Koo](https://www.linkedin.com/in/junghyun-koo-525a31251/)<sup>♮</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Koichi Saito](https://www.linkedin.com/in/koichisaito0418/?locale=ja)<sup>♮</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Yuki Mitsufuji](https://www.yukimitsufuji.com/)<sup>♮</sup></span>&nbsp;·
<span style="white-space:nowrap;">[Chris Donahue](https://chrisdonahue.com/)<sup>♭</sup></span>

<sub><sup>♭</sup>Carnegie Mellon University&nbsp;·&nbsp;<sup>♮</sup>Sony AI&nbsp;·&nbsp;<sup>♯</sup>Georgia Tech&nbsp;·&nbsp;<sup>♭♭</sup>KAIST&nbsp;·&nbsp;<sup>♮♮</sup>Peking University&nbsp;·&nbsp;<sup>♯♯</sup>Queen Mary University of London</sub>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2606.17006-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.17006)
[![Project page](https://img.shields.io/badge/🌐_Project_page-yonghyunk1m.github.io%2FTuneJury-0071e3.svg)](https://yonghyunk1m.github.io/TuneJury/)
[![Listening demo](https://img.shields.io/badge/🤗_Space-Listening_demo-FFD21E.svg)](https://huggingface.co/spaces/TuneJury/tune-jury-demo)
[![Interactive scoring](https://img.shields.io/badge/🤗_Space-Interactive_scoring-FFD21E.svg)](https://huggingface.co/spaces/TuneJury/tune-jury)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

</div>

---

> **TL;DR.** TuneJury is an **open, instance-level pairwise reward model** for text-to-music. The released checkpoint is a 2.8M-parameter MLP head over frozen LAION-CLAP-Music + MERT-v1-330M embeddings, trained on ~17.5K human-rated A vs. B pairs from four open sources **without pseudo-label augmentation**. It scores a single (text, audio) pair, drives Mode 1 best-of-N selection, Mode 2 latent optimization (DITTO-style), and Mode 3 expert-iteration post-training, and includes **anchor calibration** to recover ~5 pp of OOD agreement on newly released generators with ~100 calibration pairs and no retraining.

## Highlights

|  |  |
|---|---|
| **Pairwise accuracy** | 0.7086 on 2,035 held-out non-tie pairs (expected calibration error 0.0339) |
| **Training pairs** | ~17.5K human-rated training pairs from four open sources (Music Arena · MusicPrefs · AIME · SongEval) |
| **Parameters (head only)** | 2.8 M MLP over frozen 2,048-d (CLAP audio 512 + CLAP text 512 + MERT 1,024) |
| **Anchor calibration** | ~5 pp recovery on the Feb-Mar post-cutoff Music Arena slice with ~100 calibration pairs (no retraining) |
| **Mode 1 best-of-N** | Monotone reward gain through N=32 across MusicGen-medium/large, AudioLDM2-music, and ACE-Step Turbo Continuous |
| **Mode 2 latent optimization (DITTO)** | +0.25 on SAO-small (n=30) and +1.56 on TangoFlux (n=100) |
| **Mode 3 expert iteration** | Mean reward lift +0.17 / +0.37 / +0.42 across LR 1e-6 / 5e-6 / 1e-5 (3-point sweep); aggressive end at 75/100 prompts trades fidelity for reward |

## Demos

<table>
<tr>
<td width="33%" align="center">

[🌐 **Project page**](https://yonghyunk1m.github.io/TuneJury/)

Landing page with quick-start, applications summary, paper highlights.

</td>
<td width="33%" align="center">

[🎧 **Listening demo**](https://huggingface.co/spaces/TuneJury/tune-jury-demo)

Side-by-side Mode 1/2/3 audio examples with reward scores.

</td>
<td width="33%" align="center">

[🎚️ **Interactive scoring**](https://huggingface.co/spaces/TuneJury/tune-jury)

Upload audio (or video) + optional text → reward + position vs. seven dataset distributions.

</td>
</tr>
</table>

---

## Installation

```bash
git clone https://github.com/yonghyunk1m/TuneJury
cd TuneJury
conda env create -f environment.yml   # creates env "tunejury"
conda activate tunejury
huggingface-cli login                 # MERT-v1-330M is gated
```

`ffmpeg` and `libsndfile1` are required system-level. The frozen LAION-CLAP-Music encoder (~2.2 GB) auto-downloads on first use; the TuneJury head ships both in this repo (`checkpoints/`) and on the Hub ([`TuneJury/tunejury`](https://huggingface.co/TuneJury/tunejury)). Mode 1/2/3 backbones each need their own conda env (pin conflicts); see [`docs/reproducing.md` §0](docs/reproducing.md#0-prerequisites-stop-and-read-this-first) for the full prerequisite checklist (system deps, gated-model access, external downloads, per-Mode envs, compute).

## Quick start

After installation, score a clip with the bundled checkpoint or pull it from the Hub:

```python
from tunejury import Scorer

# The head ships in this repo (checkpoints/); or pull it from the Hub:
#   from huggingface_hub import hf_hub_download
#   ckpt = hf_hub_download("TuneJury/tunejury", "tunejury.pt")
scorer = Scorer.from_pretrained("checkpoints/tunejury.pt")
reward = scorer.score("clip.wav", prompt="a calm piano piece")  # prompt="" → empty-prompt (OOD-safe, §4.2)
print(reward)
```

For prompt distributions far from arena-style requests (e.g. PAM-style post-hoc captions) we recommend the **empty-prompt protocol** (Section 3 of the paper):

```python
reward = scorer.score("clip.wav", prompt="")
```

## Repository layout

```
tunejury/         Scorer API, MLP head, training loop
applications/     Mode 1 BoN / Mode 2 DITTO / Mode 3 expert iter / anchor calibration / probes
eval/             CMI-RewardBench, FAD-CLAP, CLAP score
checkpoints/      Released weights (download per checkpoints/README.md)
release_scores/   Pre-computed reward scores across 7 open-license collections (§A.H)
docs/             reproducing.md + project page
```

## Released checkpoints

The primary release is `tunejury.pt`, the 2048-d CLAP+MERT variant that backs every number in Sections 4–5 and the appendix.

Abbreviations: **MA** = Music Arena · **MP** = MusicPrefs · **AIME** (no shorter form) · **SE** = SongEval.

| Checkpoint                       | Training mix                       | Purpose                                |
|----------------------------------|------------------------------------|----------------------------------------|
| `tunejury.pt`                    | MA + MP + AIME + SE (bench-clean)  | **Primary release**                    |
| `tunejury_leave_MA.pt`           | MP + AIME + SE                     | Leave-one-out (fair-eval on bench MA)  |
| `tunejury_leave_MP.pt`           | MA + AIME + SE                     | Leave-one-out                          |
| `tunejury_leave_AIME.pt`         | MA + MP + SE                       | Leave-one-out                          |
| `tunejury_leave_SE.pt`           | MA + MP + AIME                     | Leave-one-out                          |
| `tunejury_leave_SE_MA.pt`        | MP + AIME                          | Leave-two-out                       |
| `tunejury_leave_MP_MA.pt`        | AIME + SE                          | Leave-two-out                       |
| `tunejury_muq_leave_MA.pt`       | MP + AIME + SE on MuQ-MuLan-large  | Encoder-swap probe (§A.D)              |

All checkpoints are released under CC-BY-NC 4.0, tracking the MERT-v1-330M upstream license. They are bundled in this repo and also on the Hugging Face Hub at [`TuneJury/tunejury`](https://huggingface.co/TuneJury/tunejury) (e.g. `Scorer.from_pretrained(hf_hub_download("TuneJury/tunejury", "tunejury.pt"))`). Download instructions in [`checkpoints/README.md`](checkpoints/README.md).

## Reproducing paper results

See [`docs/reproducing.md`](docs/reproducing.md) for the end-to-end pipeline. Headline numbers:

| Section | Result | How to reproduce |
|---|---|---|
| §4.1 (Calibration) | 0.7086 pairwise accuracy, ECE 0.0339 on 2,035 non-tie test pairs | `python -m eval.internal --checkpoint checkpoints/tunejury.pt --features-dir data/processed_features --test-ids data/splits/test.txt` |
| §4.2 / Table `head_to_head` | Head-to-head against PAM score, Audiobox-Aesthetics, SongEval-RM, CMI-RM, and MuQ-Eval on CMI-RewardBench | `python -m eval.cmi_rewardbench --checkpoint checkpoints/tunejury.pt --bench-root /path/to/CMI-RewardBench --prompt-protocol with`<br/>MuQ-Eval baseline: `python applications/baselines/muqeval_a1.py` |
| §5.1 / Fig. 3 | Mode 1 best-of-N sweep on 4 backbones (N ∈ {1, 2, 4, 8, 16, 32}) | see [`applications/mode1_bon/README.md`](applications/mode1_bon/README.md) |
| §5.2 / Table 5 (top) | Mode 2 DITTO: +0.25 on SAO-small, +1.56 on TangoFlux | `python applications/mode2_ditto/sao_small.py` / `tangoflux_ditto.py` |
| §5.3 / Table 5 (bottom) | Mode 3 expert iter: ΔRwd +0.17 / +0.37 / +0.42 across LR 1e-6 / 5e-6 / 1e-5; ΔMAD +0.293 / +0.284 / +0.669 against SDD-706 | `python applications/mode3_expert_iter/run.py` |
| §6 / §A.D | Anchor calibration: ~100 pairs recover ~5 pp OOD agreement; K=10 anchor matches K=250 retrain (~25× data-efficient on the swept K grid) | `python applications/anchor_calibration/run_experiment.py` |
| §A.E (probes) | Per-system ρ (AIME +0.98, MusicPrefs +0.96), vocal-vs-instrumental gap, popularity probe | see [`applications/probes/README.md`](applications/probes/README.md) |

## Released reward scores (~219K tracks across 7 datasets)

Pre-computed TuneJury reward scores for seven open-license collections are included in [`release_scores/`](release_scores/) (§A.H). Per-dataset CSVs total 22 MB. They are also on the Hugging Face Hub as [`TuneJury/release-scores`](https://huggingface.co/datasets/TuneJury/release-scores), one config per collection: `load_dataset("TuneJury/release-scores", "<config>")`.

| File | Source | Rows |
|---|---|---:|
| `mtg_jamendo_scores_rescored.csv` | MTG-Jamendo | 55,701 |
| `fma_large_scores.csv` | FMA-Large | 106,401 |
| `mtat_scores.csv` | MagnaTagATune | 25,860 |
| `openmic_scores.csv` | OpenMIC-2018 | 20,000 |
| `midicaps_scores.csv` | MidiCaps (symbolic, FluidR3_GM rendering) | 5,000 |
| `musiccaps_scores.csv` | MusicCaps | 5,352 |
| `sdd_scores.csv` | Song Describer Dataset | 706 |
| **Total** | | **219,020** |

Each track is scored end-to-end with an empty text branch under a uniform protocol. See [`release_scores/README.md`](release_scores/README.md) for the protocol note (Section 3 zero-vector vs. earlier CLAP("") encoding) and within-dataset structure caveats.

```bash
python figures/make_release_distribution_figure.py \
    --scores-root release_scores \
    --out figures/release_distribution.pdf
```

## Inputs (CLAP + MERT)

At inference, TuneJury calls the LAION-CLAP-Music encoder (CC0 1.0) and the MERT-v1-330M encoder (CC-BY-NC 4.0). The TuneJury head adds no further license restriction, but commercial use of the primary CLAP+MERT release must respect the non-commercial constraint inherited from MERT. A CLAP-only Apache-2.0 variant (Row A1 in the input ablation, §A.C) is also released for commercial use cases.

The encoders are loaded automatically on first use:

- **LAION-CLAP-Music**: downloaded from HuggingFace if missing (`music_audioset_epoch_15_esc_90.14.pt`, ~2.2 GB).
- **MERT-v1-330M**: loaded via `transformers.AutoModel.from_pretrained("m-a-p/MERT-v1-330M")`.

**MuQ-MuLan-large** is supported as an alternative encoder for the encoder-swap variant. See `applications/` and `docs/reproducing.md`.

## Citation

```bibtex
@misc{tunejury2026,
  title         = {TuneJury: An Open Metric for Improving Music Generation Preference Alignment},
  author        = {Kim, Yonghyun and Lee, Junwon and Xia, Haiwen and
                   Ma, Yinghao and Koo, Junghyun and Saito, Koichi and
                   Mitsufuji, Yuki and Donahue, Chris},
  year          = {2026},
  eprint        = {2606.17006},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2606.17006},
}
```

## Acknowledgements

This work uses [LAION-CLAP](https://github.com/LAION-AI/CLAP) (CC0 1.0), [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) (CC-BY-NC 4.0), and [MuQ](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) (CC-BY-NC 4.0) as frozen encoders. Training pairs are sourced from [Music Arena](https://arxiv.org/abs/2507.20900), [MusicPrefs](https://arxiv.org/abs/2503.16669), [AIME](https://arxiv.org/abs/2506.19085), and [SongEval](https://arxiv.org/abs/2505.10793). Mode 2 DITTO follows the formulation of [DITTO](https://arxiv.org/abs/2401.12179).

## License

The primary release (`tunejury.pt`, 2048-d CLAP+MERT) is CC-BY-NC 4.0, tracking the strictest upstream constraint (MERT-v1-330M weights). A commercial-friendly **Apache 2.0 CLAP-only variant** (Row A1 in §A.C, 0.705 overall accuracy, tied with the seed-matched A7 retrain and within single-seed noise of the released checkpoint's 0.7086) is **also released** for downstream users who require an Apache-2.0 stack. See [LICENSE](LICENSE).
