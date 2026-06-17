# Pre-computed TuneJury reward scores

Paper anchor: Appendix §A.I (Released Artifacts and License Interplay) + Table `release_distribution` + Figure `release_distribution`.

The released TuneJury checkpoint (`checkpoints/tunejury.pt`) was applied to seven open-license music collections, producing one track-level reward score per clip across 219,020 clips. This directory holds the seven per-dataset CSVs.

## On the Hugging Face Hub

The same scores are mirrored as a dataset at [`TuneJury/release-scores`](https://huggingface.co/datasets/TuneJury/release-scores), with one config per collection (`mtg_jamendo`, `fma_large`, `mtat`, `openmic`, `midicaps`, `musiccaps`, `sdd`).

```python
from datasets import load_dataset

ds = load_dataset("TuneJury/release-scores", "mtg_jamendo")   # one config per collection
print(ds["train"][0])   # {'track_id': ..., 'reward_score': ..., ...}
```

Pull a single file directly, or the whole dataset via the CLI:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("TuneJury/release-scores", "sdd_scores.csv", repo_type="dataset")
```

```bash
huggingface-cli download TuneJury/release-scores --repo-type dataset --local-dir release_scores_hub
```

## Files

| File | Source (official) | License | Rows | Extra columns | Audio access |
|---|---|---|---:|---|---|
| `mtg_jamendo_scores_rescored.csv` | [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | CC-BY-NC-SA 4.0 | 55,701 | `relative_path` | download via source |
| `fma_large_scores_rescored.csv` | [FMA-Large](https://github.com/mdeff/fma) | per-track (mostly CC) | 106,401 | – | [Kaggle mirror](https://www.kaggle.com/datasets/brunosette/fma-large) |
| `mtat_scores.csv` | [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) | per-track | 25,860 | `audio_path` | download via source |
| `openmic_scores.csv` | [OpenMIC-2018](https://github.com/cosmir/openmic-2018) | CC-BY 4.0 | 20,000 | `audio_path` | [Zenodo](https://zenodo.org/records/1432913) |
| `midicaps_scores.csv` | [MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps) | CC-BY 4.0 | 5,000 | `tempo`, `key`, `duration`, `genre`, `caption` | HF (MIDI, render to audio) |
| `musiccaps_scores.csv` | [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) | CC-BY-SA 4.0 (captions) | 5,352 | `caption`, `audioset_labels` | ▶ [CLAPv2/MusicCaps](https://huggingface.co/datasets/CLAPv2/MusicCaps) |
| `sdd_scores.csv` | [Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset) | captions CC-BY 4.0 / CC0; underlying MTG-Jamendo audio CC-BY-NC-SA | 706 | – | ▶ [renumics/song-describer-dataset](https://huggingface.co/datasets/renumics/song-describer-dataset) |
| **Total** | | | **219,020** | | |

**Source (official)** links the authoritative home for citation and attribution. **Audio access** is where the audio can be obtained or played (▶ plays directly in the Hugging Face dataset viewer). The renumics SDD, CLAPv2 MusicCaps, and the FMA-Large Kaggle entry are community mirrors, not the official release.

The Song Describer Dataset is a captioned subset of MTG-Jamendo: all 706 of its two-minute excerpts come from tracks also scored in the MTG-Jamendo file (the excerpt length differs, so the scores differ).

## Scoring protocol

Every score is one deterministic `tunejury` scorer call per clip, with the text branch fed a 512-d zero vector (the empty-prompt release protocol of paper §3 / §4.2). The two frozen audio encoders see the clip as follows:

* **CLAP audio branch** encodes the centre 10-second window at 48 kHz mono (clips of 10 s or less are encoded whole). The window is fixed at the centre, so the score is reproducible.
* **MERT audio branch** encodes the full track at 24 kHz mono and mean-pools the frame embeddings. Tracks longer than 300 s are encoded in consecutive 300 s segments whose frame means are length-weighted averaged, which bounds peak memory without changing the full-track mean.

`scripts/score_release_collection.py` is the exact code; `scripts/verify_release_scores.py` re-scores a random sample and confirms it reproduces the CSV to floating-point tolerance.

### Reference environment

Scores are deterministic within a pinned environment. The released CSVs were produced with:

```
torch 2.4.0+cu121   torchaudio 2.4.0+cu121   transformers 4.44.0
```

Audio decoding can differ by a small amount across torchaudio/ffmpeg backends, so re-scores on a different stack may differ in the last few digits while preserving rankings.

## Audio sources

The CSVs ship scores, not audio. To reproduce them, fetch each collection from its source and run the scoring script:

* **MTG-Jamendo / FMA-Large / MTAT / OpenMIC**: download from the official hosts (`scripts/download_*.py`); files are verified against the datasets' published checksums where available.
* **MidiCaps**: render the MIDI to audio first with `scripts/render_midicaps.py` (FluidSynth, FluidR3_GM soundfont, md5 `af289497caf8c76d97fdc67ec8409f05`). The score depends on the synthesiser, so a different soundfont will not reproduce these values.
* **MusicCaps**: the dataset card distributes only YouTube ids and timestamps, not audio. The released scores were computed on the community audio mirror [`CLAPv2/MusicCaps`](https://huggingface.co/datasets/CLAPv2/MusicCaps) (10-second clips matching the official timestamps; all 5,352 released ids present). Re-scoring under the old pipeline reproduces the prior scores within a few hundredths on the tracks we spot-checked, confirming the mirror audio matches what the original run used.

## Reproducing the release distribution figure / table

```bash
python figures/make_release_distribution_figure.py \
    --scores-root release_scores \
    --out figures/release_distribution.pdf
```

Paper Table `release_distribution` per-dataset statistics (mean, std, p10, median, p90) are computed from these CSVs.

## Notes on within-dataset structure

Paper §A.I documents reproducible within-dataset effects practitioners should know before treating TuneJury as a generic quality filter:

* **MTG-Jamendo**: genre/mood/instrument tags span ~2.0 reward units.
* **MidiCaps**: tempo and duration are essentially flat (|r|<0.03). Major-mode tracks score reliably above minor-mode (Welch t=7.4, p<10^-3).
* **OpenMIC**: instrument-dominated clips (guitar/piano/ukulele/violin/mandolin) score well above voice/drums/synthesizer clips, a ~0.8 reward-unit gap.
* **FMA-Large**: broadest spread of any released collection.

Practitioners filtering heterogeneous open-license collections should condition on relevant labels rather than treat the unconditional reward as a quality signal.
