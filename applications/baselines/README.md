# Baselines (head-to-head reward-model comparisons)

Standalone scorers reimplemented or loaded from public checkpoints, then
benchmarked against TuneJury on the same CMI-RewardBench test splits.

## Scripts

| Script | Paper anchor | Headline |
| --- | --- | --- |
| `muqeval_a1.py` | Section 4.2, Table `head_to_head` (row `MuQ-Eval-A1`) | Concurrent MuQ-Eval (Zhu & Li 2026, arXiv:2603.22677) A1 checkpoint on the four CMI-RewardBench test splits: PAM SRCC $0.4995$, MusicEval SRCC $0.8089$ (in-distribution, italicized in paper), CMI-Pref pairwise $0.6600$, Music Arena pairwise $0.6761$. TuneJury (T+A) leads on three of four splits ($+0.110$ PAM SRCC, $+5.4$ pp CMI-Pref, $+4.3$ pp Music Arena). |

> **Paper-exact reproduction default.** Running `python applications/baselines/muqeval_a1.py`
> (no args) loads the committed `results/muqeval_a1/summary.json` and prints the
> four split-level numbers — paper Table values reproduce exactly every run.
>
> **Fresh re-inference.** Pass `--reinfer --bench_root <CMI-RewardBench/data>`
> to clone the upstream MuQ-Eval repo into `--muqeval_repo` (default
> `/tmp/MuQ-Eval`), download the `zhudi2825/MuQ-Eval-A1` weights from
> HuggingFace, and re-score every clip end-to-end. Per-clip MuQ-Eval scoring
> reads each `audio-path` (and `audio2` for pairwise sources) from
> `bench_root/all_test.jsonl` and resamples to MuQ-Eval's 24 kHz mono
> 10-second protocol. Fresh re-inference is bf16-precision deterministic
> within $\pm 0.005$ SRCC on the per-clip splits and within $\pm 0.5$ pp on
> the pairwise splits across our observed runs.
>
> **In-distribution disclaimer.** MuQ-Eval is trained on MusicEval, so the
> MusicEval cell at $0.8089$ is in-distribution (italicized in the paper
> Table, excluded from OOD ranking).

## External prerequisites

The MuQ-Eval baseline runner needs three external resources:

| Resource | Where to get it |
| --- | --- |
| CMI-RewardBench bench root (`all_test.jsonl` + audio subdirs) | The CMI-RewardBench release at https://github.com/cmi-rewardbench/cmi-rewardbench (we used the directory layout described in their README; the script reads `<bench_root>/all_test.jsonl` and resolves every `audio-path` and `audio2` field relative to `<bench_root>`). |
| MuQ-Eval source code (model definitions) | `git clone https://github.com/dgtql/MuQ-Eval.git <muqeval_repo>` happens automatically the first time you pass `--reinfer`. |
| MuQ-Eval-A1 trained weights | `pip install huggingface_hub` and the script auto-downloads `zhudi2825/MuQ-Eval-A1` (`config.yaml` + `model_state_dict.pt`). |

Additionally, `pip install muq peft omegaconf` is required for the upstream
MuQ encoder, LoRA adapters (unused at inference but pulled by the import
graph), and the YAML config loader.

## Reproducing

```bash
# Paper-exact (default; reads committed summary.json):
python applications/baselines/muqeval_a1.py

# Fresh re-inference from raw audio:
python applications/baselines/muqeval_a1.py \
    --reinfer --bench_root /path/to/CMI-RewardBench/data \
    --device cuda:0
```

The script writes `results/muqeval_a1/summary.json` (per-split aggregates plus
the full per-clip scores) on every `--reinfer` run and prints the four
head-to-head numbers to stdout.
