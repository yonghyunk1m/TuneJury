# `data/processed_features/`

This directory holds the per-pair LAION-CLAP + MERT feature blobs that
`tunejury.train`, `eval/sanity.py`, and every internal-evaluation script
load (one `.pt` per pair, keyed by `pair_id` / `battle_uuid`). On a
fresh clone it is intentionally empty: the underlying audio is governed
by four upstream licenses (Music Arena, MusicPrefs, AIME, SongEval) and
is not redistributable from this repo.

To reproduce the paper's headline numbers (test pairwise 0.7086,
ECE 0.0339 on the 2,035-pair non-tie held-out test set; per-dataset
0.674 / 0.718 / 0.800 / 0.908) you must populate this directory
yourself, in one of two ways:

## Option A: Re-extract from raw audio (canonical)

End-to-end pipeline documented in
[`docs/reproducing.md` §2 "Data preparation"](../../docs/reproducing.md#2-data-preparation):

1. Download raw audio + label files for all four upstream sources
   (URLs and licenses in §2 of `docs/reproducing.md`).
2. (If you don't already have it) synthesize SongEval pair labels
   via `data/prepare_songeval_pairs.py`.
3. (Optional) filter Music Arena against the CMI-RewardBench
   `battle_uuid` list via `data/filter_music_arena_bench_clean.py`
   to guarantee item-level disjointness with the external test.
4. Run `data/extract_features.py --pairs data/splits/all_pairs.json
   --audio-root <path> --out-dir data/processed_features
   --encoders clap mert`.

The output schema (see `data/extract_features.py:218-247` and
`tunejury/dataset.py`) is one `.pt` per pair containing
`{text_emb, clap_a, clap_b, mert_a, mert_b, flag, winner, source,
prompt, uuid}`.

Peak GPU memory is ~6 GB (MERT-v1-330M dominates). Total
disk footprint after extraction is ~480 MB (`.pt` blobs are ~20 KB
each).

## Option B: Download a pre-extracted release

If/when a pre-extracted feature archive is published on Hugging Face
Datasets or Zenodo, drop the `.pt` blobs directly into this directory.
The naming convention is `{pair_id}.pt`, and the schema must match
`tunejury.dataset.TuneJuryDataset.__getitem__` (see top of
`tunejury/dataset.py`). As of this commit, no public mirror is
hosted; Option A is the canonical path.

## Sanity check

After population:

```bash
python - <<'PY'
from pathlib import Path
n = sum(1 for _ in Path("data/processed_features").glob("*.pt"))
print(f"Found {n} .pt blobs")
assert n > 0, "data/processed_features/ is empty; see this README"
PY
```

A correctly populated directory holds 24,935 feature blobs; the train/val/test splits cover 21,800 (17,554 train +
2,111 val + 2,135 test). `tunejury/dataset.py` raises
a `FileNotFoundError` with a pointer to this README if every split id
resolves to a missing file, so a fresh clone will fail loudly rather
than silently train on zero pairs.
