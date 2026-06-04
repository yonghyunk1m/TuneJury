"""Compute |s(A)-s(B)| percentile distribution over the TuneJury training pairs.

Used by the Appendix A reliability figure (`figures/make_calibration_figure.py`)
to position the two vertical dashed reference lines (training $|m|_{p_{95}}$,
$|m|_{p_{99}}$) on panel (b). The released `checkpoints/tunejury.pt` yields
pooled p95 = 4.60, p99 = 6.82, max = 9.35 over 17,554 training pairs.

Example
-------
$ python eval/compute_train_margins.py \\
      --checkpoint   checkpoints/tunejury.pt \\
      --features-dir data/processed_features \\
      --splits-dir   data/splits \\
      --out          results/train_margins.json

The four split files
(``MusicArena-random_split_bench_clean.json``, ``MusicPrefs-random_split.json``,
``AIME-random_split.json``, ``SongEval-balanced_split.json``) determine which
UUIDs go into the training pool. Each split's ``train`` field is read; the
corresponding ``data/processed_features/<uuid>.pt`` blob carries
pre-extracted ``clap_a / mert_a / clap_b / mert_b / text_emb`` features.
"""
from __future__ import annotations

import os

# Match the deterministic protocol used in `figures/make_calibration_figure.py`
# and `eval/internal.py`: cuDNN deterministic + manual seed before any CUDA op.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6 changed `torch.load` default to weights_only=True; the released
# LAION-CLAP checkpoint that the head was trained against needs the legacy path.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})  # type: ignore

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunejury.model import TuneJury

SPLIT_FILES = {
    "MA":   "MusicArena-random_split_bench_clean",
    "MP":   "MusicPrefs-random_split",
    "AIME": "AIME-random_split",
    "SE":   "SongEval-balanced_split",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/tunejury.pt")
    ap.add_argument("--features-dir", default="data/processed_features", type=Path)
    ap.add_argument("--splits-dir", default="data/splits", type=Path)
    ap.add_argument("--out", default="results/train_margins.json", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    head = TuneJury(input_dim=2048)
    state = torch.load(args.checkpoint, map_location="cpu")
    head.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    head.eval().to(args.device)
    for p in head.parameters():
        p.requires_grad_(False)

    per_dataset: dict[str, dict] = {}
    pooled: list[float] = []

    for ds_name, split_file in SPLIT_FILES.items():
        split_path = args.splits_dir / f"{split_file}.json"
        if not split_path.exists():
            print(f"[skip] {split_path} missing"); continue
        split = json.loads(split_path.read_text())
        train_uuids = split["train"] if isinstance(split["train"], list) else list(split["train"].keys())
        margins: list[float] = []
        for uuid in train_uuids:
            pt_path = args.features_dir / f"{uuid}.pt"
            if not pt_path.exists():
                continue
            feats = torch.load(pt_path, map_location="cpu")
            ca, ma, cb, mb = feats["clap_a"], feats["mert_a"], feats["clap_b"], feats["mert_b"]
            te = feats.get("text_emb", torch.zeros(512))
            if te is None or te.numel() == 0:
                te = torch.zeros(512)
            x_a = torch.cat([ca, ma, te]).unsqueeze(0).to(args.device)
            x_b = torch.cat([cb, mb, te]).unsqueeze(0).to(args.device)
            with torch.no_grad():
                s_a = head(x_a).item()
                s_b = head(x_b).item()
            margins.append(abs(s_a - s_b))
        pooled.extend(margins)
        if margins:
            arr = np.array(margins)
            per_dataset[ds_name] = {
                "n": len(margins),
                "max": float(arr.max()),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
            print(f"  {ds_name}: n={len(margins)}, max={arr.max():.4f}, "
                  f"p95={np.percentile(arr, 95):.4f}, p99={np.percentile(arr, 99):.4f}")

    pooled_arr = np.array(pooled)
    out_obj = {
        "checkpoint": args.checkpoint,
        "n_training_pairs": len(pooled),
        "pooled": {
            "p95": float(np.percentile(pooled_arr, 95)),
            "p99": float(np.percentile(pooled_arr, 99)),
            "max": float(pooled_arr.max()),
        },
        "per_dataset": per_dataset,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2))
    print(f"\nWrote {args.out}")
    print(json.dumps(out_obj["pooled"], indent=2))


if __name__ == "__main__":
    main()
