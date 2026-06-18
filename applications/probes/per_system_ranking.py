"""Per-system reward ranking validation on bench-clean test splits.

Reproduces Appendix F (``Per-System Reward Ranking on Held-Out Test Splits'')
of the paper. For each held-out test pair on the three model-labeled
training datasets, we score the two clips with the released TuneJury
head over cached LAION-CLAP + MERT-v1-330M features, aggregate by
source system, and compute Spearman rank correlation between the
per-system TuneJury mean reward and the per-system win rate.

Headline result (paper Appendix F):
    AIME test split    (1,560 pairs, 13 systems): Spearman rho = +0.978
    MusicPrefs test    (252 pairs,    7 systems): Spearman rho = +0.964
    Music Arena test   (74 pairs)              : too small for stable signal

Data layout
-----------
The probe reads cached features from the released, flat in-repo cache
``data/processed_features/<uuid>.pt`` (each .pt already carries
``clap_a``, ``mert_a``, ``clap_b``, ``mert_b``, ``text_emb``, ``winner``,
and ``source``). The split file selects which UUIDs belong to the test
fold (``data/splits/{AIME,MusicArena,MusicPrefs}-random_split*.json``).

Per-system aggregation requires *system labels* (``model_a``,
``model_b``) that are NOT carried in the released .pt cache. Supply
them via ``--meta_dir``, a directory of one JSON sidecar per UUID with
at minimum the keys ``model_a`` and ``model_b``. The expected layout is

    <meta_dir>/<UUID>.json
        {"model_a": "<system>", "model_b": "<system>", ...}

Reviewers who have the original ``music-ranknet`` preprocessing tree
can pass ``--meta_dir /path/to/music-ranknet/data/processed/<DS>/json``
directly; the script will look for sibling per-dataset subdirectories
under ``--meta_root`` if a single root is preferred.

Example
-------
$ python per_system_ranking.py \\
      --features_dir data/processed_features \\
      --splits_dir   data/splits \\
      --meta_root    /path/to/music-ranknet/data/processed \\
      --checkpoint   checkpoints/tunejury.pt \\
      --out_dir      applications/probes/results/per_system
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury


# (source-label-in-.pt, short-label, split-file-name, min-battles-for-Spearman)
DATASETS = [
    ("MusicArena", "MA", "MusicArena-random_split_bench_clean.json", 3),
    ("AIME", "AIME", "AIME-random_split.json", 20),
    ("MusicPrefs", "MP", "MusicPrefs-random_split.json", 20),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--features_dir",
        default="data/processed_features",
        help="Flat dir of <uuid>.pt feature files "
        "(default: data/processed_features, the in-repo cache).",
    )
    p.add_argument(
        "--splits_dir",
        default="data/splits",
        help="Dir holding <DS>-random_split*.json (default: data/splits).",
    )
    p.add_argument(
        "--meta_root",
        default=None,
        help="Optional root with per-dataset subdirs "
        "<meta_root>/<DS>/json/<uuid>.json carrying model_a/model_b labels. "
        "Used when --meta_dir is not given.",
    )
    p.add_argument(
        "--meta_dir",
        default=None,
        help="Single directory of <uuid>.json sidecars carrying model_a/model_b. "
        "Overrides --meta_root.",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to released TuneJury head checkpoint (tunejury.pt).",
    )
    p.add_argument(
        "--out_dir", default="applications/probes/results/per_system",
    )
    return p.parse_args()


def _meta_path(uuid: str, ds: str, args: argparse.Namespace) -> str | None:
    """Locate the model_a/model_b sidecar for a given UUID, if any."""
    if args.meta_dir is not None:
        candidate = os.path.join(args.meta_dir, f"{uuid}.json")
        return candidate if os.path.exists(candidate) else None
    if args.meta_root is not None:
        candidate = os.path.join(args.meta_root, ds, "json", f"{uuid}.json")
        return candidate if os.path.exists(candidate) else None
    return None


def main() -> None:
    args = _parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.meta_root is None and args.meta_dir is None:
        print(
            "[per_system_ranking] No --meta_root or --meta_dir supplied. "
            "Per-system aggregation requires the model_a/model_b sidecars; "
            "see the module docstring for the expected JSON layout.",
            file=sys.stderr,
        )
        sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = TuneJury(input_dim=2048)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    head.eval().to(device)
    for q in head.parameters():
        q.requires_grad_(False)

    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None

    summary = {}
    for ds, label, split_name, min_battles in DATASETS:
        split_path = os.path.join(args.splits_dir, split_name)
        if not os.path.exists(split_path):
            print(f"[skip] split file not found: {split_path}", file=sys.stderr)
            continue
        split = json.load(open(split_path))
        test_ids = split["test"] if isinstance(split["test"], list) else list(split["test"].keys())
        test_uuids = set(test_ids)
        per_model_r: dict[str, list[float]] = defaultdict(list)
        per_model_w: dict[str, list[int]] = defaultdict(list)
        n_pairs = 0
        n_missing_meta = 0
        for uuid in test_uuids:
            pt_path = os.path.join(args.features_dir, f"{uuid}.pt")
            if not os.path.exists(pt_path):
                continue
            json_path = _meta_path(uuid, ds, args)
            if json_path is None:
                n_missing_meta += 1
                continue
            feats = torch.load(pt_path, map_location="cpu", weights_only=False)
            meta = json.load(open(json_path))
            model_a = meta.get("model_a")
            model_b = meta.get("model_b")
            winner = meta.get("winner", feats.get("winner"))
            if model_a in (None, "unknown") or model_b in (None, "unknown"):
                continue
            text_emb = feats["text_emb"].to(device).float()
            feat_a = torch.cat([feats["clap_a"].to(device).float(),
                                 feats["mert_a"].to(device).float(),
                                 text_emb]).unsqueeze(0)
            feat_b = torch.cat([feats["clap_b"].to(device).float(),
                                 feats["mert_b"].to(device).float(),
                                 text_emb]).unsqueeze(0)
            with torch.no_grad():
                r_a = head(feat_a).item()
                r_b = head(feat_b).item()
            per_model_r[model_a].append(r_a)
            per_model_r[model_b].append(r_b)
            per_model_w[model_a].append(1 if winner == "model_a" else 0)
            per_model_w[model_b].append(1 if winner == "model_b" else 0)
            n_pairs += 1

        rows = [(m, float(np.mean(per_model_r[m])),
                 float(np.mean(per_model_w[m])) if per_model_w[m] else None,
                 len(per_model_w[m]))
                for m in per_model_r]
        rows.sort(key=lambda x: -x[1])

        valid = [(m, tj, wr) for m, tj, wr, nb in rows
                  if wr is not None and wr > 0.01 and nb >= min_battles]
        rho, p = (None, None)
        if spearmanr and len(valid) >= 3:
            rho_obj = spearmanr([v[1] for v in valid], [v[2] for v in valid])
            rho, p = float(rho_obj.correlation), float(rho_obj.pvalue)

        print(f"\n=== {label} TEST ({n_pairs} pairs"
              + (f", {n_missing_meta} missing meta" if n_missing_meta else "")
              + ") ===")
        print(f"{'Model':<28} {'n_clips':>8} {'TJ mean':>10} {'WinRate':>9} {'battles':>8}")
        for m, tj, wr, nb in rows:
            wrs = f"{wr*100:6.2f}%" if wr is not None else "   N/A"
            print(f"  {m:<26} {len(per_model_r[m]):>8} {tj:>+10.4f} {wrs:>9} {nb:>8}")
        if rho is not None:
            print(f"\n  Spearman (n_valid={len(valid)}): rho = {rho:+.4f}  (p = {p:.4g})")
        else:
            print(f"\n  Too few models with non-zero held-out WR (only {len(valid)}); "
                  f"split likely too small for stable per-system signal.")

        summary[label] = {
            "n_pairs": n_pairs,
            "n_missing_meta": n_missing_meta,
            "n_models_total": len(per_model_r),
            "n_models_valid": len(valid),
            "spearman": rho,
            "spearman_p": p,
            "per_model": [{"model": m, "mean_reward": float(tj),
                            "win_rate": wr, "n_clips": len(per_model_r[m]),
                            "n_battles": nb} for m, tj, wr, nb in rows],
        }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved per-dataset summary to {out/'summary.json'}")


if __name__ == "__main__":
    main()
