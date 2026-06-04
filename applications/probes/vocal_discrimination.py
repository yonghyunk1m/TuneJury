"""Vocal-vs-instrumental discrimination probe (paper Section 6, limitation (ii)).

For each cached Music Arena pair with model labels, separate clips by
whether the source pair carried a non-empty ``lyrics`` field (proxy
for vocal generation requests) and compare TuneJury reward
distributions. A weak proxy: vocal-requested clips do not guarantee
the generator actually produced singing, but the prompt distribution
is the closest we can get without per-clip vocal annotations.

Data layout
-----------
Audio + text features are read from the released in-repo cache
``data/processed_features/<uuid>.pt`` (each .pt carries ``clap_a``,
``mert_a``, ``clap_b``, ``mert_b``, ``text_emb``, ``source``). Only
pairs with ``source == "MusicArena"`` are scored.

The per-pair ``lyrics`` field is NOT carried in the .pt cache. Supply
it via ``--meta_dir`` (or ``--meta_root`` with a per-dataset subdir):

    <meta_dir>/<UUID>.json
        {"lyrics": "<requested-lyrics-or-empty-string>", ...}

The Music Arena split file at ``data/splits/MusicArena-random_split_bench_clean.json``
gives the test fold; with no split argument the probe walks every
MusicArena pair under ``--features_dir``.

Example
-------
$ python vocal_discrimination.py \\
      --features_dir data/processed_features \\
      --meta_root    /path/to/music-ranknet/data/processed \\
      --checkpoint   checkpoints/tunejury.pt \\
      --out_dir      applications/probes/results/vocal_probe
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--features_dir", default="data/processed_features",
        help="Flat dir of <uuid>.pt files (default: data/processed_features).",
    )
    p.add_argument(
        "--meta_root", default=None,
        help="Root with <meta_root>/MusicArena/json/<uuid>.json sidecars "
        "carrying the per-pair `lyrics` field. Used if --meta_dir is not given.",
    )
    p.add_argument(
        "--meta_dir", default=None,
        help="Single directory of <uuid>.json sidecars carrying `lyrics`. "
        "Overrides --meta_root.",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--out_dir", default="applications/probes/results/vocal_probe",
    )
    return p.parse_args()


def _meta_path(uuid: str, args: argparse.Namespace) -> str | None:
    if args.meta_dir is not None:
        candidate = os.path.join(args.meta_dir, f"{uuid}.json")
        return candidate if os.path.exists(candidate) else None
    if args.meta_root is not None:
        candidate = os.path.join(args.meta_root, "MusicArena", "json", f"{uuid}.json")
        return candidate if os.path.exists(candidate) else None
    return None


def main() -> None:
    args = _parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.meta_root is None and args.meta_dir is None:
        print(
            "[vocal_discrimination] No --meta_root or --meta_dir supplied. "
            "The vocal/instrumental split requires per-pair `lyrics` sidecars; "
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

    vocal_rewards: list[float] = []
    inst_rewards: list[float] = []
    n_pairs = 0
    n_missing_meta = 0
    for pt_path in sorted(glob.glob(os.path.join(args.features_dir, "*.pt"))):
        feats = torch.load(pt_path, map_location="cpu", weights_only=False)
        if feats.get("source") != "MusicArena":
            continue
        uuid = feats.get("uuid") or Path(pt_path).stem
        json_path = _meta_path(uuid, args)
        if json_path is None:
            n_missing_meta += 1
            continue
        meta = json.load(open(json_path))
        lyrics = meta.get("lyrics") or ""
        has_vocal = isinstance(lyrics, str) and lyrics.strip() != ""

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
        target = vocal_rewards if has_vocal else inst_rewards
        target.append(r_a)
        target.append(r_b)
        n_pairs += 1

    print(f"Music Arena pairs: {n_pairs} (missing meta: {n_missing_meta})")
    if not vocal_rewards or not inst_rewards:
        print("Insufficient data on at least one side (vocal / instrumental); aborting.")
        return
    print(f"vocal clips (has_lyrics):       n={len(vocal_rewards):>5}  "
          f"mean={np.mean(vocal_rewards):+.4f}  std={np.std(vocal_rewards):.4f}")
    print(f"instrumental clips (no lyrics): n={len(inst_rewards):>5}  "
          f"mean={np.mean(inst_rewards):+.4f}  std={np.std(inst_rewards):.4f}")
    gap = float(np.mean(vocal_rewards) - np.mean(inst_rewards))
    print(f"mean gap (vocal - instrumental): {gap:+.4f}")

    try:
        from scipy.stats import mannwhitneyu, ttest_ind
        u, pu = mannwhitneyu(vocal_rewards, inst_rewards, alternative="two-sided")
        t, pt = ttest_ind(vocal_rewards, inst_rewards, equal_var=False)
        print(f"Mann-Whitney U={u:.0f}  p={pu:.4g}")
        print(f"Welch's t={t:+.3f}  p={pt:.4g}")
    except ImportError:
        u, pu, t, pt = None, None, None, None

    with open(out / "summary.json", "w") as f:
        json.dump({
            "n_pairs": n_pairs,
            "n_missing_meta": n_missing_meta,
            "n_vocal_clips": len(vocal_rewards),
            "n_inst_clips": len(inst_rewards),
            "vocal_mean": float(np.mean(vocal_rewards)),
            "inst_mean": float(np.mean(inst_rewards)),
            "vocal_std": float(np.std(vocal_rewards)),
            "inst_std": float(np.std(inst_rewards)),
            "gap": gap,
            "welch_t": float(t) if t is not None else None,
            "welch_p": float(pt) if pt is not None else None,
            "mannwhitney_u": float(u) if u is not None else None,
            "mannwhitney_p": float(pu) if pu is not None else None,
        }, f, indent=2)
    print(f"\nSaved to {out/'summary.json'}")


if __name__ == "__main__":
    main()
