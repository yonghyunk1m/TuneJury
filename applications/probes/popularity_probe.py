"""Popularity-stratified TuneJury reward probe on FMA-Large.

FMA-Large includes per-track ``original_listens`` (track play counts on
the Free Music Archive platform). This is a noisy popularity signal,
not a clean amateur-vs-professional label: track age, genre
popularity, fan base, FMA recommendation surfacing, and external
linking all contribute to play counts independently of production
quality. We therefore bucket tracks by listens percentile and report
per-bucket TuneJury reward distributions plus the Spearman
correlation between log-listens and TuneJury reward.

A monotone gradient in the per-bucket means is consistent with the
``TuneJury captures a population-level quality signal''
interpretation; the absolute correlation is moderate because the
above confounds dilute the quality component of listens. The probe
does not validate per-track amateur-vs-professional discrimination.

Data layout
-----------
This probe needs *per-track* (single-clip) features for the FMA-Large
corpus (~106K tracks). The released TuneJury repo only ships paired
preference features (``data/processed_features/``) and does NOT carry
the FMA-Large single-clip cache. Two ways to reproduce:

(1) Score CSV (fast path, default): pass
        --scores_csv release_scores/fma_large_scores.csv
    which carries pre-computed per-track TuneJury rewards
    (``track_id,reward_score``). Join with FMA-Large's ``original_listens``
    from ``--listens_csv`` (a ``track_id,listens`` two-column CSV
    derived from ``fma_metadata/tracks.csv``).

(2) Re-extract from .pt cache: pass
        --features_root /path/to/music-ranknet/data/processed
        --listens_csv   /path/to/fma_listens.csv
    The probe will walk
    ``<features_root>/FMA_Scoring/features/<track_id>.pt`` and rescore
    each clip with the released checkpoint. This requires the external
    music-ranknet preprocessing tree.

Example
-------
$ python popularity_probe.py \\
      --scores_csv  release_scores/fma_large_scores.csv \\
      --listens_csv path/to/fma_listens.csv \\
      --out_dir     applications/probes/results/popularity_probe
"""
from __future__ import annotations

import argparse
import csv
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
    # Fast path: read pre-computed scores from the released CSV.
    p.add_argument(
        "--scores_csv", default=None,
        help="Path to release_scores/fma_large_scores.csv (track_id,reward_score). "
        "If supplied, the probe uses these pre-computed scores and skips "
        "feature extraction.",
    )
    # Re-extract path: walk an external FMA_Scoring feature cache.
    p.add_argument(
        "--features_root", default=None,
        help="Optional. Root of music-ranknet processed data tree containing "
        "FMA_Scoring/features/<track_id>.pt. Used only if --scores_csv is not given.",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Required when --scores_csv is not given. Path to tunejury.pt.",
    )
    p.add_argument(
        "--listens_csv", required=True,
        help="CSV with two columns: track_id,listens. Derive from FMA-Large "
        "fma_metadata/tracks.csv (col `track_listens`).",
    )
    p.add_argument(
        "--out_dir", default="applications/probes/results/popularity_probe",
    )
    return p.parse_args()


def _load_listens(path: str) -> dict[str, int]:
    listens: dict[str, int] = {}
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return listens
        # detect header vs. data row
        if not header[0].lstrip("-").isdigit() and header[0].lower() not in {"track_id", "id"}:
            # header was actually a data row; rewind by replaying it as data
            try:
                listens[header[0]] = int(float(header[1]))
            except (ValueError, IndexError):
                pass
        for row in reader:
            if len(row) < 2:
                continue
            try:
                listens[row[0]] = int(float(row[1]))
            except ValueError:
                continue
    return listens


def _scores_from_csv(path: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with open(path) as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                scores[row[0]] = float(row[1])
            except ValueError:
                continue
    return scores


def _scores_from_features(args: argparse.Namespace) -> dict[str, float]:
    if not args.checkpoint:
        raise SystemExit(
            "[popularity_probe] --checkpoint is required when --scores_csv is not given."
        )
    if not args.features_root:
        raise SystemExit(
            "[popularity_probe] Provide either --scores_csv (fast path) or "
            "--features_root with an FMA_Scoring/features/ subdir."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = TuneJury(input_dim=2048)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    head.eval().to(device)
    for q in head.parameters():
        q.requires_grad_(False)

    feat_dir = os.path.join(args.features_root, "FMA_Scoring", "features")
    scores: dict[str, float] = {}
    for pt_path in sorted(glob.glob(os.path.join(feat_dir, "*.pt"))):
        feats = torch.load(pt_path, map_location="cpu", weights_only=False)
        track_id = feats.get("uuid") or feats.get("track_id") or Path(pt_path).stem
        text_emb = feats["text_emb"].to(device).float() if "text_emb" in feats \
            else torch.zeros(512, device=device)
        clap = feats.get("clap") if "clap" in feats else feats.get("clap_a")
        mert = feats.get("mert") if "mert" in feats else feats.get("mert_a")
        if clap is None or mert is None:
            continue
        feat = torch.cat([clap.to(device).float(),
                           mert.to(device).float(),
                           text_emb]).unsqueeze(0)
        with torch.no_grad():
            scores[track_id] = float(head(feat).item())
    return scores


def main() -> None:
    args = _parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.scores_csv:
        scores = _scores_from_csv(args.scores_csv)
        print(f"[popularity_probe] Loaded {len(scores)} scores from {args.scores_csv}")
    else:
        scores = _scores_from_features(args)
        print(f"[popularity_probe] Re-extracted {len(scores)} scores from features.")

    listens_map = _load_listens(args.listens_csv)
    print(f"[popularity_probe] Loaded {len(listens_map)} listens rows from {args.listens_csv}")

    rows = []
    skipped = 0
    for tid, reward in scores.items():
        listens = listens_map.get(tid)
        if listens is None or listens < 0:
            skipped += 1
            continue
        rows.append({"track_id": tid, "listens": int(listens), "reward": float(reward)})

    print(f"Scored {len(rows)} FMA-Large tracks (skipped {skipped})")
    if not rows:
        return

    rewards = np.array([r["reward"] for r in rows])
    listens = np.array([r["listens"] for r in rows])
    log_listens = np.log1p(listens)

    # Per-percentile bucket
    pcts = [0, 10, 25, 50, 75, 90, 100]
    edges = np.percentile(log_listens, pcts)
    print(f"\nLog-listens percentile edges: {[f'{e:.2f}' for e in edges]}")
    print(f"{'Bucket':<14} {'n':>6} {'listens (median)':>18} {'TJ mean':>10}")
    bucket_summary = []
    for i in range(len(pcts) - 1):
        mask = (log_listens >= edges[i]) & (log_listens <= edges[i + 1])
        if not mask.any():
            continue
        b_reward = rewards[mask]
        b_listens = listens[mask]
        label = f"P{pcts[i]:>3}-P{pcts[i+1]:>3}"
        print(f"  {label:<12} {int(mask.sum()):>6} "
              f"{int(np.median(b_listens)):>18} {np.mean(b_reward):>+10.4f}")
        bucket_summary.append({
            "bucket": label, "n": int(mask.sum()),
            "median_listens": int(np.median(b_listens)),
            "mean_reward": float(np.mean(b_reward)),
        })

    try:
        from scipy.stats import spearmanr, pearsonr
        rho, p_rho = spearmanr(log_listens, rewards)
        r, p_r = pearsonr(log_listens, rewards)
        print(f"\nSpearman(log_listens, reward) = {rho:+.4f}  (p={p_rho:.4g})")
        print(f"Pearson(log_listens, reward)  = {r:+.4f}  (p={p_r:.4g})")
    except ImportError:
        rho, p_rho, r, p_r = None, None, None, None

    summary = {
        "n_tracks": len(rows),
        "skipped": skipped,
        "spearman": float(rho) if rho is not None else None,
        "spearman_p": float(p_rho) if p_rho is not None else None,
        "pearson": float(r) if r is not None else None,
        "pearson_p": float(p_r) if p_r is not None else None,
        "buckets": bucket_summary,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out/'summary.json'}")

    # Save raw (listens, rewards) for figure scatter overlay (Figure probes panel c)
    raw_out = {
        "n": len(rows),
        "listens": listens.tolist(),
        "rewards": rewards.tolist(),
    }
    with open(out / "raw_fma.json", "w") as f:
        json.dump(raw_out, f)
    print(f"Saved to {out/'raw_fma.json'}")


if __name__ == "__main__":
    main()
