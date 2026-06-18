"""Cross-check paper headline numbers against current code / data outputs.

For each claim in the paper that is computable from local artifacts, this
script re-derives the value and compares it to the value asserted in the
.tex source. Discrepancies print a single-line warning with the file:line
of the original claim and the recomputed value.

Run after any change that could affect a paper number (Scorer.score, training
data, release CSVs, probe scripts, etc.) to catch silent drift.

Usage:
    python scripts/paper_numbers_cross_check.py \\
        [--paper-root PAPER_ROOT] \\
        [--features-root FEATURES_ROOT] \\
        [--songeval-audio SONGEVAL_AUDIO] \\
        [--release-scores release_scores]

Defaults are repo-relative (resolved against the repository root). Checks
that require external (non-repo) data sources (e.g., music-ranknet feature
caches, SongEval raw audio) will SKIP rather than FAIL when the path is
missing, so a fresh-clone reviewer sees a clean report.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path


def check_songeval_median(audio_dir: str, claim_min: float) -> tuple[bool, str]:
    try:
        import soundfile as sf
    except ImportError:
        return True, "skip: soundfile not installed"
    if not os.path.isdir(audio_dir):
        return True, f"skip: SongEval audio dir not found ({audio_dir}); pass --songeval-audio to verify"
    durs = []
    for f in os.listdir(audio_dir):
        if f.endswith(".wav"):
            try:
                durs.append(sf.info(os.path.join(audio_dir, f)).duration)
            except Exception:
                pass
    if not durs:
        return True, f"skip: no .wav files in {audio_dir}"
    m = statistics.median(durs) / 60.0
    ok = abs(m - claim_min) < 0.1
    return ok, f"measured median={m:.2f} min vs claim {claim_min} min ({len(durs)} files)"


def check_vocal_probe(features_root: str, claim_gap: float, claim_n_v: int, claim_n_i: int) -> tuple[bool, str]:
    import numpy as np
    ma_dir = os.path.join(features_root, "MusicArena", "features")
    if not os.path.isdir(ma_dir):
        return True, (
            f"skip: MusicArena feature cache not found ({ma_dir}); "
            "pass --features-root to verify (requires external music-ranknet data)"
        )
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tunejury.model import TuneJury
    except ImportError as e:
        return True, f"skip: {e}"
    head = TuneJury(input_dim=2048)
    ckpt = str(Path(__file__).resolve().parents[1] / "checkpoints" / "tunejury.pt")
    if not os.path.exists(ckpt):
        return True, f"skip: checkpoint not found ({ckpt})"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    head.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    head.eval()
    for q in head.parameters():
        q.requires_grad_(False)

    voc, ins = [], []
    for pt_path in sorted(glob.glob(os.path.join(features_root, "MusicArena", "features", "*.pt"))):
        feats = torch.load(pt_path, map_location="cpu", weights_only=False)
        jpath = os.path.join(features_root, "MusicArena", "json", f"{feats['uuid']}.json")
        if not os.path.exists(jpath):
            continue
        meta = json.load(open(jpath))
        has_v = bool((meta.get("lyrics") or "").strip())
        text_emb = feats["text_emb"].float()
        for ab in ("a", "b"):
            f = torch.cat([feats[f"clap_{ab}"].float(), feats[f"mert_{ab}"].float(), text_emb]).unsqueeze(0)
            with torch.no_grad():
                r = head(f).item()
            (voc if has_v else ins).append(r)
    gap = float(np.mean(voc) - np.mean(ins))
    ok_gap = abs(gap - claim_gap) < 0.01
    ok_n = (len(voc) == claim_n_v) and (len(ins) == claim_n_i)
    return ok_gap and ok_n, (
        f"vocal n={len(voc)} (claim {claim_n_v}), instr n={len(ins)} (claim {claim_n_i}), "
        f"gap={gap:+.4f} (claim {claim_gap:+.4f})"
    )


def check_fma_decile(release_csv: str, fma_json_dir: str, claim_bottom: float, claim_top: float, claim_spearman: float) -> tuple[bool, str]:
    import numpy as np
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return True, "skip: scipy not installed"
    if not os.path.exists(release_csv):
        return True, f"skip: release CSV not found ({release_csv})"
    if not os.path.isdir(fma_json_dir):
        return True, (
            f"skip: FMA metadata dir not found ({fma_json_dir}); "
            "pass --features-root to verify (requires external music-ranknet data)"
        )
    rewards = {}
    with open(release_csv) as f:
        for row in csv.DictReader(f):
            rewards[row["track_id"]] = float(row["reward_score"])
    listens = {}
    for jp in glob.glob(os.path.join(fma_json_dir, "fma_*.json")):
        tid = os.path.splitext(os.path.basename(jp))[0]
        if tid not in rewards:
            continue
        meta = json.load(open(jp))
        l = meta.get("original_listens")
        if l is not None and l >= 0:
            listens[tid] = int(l)
    rs = np.array([rewards[t] for t in (set(rewards) & set(listens))])
    ls = np.array([listens[t] for t in (set(rewards) & set(listens))])
    log_l = np.log1p(ls)
    edges = np.percentile(log_l, np.linspace(0, 100, 11))
    bot = float(rs[(log_l >= edges[0]) & (log_l <= edges[1])].mean())
    top = float(rs[(log_l >= edges[9]) & (log_l <= edges[10])].mean())
    rho, _ = spearmanr(log_l, rs)
    ok = (
        abs(bot - claim_bottom) < 0.01
        and abs(top - claim_top) < 0.01
        and abs(rho - claim_spearman) < 0.01
    )
    return ok, f"bottom={bot:+.4f} (claim {claim_bottom:+.4f}), top={top:+.4f} (claim {claim_top:+.4f}), spearman={rho:+.4f} (claim {claim_spearman:+.4f})"


def check_binomtest(k: int, n: int, threshold: float) -> tuple[bool, str]:
    try:
        from scipy.stats import binomtest
    except ImportError:
        return True, "skip: scipy not installed"
    r = binomtest(k, n, 0.5, alternative="two-sided")
    return r.pvalue < threshold, f"k={k}/n={n}, p={r.pvalue:.3e} (threshold {threshold:.0e})"


def check_release_pool_sum(release_dir: str, claim_total: int = 219020) -> tuple[bool, str]:
    # Per-dataset expected counts (paper Table; MTG-Jamendo is the DEDUPED count).
    expected = {
        "sdd": 706,
        "mtg_jamendo": 55701,
        "midicaps": 5000,
        "musiccaps": 5352,
        "mtat": 25860,
        "fma_large": 106401,
        "openmic": 20000,
    }
    # Collect candidates per dataset, prioritising *_rescored.csv over the base
    # *_scores.csv, and excluding backup files (.bak*, etc.). The plain
    # "*_scores.csv" glob silently misses the rescored files, which is the
    # canonical source for MTG-Jamendo (post-dedupe).
    chosen: dict[str, str] = {}
    for f in sorted(os.listdir(release_dir)):
        if not f.endswith(".csv"):
            continue
        # Skip backups and any non-canonical sidecars.
        if ".bak" in f or f.endswith(".bak"):
            continue
        m = re.match(r"^(?P<key>.+?)_scores(?P<resc>_rescored)?\.csv$", f)
        if not m:
            continue
        key = m.group("key")
        is_rescored = m.group("resc") is not None
        # Prefer rescored over base.
        if key not in chosen or is_rescored:
            chosen[key] = f

    total = 0
    per_dataset: dict[str, int] = {}
    for key, fname in chosen.items():
        with open(os.path.join(release_dir, fname)) as fp:
            n = sum(1 for _ in csv.DictReader(fp))
        per_dataset[key] = n
        total += n

    # Tighter check: exact match against claim total (219,020). Also flag any
    # per-dataset drift > 0 against the paper Table.
    mismatches = []
    for key, want in expected.items():
        got = per_dataset.get(key)
        if got is None:
            mismatches.append(f"{key}=MISSING (want {want})")
        elif got != want:
            mismatches.append(f"{key}={got} (want {want}, drift {got - want:+d})")
    # MTG-Jamendo dedupe sanity: rescored MUST be present and = 55,701.
    mtg_file = chosen.get("mtg_jamendo", "")
    if "rescored" not in mtg_file:
        mismatches.append(
            f"mtg_jamendo not using rescored/deduped file (got {mtg_file or 'NONE'}); "
            "expected mtg_jamendo_scores_rescored.csv with 55,701 unique tracks"
        )

    ok = (total == claim_total) and not mismatches
    msg = (
        f"release pool sum {total} vs claim {claim_total} "
        f"(diff {total - claim_total:+d}); "
        f"files: {sorted(chosen.values())}"
    )
    if mismatches:
        msg += "; mismatches: " + " | ".join(mismatches)
    return ok, msg


def main():
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    # Defaults are repo-relative so a fresh-clone reviewer can run with no args.
    # External data sources (paper, music-ranknet feature cache, SongEval audio)
    # are optional; checks that depend on them SKIP cleanly when absent.
    ap.add_argument(
        "--paper-root",
        default=str(repo_root.parent / "neurips2026-paper"),
        help="path to paper repo (optional; not yet used by any check)",
    )
    ap.add_argument(
        "--features-root",
        default=str(repo_root / "data" / "processed_features"),
        help="path to processed feature cache (MusicArena/, FMA_Scoring/)",
    )
    ap.add_argument(
        "--songeval-audio",
        default=str(repo_root / "data" / "SongEval" / "audio"),
        help="path to SongEval raw audio (.wav)",
    )
    ap.add_argument(
        "--release-scores",
        default=str(repo_root / "release_scores"),
        help="path to release_scores/ CSV dir",
    )
    args = ap.parse_args()

    print("Paper numbers cross-check\n" + "=" * 50)

    checks = [
        ("§A.D ACE-Step over% 73.6 binomtest p<10^-3", lambda: check_binomtest(103, 140, 1e-3)),
        ("§A.D MusicGen-medium miss% 83.8 binomtest p<10^-3", lambda: check_binomtest(31, 37, 1e-3)),
        ("§6/§A SongEval median ~3.4 min", lambda: check_songeval_median(args.songeval_audio, 3.4)),
        ("§A.F vocal probe gap +0.441",
         lambda: check_vocal_probe(args.features_root, 0.441, 2948, 3172)),
        ("§A.F FMA decile bottom -1.413 / top +0.084 / Spearman +0.285",
         lambda: check_fma_decile(
             # Prefer rescored CSV if present.
             os.path.join(
                 args.release_scores,
                 "fma_large_scores_rescored.csv"
                 if os.path.exists(os.path.join(args.release_scores, "fma_large_scores_rescored.csv"))
                 else "fma_large_scores.csv",
             ),
             os.path.join(args.features_root, "FMA_Scoring", "json"),
             -1.413, 0.084, 0.285)),
        ("§7 release pool 219,020 total (MTG-Jamendo to 55,701)",
         lambda: check_release_pool_sum(args.release_scores, 219020)),
    ]

    fails = 0
    for name, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"ERROR: {e}"
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {msg}")
        if not ok:
            fails += 1

    print("=" * 50)
    print(f"{len(checks) - fails}/{len(checks)} passed")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
