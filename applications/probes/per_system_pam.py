"""Per-system PAM musicality MOS reproducer for the §4.2 per-system PAM probe.

For each of the 5 PAM systems (MusicGen-large, MusicGen-melody,
AudioLDM2-music, MusicLDM, real-music), compute mean TuneJury reward
under both protocols (text+audio and audio-only / empty prompt), then
Spearman rank correlation against per-system mean PAM musicality MOS.

Paper claim (§4.2): SRCC = 0.90 on n=5 systems, with the four-TTM
ordering MusicGen-large ≻ MusicGen-melody ≻ AudioLDM2-music ≻ MusicLDM
exact, and real-music ranked 2nd in TuneJury (vs 1st in PAM MOS).

Data source: CMI-RewardBench's all_test.jsonl
(filter `source` containing 'PAM'). Audio files under
human_eval/music/<system>/<basename>.wav.

Determinism: manual_seed=42, cudnn.deterministic=True,
benchmark=False, TF32 off, CUBLAS_WORKSPACE_CONFIG=:4096:8. Same
settings as the Demucs probe.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# PyTorch 2.6 default weights_only=True breaks LAION-CLAP load.
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury import Scorer


def _system_from_path(audio_rel: str) -> str | None:
    """Parse system label from CMI-RewardBench PAM audio path.

    PAM paths look like 'human_eval/music/<system>/<basename>.wav'."""
    parts = Path(audio_rel).parts
    if len(parts) >= 3 and parts[0] == "human_eval" and parts[1] == "music":
        return parts[2]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/tunejury.pt")
    ap.add_argument("--bench_root", required=True,
                    help="CMI-RewardBench data root (contains all_test.jsonl + human_eval/).")
    ap.add_argument("--out_dir", default="applications/probes/results/per_system_pam")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_root = Path(args.bench_root)
    test_jsonl = bench_root / "all_test.jsonl"
    if not test_jsonl.exists():
        sys.exit(f"all_test.jsonl not found at {test_jsonl}")

    pam_rows = []
    for line in test_jsonl.open():
        d = json.loads(line)
        if "PAM" in str(d.get("source", "")):
            pam_rows.append(d)
    print(f"[per_system_pam] Loaded {len(pam_rows)} PAM clips from {test_jsonl}", flush=True)

    scorer = Scorer.from_pretrained(args.checkpoint, device=args.device)
    print(f"[per_system_pam] Scorer loaded on {args.device}", flush=True)

    per_system_tj_ta: dict[str, list[float]] = defaultdict(list)
    per_system_tj_a: dict[str, list[float]] = defaultdict(list)
    per_system_pam: dict[str, list[float]] = defaultdict(list)
    per_clip = []
    n_missing_audio = 0
    n_missing_mos = 0
    n_missing_system = 0

    for i, row in enumerate(pam_rows):
        audio_rel = row.get("audio-path", "")
        system = _system_from_path(audio_rel)
        if system is None:
            n_missing_system += 1
            continue
        audio_path = bench_root / audio_rel
        if not audio_path.exists():
            n_missing_audio += 1
            continue
        try:
            mos = float(row.get("musicality"))
        except (TypeError, ValueError):
            n_missing_mos += 1
            continue
        prompt = (row.get("prompt") or "").strip()

        # T+A protocol: prompt fed to text branch (matches paper §4.2)
        s_ta = float(scorer.score(str(audio_path), prompt=prompt))
        # A-only protocol: empty prompt → zero text vector (§3 convention)
        s_a = float(scorer.score(str(audio_path), prompt=""))

        per_system_tj_ta[system].append(s_ta)
        per_system_tj_a[system].append(s_a)
        per_system_pam[system].append(mos)
        per_clip.append({
            "audio_path": audio_rel,
            "system": system,
            "prompt": prompt,
            "musicality_mos": mos,
            "tj_ta": s_ta,
            "tj_a": s_a,
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(pam_rows)}] scored {system}", flush=True)

    print(f"\n[per_system_pam] N scored: {len(per_clip)} "
          f"(missing audio: {n_missing_audio}, missing MOS: {n_missing_mos}, "
          f"missing system: {n_missing_system})", flush=True)

    systems = sorted(per_system_pam.keys())
    print(f"[per_system_pam] Systems: {systems}", flush=True)

    rows = []
    for s in systems:
        rows.append({
            "system": s,
            "n_clips": len(per_system_pam[s]),
            "pam_mean": float(np.mean(per_system_pam[s])),
            "pam_std": float(np.std(per_system_pam[s])),
            "tj_ta_mean": float(np.mean(per_system_tj_ta[s])),
            "tj_ta_std": float(np.std(per_system_tj_ta[s])),
            "tj_a_mean": float(np.mean(per_system_tj_a[s])),
            "tj_a_std": float(np.std(per_system_tj_a[s])),
        })

    from scipy.stats import spearmanr
    pam_means = [r["pam_mean"] for r in rows]
    tj_ta_means = [r["tj_ta_mean"] for r in rows]
    tj_a_means = [r["tj_a_mean"] for r in rows]
    ai_mask = [r["system"] != "real" for r in rows]
    pam_means_ai = [m for m, k in zip(pam_means, ai_mask) if k]
    tj_ta_means_ai = [m for m, k in zip(tj_ta_means, ai_mask) if k]
    tj_a_means_ai = [m for m, k in zip(tj_a_means, ai_mask) if k]

    res = spearmanr(tj_ta_means, pam_means)
    srcc_ta_all = float(res.statistic if hasattr(res, "statistic") else res[0])
    res = spearmanr(tj_a_means, pam_means)
    srcc_a_all = float(res.statistic if hasattr(res, "statistic") else res[0])
    res = spearmanr(tj_ta_means_ai, pam_means_ai)
    srcc_ta_ai = float(res.statistic if hasattr(res, "statistic") else res[0])
    res = spearmanr(tj_a_means_ai, pam_means_ai)
    srcc_a_ai = float(res.statistic if hasattr(res, "statistic") else res[0])

    rank_by = lambda values, names: [n for _, n in sorted(zip(values, names), reverse=True)]
    pam_rank = rank_by(pam_means, [r["system"] for r in rows])
    tj_ta_rank = rank_by(tj_ta_means, [r["system"] for r in rows])
    tj_a_rank = rank_by(tj_a_means, [r["system"] for r in rows])
    pam_rank_ai = rank_by(pam_means_ai, [r["system"] for r, k in zip(rows, ai_mask) if k])
    tj_ta_rank_ai = rank_by(tj_ta_means_ai, [r["system"] for r, k in zip(rows, ai_mask) if k])
    tj_a_rank_ai = rank_by(tj_a_means_ai, [r["system"] for r, k in zip(rows, ai_mask) if k])

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "n_systems": len(systems),
        "n_clips_total": len(per_clip),
        "per_system": rows,
        "srcc_all_n5": {
            "tj_ta_vs_pam": srcc_ta_all,
            "tj_a_vs_pam": srcc_a_all,
        },
        "srcc_ai_only_n4": {
            "tj_ta_vs_pam": srcc_ta_ai,
            "tj_a_vs_pam": srcc_a_ai,
        },
        "rankings": {
            "pam_n5": pam_rank,
            "tj_ta_n5": tj_ta_rank,
            "tj_a_n5": tj_a_rank,
            "pam_ai_n4": pam_rank_ai,
            "tj_ta_ai_n4": tj_ta_rank_ai,
            "tj_a_ai_n4": tj_a_rank_ai,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_clip.json").write_text(json.dumps(per_clip, indent=2))

    print(f"\n=== Per-system table ===")
    print(f"{'system':<20} {'n':>4} {'PAM MOS':>10} {'TJ T+A':>10} {'TJ A-only':>12}")
    for r in rows:
        print(f"  {r['system']:<20} {r['n_clips']:>4} "
              f"{r['pam_mean']:>+10.4f} {r['tj_ta_mean']:>+10.4f} {r['tj_a_mean']:>+12.4f}")
    print(f"\n=== SRCC ===")
    print(f"  T+A protocol, n=5 (all):  {srcc_ta_all:+.4f}")
    print(f"  T+A protocol, n=4 (AI):   {srcc_ta_ai:+.4f}")
    print(f"  A-only protocol, n=5:     {srcc_a_all:+.4f}")
    print(f"  A-only protocol, n=4:     {srcc_a_ai:+.4f}")
    print(f"\n=== Rankings (descending mean) ===")
    print(f"  PAM (n=5):       {' ≻ '.join(pam_rank)}")
    print(f"  TJ T+A (n=5):    {' ≻ '.join(tj_ta_rank)}")
    print(f"  TJ A-only (n=5): {' ≻ '.join(tj_a_rank)}")
    print(f"\nWrote {out_dir/'summary.json'} ({len(per_clip)} clips)", flush=True)


if __name__ == "__main__":
    main()
