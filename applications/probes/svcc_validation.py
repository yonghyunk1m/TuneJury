"""External vocal-technique validation on SVCC 2025 (test_gt + test).

Paper anchor: Appendix §A.F ("External singing-voice MOS validation")
+ Figure ``probes`` (middle panel references SingMOS; SVCC inline text).

Iterates over real-human ground-truth recordings (test_gt) and converted
SVC outputs (test), computes TuneJury reward per clip, reports
per-technique mean + one-way ANOVA across vocal techniques. Also
compares real (test_gt) vs SVC outputs (Welch t).

Data layout (download once)
---------------------------
Download via HuggingFace:
    huggingface-cli download lestervioleta/svcc2025 --repo-type dataset \\
        --local-dir <svcc_root>

The .tar.gz is auto-extracted on snapshot_download; resulting structure:
    <svcc_root>/
        test_gt/singerA/0000_Vibrato.wav   # 48 real recordings
        test_gt/singerB/...
        test/<system>/<file>.wav            # 64 SVC outputs

Filename convention: <utt_idx>_<technique>.wav where technique is one of
{Breathy, Falsetto, Glissando, Mixed_Voice, Pharyngeal, Vibrato}.

Usage
-----
$ python applications/probes/svcc_validation.py \\
      --checkpoint    checkpoints/tunejury.pt \\
      --svcc_root     /path/to/svcc2025 \\
      --device        cuda:0 \\
      --out_dir       applications/probes/results/svcc2025

Reproduces paper §A.F SVCC numbers: ANOVA F=3.82 p<0.01 across 6 techniques
on 48 real-human recordings; Mixed Voice highest mean reward (-0.11),
Pharyngeal lowest (-0.70); Welch t=-3.20 p<0.01 (real test_gt vs SVC test
outputs, sign indicates training-distribution alignment of SVC outputs).
"""
from __future__ import annotations

# PyTorch 2.6 default weights_only=True breaks LAION-CLAP checkpoint load.
import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

# Deterministic cuDNN for paper-matching repeatable inference (no random ops in
# Scorer.score forward pass, but cuDNN algorithm selection and fp/bf16 reduction
# order can vary across runs and GPUs without these flags).
_torch.backends.cudnn.deterministic = True
_torch.backends.cudnn.benchmark = False
_torch.manual_seed(42)

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as ss
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.score import Scorer


def _parse_filename(path: Path) -> tuple[str, int, str]:
    """test_gt/singerA/0000_Vibrato.wav -> ('A', 0, 'Vibrato')"""
    singer = path.parts[-2].replace("singer", "")
    fname = path.stem
    idx_str, _, tech = fname.partition("_")
    try:
        idx = int(idx_str)
    except ValueError:
        idx = -1
    return singer, idx, tech


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/tunejury.pt")
    ap.add_argument("--svcc_root", default=None,
                    help="SVCC 2025 dataset root (required when --reinfer)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="applications/probes/results/svcc2025")
    ap.add_argument("--reinfer", action="store_true",
                    help="Re-run audio inference (may show small variance vs committed scores due to "
                         "bf16 attention reduction in transformer encoders). Default: load committed "
                         "scores from out_dir/summary.json and recompute analysis only (paper-exact).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    committed = out_dir / "summary.json"

    if not args.reinfer and committed.exists():
        print(f"[svcc] Loading committed scores from {committed} (use --reinfer to re-run inference)", flush=True)
        prev = json.loads(committed.read_text())
        results = {"test_gt": prev.get("test_gt", []), "test": prev.get("test", [])}
        if not results["test_gt"]:
            sys.exit(f"Committed {committed} has no test_gt entries; pass --reinfer with --svcc_root")
        print(f"  test_gt: {len(results['test_gt'])} | test: {len(results['test'])}", flush=True)
    else:
        if not args.svcc_root:
            sys.exit("--svcc_root required for --reinfer (or to bootstrap a fresh summary.json)")
        files_gt = sorted(Path(args.svcc_root, "test_gt").rglob("*.wav"))
        files_svc = sorted(Path(args.svcc_root, "test").rglob("*.wav"))
        print(f"[svcc] Re-inferring. test_gt: {len(files_gt)} | test (SVC outputs): {len(files_svc)}", flush=True)
        scorer = Scorer.from_pretrained(args.checkpoint, device=args.device)
        print("scorer loaded", flush=True)
        results = {"test_gt": [], "test": []}
        t0 = time.time()
        for tag, files in [("test_gt", files_gt), ("test", files_svc)]:
            for fp in files:
                try:
                    score = scorer.score(str(fp), prompt="")
                    singer, idx, tech = _parse_filename(fp)
                    results[tag].append({
                        "path": str(fp.relative_to(args.svcc_root)),
                        "singer": singer, "utt_idx": idx, "technique": tech,
                        "tj_score": float(score),
                    })
                except Exception as e:
                    print(f"FAIL {fp}: {e}", flush=True)
            print(f"  {tag}: {len(results[tag])}/{len(files)} ({time.time()-t0:.1f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-technique on test_gt (real human)
    by_tech = defaultdict(list)
    for r in results["test_gt"]:
        by_tech[r["technique"]].append(r["tj_score"])
    tech_stats = {
        tech: {
            "n": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(min(scores)),
            "max": float(max(scores)),
        }
        for tech, scores in by_tech.items()
    }

    # One-way ANOVA across techniques (n>=2)
    groups = [s for s in by_tech.values() if len(s) >= 2]
    anova_f, anova_p = (float("nan"), float("nan"))
    if len(groups) >= 2:
        anova_f, anova_p = ss.f_oneway(*groups)

    # Real (test_gt) vs SVC (test) Welch's t
    gt_scores = [r["tj_score"] for r in results["test_gt"]]
    svc_scores = [r["tj_score"] for r in results["test"]]
    welch_t, welch_p = (float("nan"), float("nan"))
    if gt_scores and svc_scores:
        welch_t, welch_p = ss.ttest_ind(gt_scores, svc_scores, equal_var=False)

    summary = {
        "test_gt": results["test_gt"],
        "test": results["test"],
        "tech_stats": tech_stats,
        "anova": {"F": float(anova_f), "p": float(anova_p), "n_groups": len(groups)},
        "real_vs_svc": {
            "welch_t": float(welch_t), "welch_p": float(welch_p),
            "real_mean": float(np.mean(gt_scores)) if gt_scores else None,
            "svc_mean": float(np.mean(svc_scores)) if svc_scores else None,
            "n_real": len(gt_scores), "n_svc": len(svc_scores),
        },
    }
    out_path = out_dir / "summary.json"
    json.dump(summary, open(out_path, "w"))
    print(f"\nwrote {out_path}")

    print(f"\n=== Per-technique (test_gt real human) ===")
    for tech, st in sorted(tech_stats.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {tech:<15} n={st['n']:<3} mean={st['mean']:+.4f} std={st['std']:.4f}")
    print(f"\nANOVA across {len(groups)} techniques: F={anova_f:.3f}, p={anova_p:.4g}")

    print(f"\n=== Real (test_gt) vs SVC (test) ===")
    if gt_scores and svc_scores:
        print(f"  real mean: {np.mean(gt_scores):+.4f} (n={len(gt_scores)})")
        print(f"  svc  mean: {np.mean(svc_scores):+.4f} (n={len(svc_scores)})")
        print(f"  Welch t={welch_t:+.3f}, p={welch_p:.4g}")


if __name__ == "__main__":
    main()
