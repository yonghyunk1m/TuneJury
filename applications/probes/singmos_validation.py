"""External singing-voice MOS validation on SingMOS-Pro.

Paper anchor: Appendix §A.F ("External singing-voice MOS validation")
+ Figure ``probes`` middle panel.

Computes TuneJury reward on every SingMOS-Pro utterance with the empty-prompt
protocol of Section 3, then reports per-utterance and per-system Spearman
rank correlation against human MOS.

Data layout (download once)
---------------------------
git clone https://huggingface.co/datasets/TangRain/SingMOS-Pro <singmos_root>

Expected structure:
    <singmos_root>/
        metadata.json
        wav/sys####-utt####.wav
        info/score.json
        info/sys_info.json

Usage
-----
$ python applications/probes/singmos_validation.py \\
      --checkpoint    checkpoints/tunejury.pt \\
      --singmos_root  /path/to/SingMOS-Pro \\
      --device        cuda:0 \\
      --shard         0 --n_shards 4 \\
      --out_dir       applications/probes/results/singmos_pro

After all 4 shards complete, aggregate with --aggregate:

$ python applications/probes/singmos_validation.py \\
      --aggregate --out_dir applications/probes/results/singmos_pro

Reproduces paper §A.F: per-utterance Spearman +0.19 and per-system
Spearman +0.44 (n=141 singing-voice synthesis systems, Chinese+Japanese).
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


def _infer(args: argparse.Namespace) -> None:
    metadata = json.load(open(Path(args.singmos_root) / "metadata.json"))
    shard = [m for i, m in enumerate(metadata) if i % args.n_shards == args.shard]
    print(f"[shard {args.shard}/{args.n_shards}] {len(shard)} items, device={args.device}", flush=True)

    scorer = Scorer.from_pretrained(args.checkpoint, device=args.device)
    print(f"[shard {args.shard}] scorer loaded", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"singmos_shard_{args.shard}.json"

    results = []
    skipped = 0
    t0 = time.time()
    for i, item in enumerate(shard):
        audio = Path(args.singmos_root) / "wav" / f"{item['id']}.wav"
        if not audio.exists():
            skipped += 1
            continue
        try:
            score = scorer.score(str(audio), prompt="")
            js = item.get("judge_score", [])
            mos = sum(js) / len(js) if js else None
            results.append({
                "id": item["id"],
                "tj_score": float(score),
                "mos": float(mos) if mos is not None else None,
                "n_judges": len(js),
                "dataset": item.get("dataset"),
                "language": item.get("language"),
            })
            if (i + 1) % 100 == 0:
                rate = (i + 1) / max(time.time() - t0, 0.01)
                eta = (len(shard) - i - 1) / max(rate, 0.01)
                print(f"[shard {args.shard}] {i+1}/{len(shard)} ({rate:.1f}/s, ETA {eta/60:.1f}min)", flush=True)
        except Exception as e:
            print(f"[shard {args.shard}] FAIL {item['id']}: {e}", flush=True)

    json.dump(results, open(out_path, "w"))
    print(f"[shard {args.shard}] wrote {out_path} ({len(results)} items, skipped {skipped}, {time.time()-t0:.1f}s)", flush=True)


def _aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    shard_files = sorted(out_dir.glob("singmos_shard_*.json"))
    if not shard_files:
        sys.exit(f"No singmos_shard_*.json in {out_dir}")

    rows = []
    for f in shard_files:
        rows.extend(json.load(open(f)))
    print(f"Total rows: {len(rows)}")

    tj = np.array([r["tj_score"] for r in rows if r.get("mos") is not None])
    mos = np.array([r["mos"] for r in rows if r.get("mos") is not None])
    rho, p = ss.spearmanr(tj, mos)
    r_p, _ = ss.pearsonr(tj, mos)
    print(f"\n=== OVERALL ===")
    print(f"  Spearman rho: {rho:+.4f} (p={p:.3g})")
    print(f"  Pearson r   : {r_p:+.4f}")
    print(f"  n           : {len(tj)}")

    sys_groups = defaultdict(list)
    for r in rows:
        if r.get("mos") is None: continue
        sys_id = r["id"].split("-")[0]
        sys_groups[sys_id].append((r["tj_score"], r["mos"]))

    sys_means_tj, sys_means_mos, sys_names = [], [], []
    for s, vals in sorted(sys_groups.items()):
        sys_means_tj.append(float(np.mean([v[0] for v in vals])))
        sys_means_mos.append(float(np.mean([v[1] for v in vals])))
        sys_names.append(s)
    sys_rho, sys_p = ss.spearmanr(sys_means_tj, sys_means_mos)
    print(f"\n=== PER-SYSTEM ({len(sys_groups)} systems) ===")
    print(f"  Spearman rho: {sys_rho:+.4f} (p={sys_p:.3g})")

    out = {
        "overall": {"spearman": float(rho), "pearson": float(r_p), "n": int(len(tj))},
        "per_system": {
            "spearman": float(sys_rho), "n_systems": len(sys_groups),
            "systems": sys_names, "tj_means": sys_means_tj, "mos_means": sys_means_mos,
        },
        "rows": rows,
    }
    out_path = out_dir / "summary.json"
    json.dump(out, open(out_path, "w"))
    print(f"\nwrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/tunejury.pt")
    ap.add_argument("--singmos_root", help="SingMOS-Pro repo root (from git clone)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--out_dir", default="applications/probes/results/singmos_pro")
    ap.add_argument("--aggregate", action="store_true", help="Aggregate existing shards into summary.json")
    ap.add_argument("--report", action="store_true",
                    help="Report stats from committed summary.json (no inference, paper-exact)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    committed = out_dir / "summary.json"
    # Default: if committed summary.json exists and neither --aggregate nor --reinfer-like
    # flag is set, just report the canonical numbers (paper-exact, no inference variance).
    if not args.aggregate and not args.singmos_root and committed.exists():
        args.report = True

    if args.report:
        if not committed.exists():
            sys.exit(f"{committed} not found; run with --aggregate (after shards) or pass --singmos_root to bootstrap")
        d = json.loads(committed.read_text())
        print(f"[singmos] Loading committed summary from {committed} (paper-exact reproduction)")
        ov = d.get("overall", {})
        ps = d.get("per_system", {})
        print(f"\n=== OVERALL ===")
        print(f"  Spearman rho: {ov.get('spearman', 0):+.4f}")
        print(f"  Pearson r   : {ov.get('pearson', 0):+.4f}")
        print(f"  n           : {ov.get('n', 0)}")
        print(f"\n=== PER-SYSTEM ({ps.get('n_systems', 0)} systems) ===")
        print(f"  Spearman rho: {ps.get('spearman', 0):+.4f}")
        return
    if args.aggregate:
        _aggregate(args)
    else:
        if not args.singmos_root:
            sys.exit("--singmos_root required for inference")
        _infer(args)


if __name__ == "__main__":
    main()
