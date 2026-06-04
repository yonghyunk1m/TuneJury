"""Intrinsic difficulty probe (Paper Appendix §A.D, Table `intrinsic_difficulty`).

Compares |Delta_TJ| (released TuneJury absolute pairwise margin) across:
  - bench_clean test split (in-distribution)
  - Feb-Mar held-out post-cutoff slice
  - April held-out post-cutoff slice (truly future month)

A right-tail-shrinkage on the post-cutoff slices is evidence that the
two clips in each pair are closer in TuneJury's frozen feature space:
post-cutoff battles are intrinsically harder for the model, not just a
TuneJury calibration failure. Pair this with the human BOTH_BAD / TIE
fraction reported by April raw labels for a converging argument.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def margin_set(model: TuneJury, feat_dir: Path, uuids: list[str],
               require_decisive_winner: bool = False) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for u in uuids:
            pt = feat_dir / f"{u}.pt"
            if not pt.exists():
                continue
            d = torch.load(pt, map_location=DEVICE, weights_only=False)
            if require_decisive_winner:
                w = str(d.get("winner", "tie")).strip().lower()
                if w not in ("model_a", "model_b"):
                    continue
            te = d["text_emb"].to(DEVICE).float()
            f_a = torch.cat(
                [d["clap_a"].to(DEVICE).float(), d["mert_a"].to(DEVICE).float(), te]
            ).unsqueeze(0)
            f_b = torch.cat(
                [d["clap_b"].to(DEVICE).float(), d["mert_b"].to(DEVICE).float(), te]
            ).unsqueeze(0)
            out.append(abs(float(model(f_a).item()) - float(model(f_b).item())))
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tunejury.pt")
    parser.add_argument("--feat-dir-fm", required=True,
                        help="MusicArena feature cache (Feb-Mar + bench-clean)")
    parser.add_argument("--feat-dir-apr", required=True)
    parser.add_argument("--bench-test", required=True,
                        help="JSON list of bench_clean test UUIDs")
    parser.add_argument("--increment-uuids", required=True,
                        help="JSON list of Feb-Mar increment UUIDs")
    parser.add_argument("--label-csv", required=True,
                        help="ma_postcut_scored.csv; restricts the Feb-Mar "
                             "universe to the 598 anchor-calibration pairs "
                             "(incr ∩ csv-decisive), matching paper "
                             "Section A.D.")
    parser.add_argument("--raw-april-dir", default=None,
                        help="raw April battle_data dir; if given, also reports "
                             "TIE/BOTH_BAD fractions on the human side")
    args = parser.parse_args()

    model = TuneJury(input_dim=2048)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state if "state_dict" not in state else state["state_dict"])
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    bench = json.loads(Path(args.bench_test).read_text())
    incr = json.loads(Path(args.increment_uuids).read_text())
    if isinstance(incr, dict):
        incr = incr["new_uuids"]
    import csv as _csv
    csv_decisive = {
        r["battle_id"] for r in _csv.DictReader(open(args.label_csv))
        if r.get("preference") in ("A", "B")
    }
    incr = [u for u in incr if u in csv_decisive]
    apr_uuids = sorted(p.stem for p in Path(args.feat_dir_apr).glob("*.pt"))

    print("Scoring bench_clean.test (decisive subset)...")
    m_bench = margin_set(model, Path(args.feat_dir_fm), bench, require_decisive_winner=True)
    print(f"  n={len(m_bench)}")
    print("Scoring Feb-Mar universe...")
    m_fm = margin_set(model, Path(args.feat_dir_fm), incr)
    print(f"  n={len(m_fm)}")
    print("Scoring April...")
    m_apr = margin_set(model, Path(args.feat_dir_apr), apr_uuids)
    print(f"  n={len(m_apr)}")

    rows = [("bench-clean test (decisive)", m_bench),
            ("Feb-Mar universe", m_fm),
            ("April 2026", m_apr)]
    print("\n=== |Delta_TJ| distribution ===")
    print(f'{"set":30s} {"n":>5s} {"mean":>7s} {"median":>7s} {"<0.5":>6s} {"<1.0":>6s}')
    for name, m in rows:
        if m.size == 0:
            continue
        print(f'{name:30s} {len(m):>5d} '
              f' {m.mean():>6.3f}  {np.median(m):>6.3f}'
              f'  {(m<0.5).mean()*100:>5.1f}%  {(m<1.0).mean()*100:>5.1f}%')

    # April human-side TIE / BOTH_BAD
    if args.raw_april_dir:
        raw = Path(args.raw_april_dir)
        from collections import Counter
        prefs = Counter()
        for j in raw.glob("*.json"):
            d = json.loads(j.read_text())
            prefs[d.get("preference", "?")] += 1
        n = sum(prefs.values())
        print(f"\nApril human votes (raw, n={n}): {dict(prefs)}")
        non_decisive = prefs.get("BOTH_BAD", 0) + prefs.get("TIE", 0)
        print(f"  BOTH_BAD + TIE = {non_decisive} ({non_decisive/n*100:.1f}%)")


if __name__ == "__main__":
    main()
