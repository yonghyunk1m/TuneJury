"""Full-test gain'_d (Section 4.1, Table 3 second gain column).

For each training source d in {MA, MP, AIME, SE}, reports

    gain'_d = model_all(all_test) - model_{all - d}(all_test)

i.e. the gain on the FULL 2,035-pair held-out test set from including
dataset d during training (vs the per-dataset test in Table 3's first
gain column).

Expected (paper Table 3, second gain column):

    AIME (n_test=1,560 of 2,035, 77%):  gain'_d  +0.0408
    MusicArena (n_test=20, 1%):          gain'_d  +0.0029
    MusicPrefs (n_test=206, 10%):        gain'_d  +0.0054
    SongEval   (n_test=249, 12%):        gain'_d  +0.0029

Reads the same processed .pt features and leave-out checkpoints used by
``eval/internal_per_dataset.py``.

Usage
-----
$ python -m eval.full_test_gain \\
      --checkpoint     checkpoints/tunejury.pt \\
      --leave-out-ckpts checkpoints/tunejury_leave_MA.pt \\
                        checkpoints/tunejury_leave_MP.pt \\
                        checkpoints/tunejury_leave_AIME.pt \\
                        checkpoints/tunejury_leave_SE.pt \\
      --features-dir   data/processed_features \\
      --test-ids       data/splits/test.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunejury.model import TuneJury  # noqa: E402

_TORCH_LOAD_ORIG = torch.load
torch.load = lambda *a, **k: _TORCH_LOAD_ORIG(*a, **{**k, "weights_only": False})


def _winner_to_label(w: str) -> int:
    if w == "model_a":
        return 1
    if w == "model_b":
        return 0
    return -1


def _load_features(test_ids_file: Path, features_dir: Path) -> List[Tuple[torch.Tensor, torch.Tensor, int, str]]:
    with open(test_ids_file) as f:
        uuids = [l.strip() for l in f if l.strip()]
    zero_te = torch.zeros(512)
    pairs = []
    for uuid in uuids:
        pt = features_dir / f"{uuid}.pt"
        if not pt.exists():
            continue
        d = torch.load(pt, map_location="cpu")
        label = _winner_to_label(d.get("winner"))
        if label < 0:  # exclude ties
            continue
        te = d.get("text_emb", zero_te)
        if te is None or te.numel() == 0:
            te = zero_te
        x_a = torch.cat([d["clap_a"], d["mert_a"], te])
        x_b = torch.cat([d["clap_b"], d["mert_b"], te])
        pairs.append((x_a, x_b, label, str(d.get("source", ""))))
    return pairs


def _score_all(head: TuneJury, pairs, device: str) -> Tuple[float, Dict[str, Tuple[int, int]]]:
    head.eval()
    correct, total = 0, 0
    per_source: Dict[str, List[int]] = {}
    with torch.no_grad():
        for x_a, x_b, label, src in pairs:
            sa = head(x_a.unsqueeze(0).to(device)).item()
            sb = head(x_b.unsqueeze(0).to(device)).item()
            pred = 1 if sa > sb else 0
            ok = int(pred == label)
            correct += ok; total += 1
            per_source.setdefault(src, [0, 0])
            per_source[src][0] += ok; per_source[src][1] += 1
    return correct / max(total, 1), {s: tuple(v) for s, v in per_source.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--leave-out-ckpts", nargs="+", required=True, type=Path,
                        help="Leave-out checkpoints (one per dataset).")
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--test-ids", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", default="full_test_gain.json")
    args = parser.parse_args()

    print(f"Loading features from {args.features_dir}", flush=True)
    pairs = _load_features(args.test_ids, args.features_dir)
    print(f"  non-tie pairs: {len(pairs)}", flush=True)

    def run(ckpt_path: Path) -> Tuple[float, Dict[str, Tuple[int, int]]]:
        head = TuneJury(input_dim=2048)
        state = torch.load(ckpt_path, map_location="cpu")
        head.load_state_dict(state if "state_dict" not in state else state["state_dict"])
        head.to(args.device)
        return _score_all(head, pairs, args.device)

    print(f"\nEvaluating released full mix: {args.checkpoint}", flush=True)
    all_acc, all_per = run(args.checkpoint)
    print(f"  all (released) full-test acc: {all_acc:.4f}", flush=True)

    leave_results = {}
    for ckpt in args.leave_out_ckpts:
        print(f"\nEvaluating leave-out: {ckpt}", flush=True)
        acc, per_src = run(ckpt)
        gain_prime = all_acc - acc
        leave_results[ckpt.name] = {
            "acc": acc,
            "gain_prime": gain_prime,
            "per_source": {s: {"acc": c / n, "n": n} for s, (c, n) in per_src.items()},
        }
        print(f"  {ckpt.name}: full-test acc {acc:.4f}, gain'_d {gain_prime:+.4f}", flush=True)

    out = {
        "n_nontie": len(pairs),
        "all_full_test_acc": all_acc,
        "per_source_all": {s: {"acc": c / n, "n": n} for s, (c, n) in all_per.items()},
        "leave_out": leave_results,
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
