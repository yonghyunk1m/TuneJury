"""Per-dataset leave-out internal evaluation (Section 4.1 / Table internal_per_dataset).

For each of the four training sources X ∈ {MA, MP, AIME, SE}, compares:

* Released TuneJury (full 4-dataset mix) ``checkpoints/tunejury.pt``
* leave-X-out variant ``checkpoints/tunejury_leave_X.pt``

both evaluated on X's held-out test split. Computes:

* In-mix accuracy (released checkpoint)
* Leave-out accuracy (leave-X-out checkpoint)
* Gain = in-mix - leave-out

Expected (paper Table internal_per_dataset):

* Music Arena (n=20):  full 0.800, leave-out 0.750, gain +0.050
* MusicPrefs (n=206):  full 0.718, leave-out 0.689, gain +0.029
* AIME (n=1,560):      full 0.674, leave-out 0.625, gain +0.049
* SongEval (n=249):    full 0.908, leave-out 0.815, gain +0.093

Usage
-----
$ python -m eval.internal_per_dataset \\
      --checkpoint checkpoints/tunejury.pt \\
      --leave-out-checkpoints checkpoints/tunejury_leave_MA.pt \\
                              checkpoints/tunejury_leave_MP.pt \\
                              checkpoints/tunejury_leave_AIME.pt \\
                              checkpoints/tunejury_leave_SE.pt \\
      --features-dir data/processed_features \\
      --splits-dir   data/splits
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunejury import TuneJury, TuneJuryDataset


DATASET_TAGS = {
    "MA": "music_arena",
    "MP": "musicprefs",
    "AIME": "aime",
    "SE": "songeval",
}


def _input_block(batch: dict, side: str) -> torch.Tensor:
    return torch.cat(
        [batch[f"clap_{side}"], batch[f"mert_{side}"], batch["text"]], dim=-1
    )


def _evaluate(ckpt_path: Path, ids_file: Path, features_dir: Path) -> tuple[float, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TuneJury()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    ids = Path(ids_file).read_text().strip().splitlines()
    ds = TuneJuryDataset(data_dir=str(features_dir), split_ids=ids, mode="test")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    correct = 0
    n = 0
    for batch in loader:
        x_a = _input_block(batch, "a").to(device)
        x_b = _input_block(batch, "b").to(device)
        target = batch["target"].to(device).view(-1)
        with torch.no_grad():
            logits = model(x_a, x_b).view(-1)
        non_tie = target != 0.5
        if non_tie.any():
            preds = (logits[non_tie] > 0).float()
            correct += int((preds == target[non_tie]).sum())
            n += int(non_tie.sum())
    return correct / max(n, 1), n


def _detect_tag(ckpt_path: Path) -> str:
    name = ckpt_path.stem
    for tag in DATASET_TAGS:
        if re.search(rf"leave_?{tag}\b", name, re.IGNORECASE):
            return tag
    raise ValueError(f"cannot detect dataset tag from {ckpt_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--leave-out-checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    print(f"{'Dataset':<10} {'n':>5} {'in-mix':>8} {'leave-out':>10} {'gain':>7}")
    rows = []
    for ckpt in args.leave_out_checkpoints:
        tag = _detect_tag(ckpt)
        test_ids = args.splits_dir / f"test_{DATASET_TAGS[tag]}.txt"
        if not test_ids.exists():
            print(f"  WARNING: {test_ids} not found, skipping {tag}")
            continue
        in_mix, n = _evaluate(args.checkpoint, test_ids, args.features_dir)
        leave_out, _ = _evaluate(ckpt, test_ids, args.features_dir)
        gain = in_mix - leave_out
        print(f"{tag:<10} {n:>5} {in_mix:>8.4f} {leave_out:>10.4f} {gain:>+7.4f}")
        rows.append(
            {
                "dataset": tag,
                "n": n,
                "in_mix": in_mix,
                "leave_out": leave_out,
                "gain": gain,
            }
        )

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w") as fp:
            fp.write("dataset,n,in_mix,leave_out,gain\n")
            for r in rows:
                fp.write(f"{r['dataset']},{r['n']},{r['in_mix']:.4f},{r['leave_out']:.4f},{r['gain']:+.4f}\n")
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
