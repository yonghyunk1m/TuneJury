"""Per-variant input-ablation evaluation (Appendix C / Table feature_modality).

Reads each `checkpoints/input_ablation/A{1..7}_*.pt` together with its
sidecar `.json` (written by `tunejury.train`), rebuilds the head with
the correct `input_dim`, and evaluates pairwise accuracy on the four
held-out test splits plus their aggregate (Overall).

The seven variants and expected Overall accuracies (paper Table `feature_modality`, single-seed retrain at seed 42, n=2,035 held-out non-tie aggregate):

| ID | Features                         | Overall |
|----|----------------------------------|---------|
| A1 | CLAP audio only                  | 0.705   |
| A2 | MERT only                        | 0.695   |
| A3 | CLAP text only                   | 0.515   |
| A4 | CLAP audio + MERT                | 0.701   |
| A5 | CLAP audio + CLAP text           | 0.708   |
| A6 | MERT + CLAP text                 | 0.698   |
| A7 | CLAP audio + MERT + CLAP text    | 0.705   |

Usage
-----
$ python -m eval.input_ablation_eval \\
      --ckpt-dir   checkpoints/input_ablation/ \\
      --features-dir data/processed_features \\
      --splits-dir   data/splits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunejury import TuneJury, TuneJuryDataset
from tunejury.train import _input_block


PER_DATASET = {
    "Music Arena": "test_music_arena.txt",
    "MusicPrefs":  "test_musicprefs.txt",
    "AIME":        "test_aime.txt",
    "SongEval":    "test_songeval.txt",
}


def _accuracy(
    model: TuneJury,
    ids_file: Path,
    features_dir: Path,
    exclude: frozenset[str],
    device: str,
) -> tuple[float, int]:
    ids = ids_file.read_text().strip().splitlines()
    ds = TuneJuryDataset(data_dir=str(features_dir), split_ids=ids, mode="test")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    correct = 0
    n = 0
    for batch in loader:
        x_a = _input_block(batch, "a", exclude).to(device)
        x_b = _input_block(batch, "b", exclude).to(device)
        target = batch["target"].to(device).view(-1)
        with torch.no_grad():
            logits = model(x_a, x_b).view(-1)
        non_tie = target != 0.5
        if non_tie.any():
            preds = (logits[non_tie] > 0).float()
            correct += int((preds == target[non_tie]).sum())
            n += int(non_tie.sum())
    return correct / max(n, 1), n


def _load(ckpt_path: Path, device: str) -> tuple[TuneJury, frozenset[str]]:
    sidecar = ckpt_path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"missing sidecar JSON next to {ckpt_path.name}; "
            f"re-train via tunejury.train to emit the sidecar metadata."
        )
    meta = json.loads(sidecar.read_text())
    model = TuneJury(input_dim=meta["input_dim"])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    exclude = frozenset(meta["exclude"])
    return model, exclude


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = sorted(args.ckpt_dir.glob("A?_*.pt"))
    if not ckpts:
        print(f"no A?_*.pt checkpoints in {args.ckpt_dir}")
        return

    print(
        f"{'Variant':<28} {'n_overall':>9} {'Overall':>8} "
        + " ".join(f"{name:>12}" for name in PER_DATASET)
    )
    rows = []
    for ckpt in ckpts:
        model, exclude = _load(ckpt, device)
        overall_acc, overall_n = _accuracy(
            model, args.splits_dir / "test.txt", args.features_dir, exclude, device,
        )
        per_ds = {}
        for name, fname in PER_DATASET.items():
            acc, n = _accuracy(
                model, args.splits_dir / fname, args.features_dir, exclude, device,
            )
            per_ds[name] = (acc, n)
        print(
            f"{ckpt.stem:<28} {overall_n:>9d} {overall_acc:>8.4f} "
            + " ".join(f"{per_ds[name][0]:>12.4f}" for name in PER_DATASET)
        )
        rows.append({
            "variant": ckpt.stem,
            "overall_n": overall_n,
            "overall": overall_acc,
            **{name: per_ds[name][0] for name in PER_DATASET},
            **{f"n_{name}": per_ds[name][1] for name in PER_DATASET},
        })

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = ["variant", "overall_n", "overall"] + list(PER_DATASET) + [
            f"n_{name}" for name in PER_DATASET
        ]
        with args.out_csv.open("w") as fp:
            fp.write(",".join(cols) + "\n")
            for r in rows:
                fp.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
