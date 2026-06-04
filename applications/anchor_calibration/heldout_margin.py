"""Compute |Delta r| distribution on the released TuneJury's full
non-tie held-out test (paper Section 3 / Appendix §A.D
in-distribution baseline row).

The earlier `intrinsic_difficulty.py` script reported the bench-clean
Music Arena decisive subset (n=20 after the 1,340-uuid CMI-RewardBench
removal) as the in-distribution reference. That subset is too small to
be a robust comparison anchor versus the 598-pair Feb-Mar and 397-pair
April post-cutoff slices. This script computes the same statistics on
the full 4-dataset held-out test (n=2,035 non-tie pairs), which is the
baseline reported in Table `intrinsic_difficulty` of the paper.

Headline numbers (released checkpoint, seed 42):
    n=2,035  mean |Delta r| = 1.148  median = 0.728
    |Delta r| < 0.5 = 36.3%
    |Delta r| < 1.0 = 62.3%

Both post-cutoff slices (Feb-Mar mean 0.560 / April mean 0.455) sit
clearly below this in-distribution mean, supporting the paper's
'post-cutoff battles are intrinsically harder' claim with a
statistically robust baseline.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Defer import so this script runs from anywhere in the repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from tunejury import TuneJury, TuneJuryDataset  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(_REPO_ROOT / "checkpoints/tunejury.pt"))
    p.add_argument("--features_dir", default=str(_REPO_ROOT / "data/processed_features"))
    p.add_argument("--test_ids", default=str(_REPO_ROOT / "data/splits/test.txt"))
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def _input_block(batch, side):
    return torch.cat(
        [batch[f"clap_{side}"], batch[f"mert_{side}"], batch["text"]],
        dim=-1,
    )


@torch.no_grad()
def main():
    args = parse_args()
    test_ids = Path(args.test_ids).read_text().strip().splitlines()
    ds = TuneJuryDataset(args.features_dir, test_ids, "test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    head = TuneJury(input_dim=2048).to(DEVICE).eval()
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    head.load_state_dict(state if "state_dict" not in state else state["state_dict"])

    margins, targets = [], []
    for batch in loader:
        x_a = _input_block(batch, "a").to(DEVICE)
        x_b = _input_block(batch, "b").to(DEVICE)
        m = head(x_a, x_b).view(-1).cpu().numpy()
        margins.append(m)
        targets.append(batch["target"].view(-1).numpy())
    margins = np.concatenate(margins)
    targets = np.concatenate(targets)

    non_tie = targets != 0.5
    delta_r = np.abs(margins[non_tie])

    out = {
        "n_total_pairs": int(len(margins)),
        "n_non_tie": int(non_tie.sum()),
        "mean_abs_delta_r": float(delta_r.mean()),
        "median_abs_delta_r": float(np.median(delta_r)),
        "frac_below_0_5": float((delta_r < 0.5).mean()),
        "frac_below_1_0": float((delta_r < 1.0).mean()),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
