"""Pairwise logistic training loop for TuneJury.

Loads pre-extracted CLAP + MERT features (one .pt file per pair) and
fits the 4-layer MLP head with the shared-weight pairwise logistic
loss

    L = -log sigma(s(A) - s(B))

Ties are kept in training with a soft 0.5 target. AdamW with early
stopping. A full single-GPU run converges in under 10 minutes on an
RTX A5000 (Section 3 of the paper).

Input-modality flags (`--no-clap-audio`, `--no-mert`, `--no-text`)
support the seven-variant ablation in Appendix C / Table `feature_modality`. Each
flag zeros out that block from the concatenated input; the model
`input_dim` is set dynamically from the remaining blocks.

Example
-------
$ python -m tunejury.train \\
      --features-dir data/processed_features \\
      --train-ids data/splits/train.txt \\
      --val-ids data/splits/val.txt \\
      --out checkpoints/tunejury.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import TuneJuryDataset
from .model import TuneJury


BLOCK_DIMS = {"clap_audio": 512, "mert": 1024, "text": 512}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _input_block(
    batch: dict, side: str, exclude: frozenset[str] = frozenset()
) -> torch.Tensor:
    """Concatenate the configured input blocks for clip `side` ('a' or 'b').

    Concat order matches the released eval path: [clap_audio, mert, text].
    """
    parts: list[torch.Tensor] = []
    if "clap_audio" not in exclude:
        parts.append(batch[f"clap_{side}"])
    if "mert" not in exclude:
        parts.append(batch[f"mert_{side}"])
    if "text" not in exclude:
        parts.append(batch["text"])
    return torch.cat(parts, dim=-1)


def _input_dim(exclude: frozenset[str]) -> int:
    return sum(d for k, d in BLOCK_DIMS.items() if k not in exclude)


def _run_epoch(
    model: TuneJury,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    exclude: frozenset[str],
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    n_nontie = 0
    for batch in loader:
        x_a = _input_block(batch, "a", exclude).to(device)
        x_b = _input_block(batch, "b", exclude).to(device)
        target = batch["target"].to(device).view(-1)

        with torch.set_grad_enabled(is_train):
            logits = model(x_a, x_b).view(-1)  # s(A) - s(B)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * target.numel()
        non_tie = target != 0.5
        n_nontie += int(non_tie.sum())
        if non_tie.any():
            preds = (logits[non_tie] > 0).float()
            correct += int((preds == target[non_tie]).sum())

    n = len(loader.dataset)
    acc = correct / max(n_nontie, 1)
    return total_loss / max(n, 1), acc


def train(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exclude: frozenset[str] = frozenset(
        k
        for k, flag in [
            ("clap_audio", args.no_clap_audio),
            ("mert", args.no_mert),
            ("text", args.no_text),
        ]
        if flag
    )
    input_dim = _input_dim(exclude)
    if input_dim == 0:
        raise ValueError("at least one input block must be enabled")
    print(f"[train] exclude={sorted(exclude) or '[]'} input_dim={input_dim}")

    train_ids = Path(args.train_ids).read_text().strip().splitlines()
    val_ids = Path(args.val_ids).read_text().strip().splitlines()
    mp_zero = bool(getattr(args, "musicprefs_zero_vector", False))
    if mp_zero:
        print("[train] MusicPrefs text branch zero-vectored at training (ablation).")
    train_ds = TuneJuryDataset(
        args.features_dir, train_ids, "train", musicprefs_zero_vector=mp_zero
    )
    val_ds = TuneJuryDataset(
        args.features_dir, val_ids, "val", musicprefs_zero_vector=mp_zero
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    model = TuneJury(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    patience = args.patience
    for epoch in range(args.epochs):
        train_loss, train_acc = _run_epoch(
            model, train_loader, optimizer, device, exclude
        )
        val_loss, val_acc = _run_epoch(model, val_loader, None, device, exclude)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience = args.patience
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out)
            sidecar = Path(args.out).with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {"input_dim": input_dim, "exclude": sorted(exclude)},
                    indent=2,
                )
            )
        else:
            patience -= 1
            if patience <= 0:
                print(f"[Early stop] No improvement for {args.patience} epochs.")
                break


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True)
    p.add_argument("--train-ids", required=True)
    p.add_argument("--val-ids", required=True)
    p.add_argument("--out", default="checkpoints/tunejury.pt")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-clap-audio", action="store_true")
    p.add_argument("--no-mert", action="store_true")
    p.add_argument("--no-text", action="store_true")
    p.add_argument(
        "--musicprefs-zero-vector",
        action="store_true",
        help=(
            "Zero the CLAP text embedding for MusicPrefs pairs during "
            "training (ablation; MusicPrefs annotators rated pairs "
            "without prompt access)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
