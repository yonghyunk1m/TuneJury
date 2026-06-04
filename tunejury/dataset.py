"""Pairwise preference dataset loader for TuneJury training.

Each .pt feature file holds a single preference pair (A, B) with the
following keys:

  text_emb : 512-d LAION-CLAP text embedding of the prompt
  clap_a   : 512-d LAION-CLAP audio embedding of clip A
  clap_b   : 512-d LAION-CLAP audio embedding of clip B
  mert_a   : 1024-d MERT-v1-330M audio embedding of clip A
  mert_b   : 1024-d MERT-v1-330M audio embedding of clip B
  flag     : 1-d source flag (one of {Music Arena, MusicPrefs, AIME,
             SongEval})
  winner   : "model_a" | "model_b" | "tie"

Ties are kept in training with soft label 0.5 and excluded from
binary accuracy reporting (Section 3 of the paper).
"""

from __future__ import annotations

import os
from typing import Sequence

import torch
from torch.utils.data import Dataset


class TuneJuryDataset(Dataset):
    """Loads pre-extracted (CLAP audio, CLAP text, MERT) features."""

    WINNER_TO_TARGET = {"model_a": 1.0, "model_b": 0.0, "tie": 0.5}

    def __init__(
        self,
        data_dir: str,
        split_ids: Sequence[str],
        mode: str = "train",
        musicprefs_zero_vector: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.mode = mode
        self.musicprefs_zero_vector = musicprefs_zero_vector
        split_ids = list(split_ids)
        self.valid_uuids = [
            u for u in split_ids
            if os.path.exists(os.path.join(data_dir, f"{u}.pt"))
        ]
        if len(split_ids) > 0 and len(self.valid_uuids) == 0:
            # Fresh-clone safety: if the split file references pairs but
            # zero of them resolve to a real .pt under data_dir, training
            # and evaluation would silently produce empty batches (and,
            # downstream, accuracy 0 / NaN ECE). Fail loudly with a
            # pointer to the reproducing doc and the in-tree README.
            sample = [str(u) for u in split_ids[:3]]
            raise FileNotFoundError(
                f"[{mode.upper()}] 0 of {len(split_ids)} split ids resolved to "
                f"a .pt under '{data_dir}'. Example missing files: "
                + ", ".join(
                    os.path.join(data_dir, f"{u}.pt") for u in sample
                )
                + ". On a fresh clone 'data/processed_features/' is "
                "intentionally empty; populate it via "
                "'data/extract_features.py' (see docs/reproducing.md §2 "
                "and data/processed_features/README.md)."
            )
        print(f"[{mode.upper()}] Loaded {len(self.valid_uuids)} pairs.")

    def __len__(self) -> int:
        return len(self.valid_uuids)

    def __getitem__(self, idx: int) -> dict:
        uuid = self.valid_uuids[idx]
        data = torch.load(
            os.path.join(self.data_dir, f"{uuid}.pt"),
            map_location="cpu",
            weights_only=False,
        )

        def to_1d(value):
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            return value.float().reshape(-1)

        flag = to_1d(data["flag"])
        if flag.numel() == 0:
            flag = torch.zeros(1, dtype=torch.float32)
        elif flag.numel() > 1:
            flag = flag[:1]

        winner = str(data.get("winner", "tie")).strip().lower()
        target = torch.tensor(
            [self.WINNER_TO_TARGET.get(winner, 0.5)],
            dtype=torch.float32,
        )

        text_emb = to_1d(data["text_emb"])  # (512,)
        if self.musicprefs_zero_vector and str(data.get("source", "")) == "MusicPrefs":
            text_emb = torch.zeros(512, dtype=torch.float32)

        return {
            "text": text_emb,
            "clap_a": to_1d(data["clap_a"]),  # (512,)
            "mert_a": to_1d(data["mert_a"]),  # (1024,)
            "clap_b": to_1d(data["clap_b"]),  # (512,)
            "mert_b": to_1d(data["mert_b"]),  # (1024,)
            "flag": flag,
            "target": target,
            "uuid": uuid,
        }
