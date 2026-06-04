"""Mode 1 mode-collapse diagnostic: mean pairwise cosine distance (MPCD).

Paper anchor: Appendix G ("Mode-collapse diagnostic for the N=32 MAD
decline"), Table mode1_diversity.

For each backbone and each N ∈ {1, 8, 16, 32}, extract MERT-v1-330M
embeddings of the 100 top-1 picks and compute mean pairwise cosine
distance (higher = more spread). Rules out mode collapse as the mechanism
behind the N=32 MAD decline on AudioLDM2-music and ACE-Step Turbo Continuous.

Expected (paper §A.G, N=1 and N=32 endpoints):

| Backbone                   | N=1   | N=32  | Δ(N=32 − N=1) |
|----------------------------|-------|-------|--------------:|
| MusicGen-medium            | 0.103 | 0.108 | within ±10%   |
| MusicGen-large             | 0.099 | 0.107 | within ±10%   |
| AudioLDM2-music            | 0.151 | 0.209 | **+39%**      |
| ACE-Step Turbo Continuous  | 0.092 | 0.153 | **+66%**      |

AudioLDM2 / ACE-Step Turbo Continuous *increase* MPCD with N, ruling out
mode collapse and isolating the MAD decline as distributional drift away
from SDD-706 rather than narrowing diversity.

Usage
-----
$ python applications/mode1_bon/mpcd_diagnostic.py \\
      --backbones musicgen_medium musicgen_large audioldm2_music acestep \\
      --results-root results \\
      --N 1 8 16 32 \\
      --out-csv results/mode1_diversity.csv
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF


def _load_mert(device: str):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    name = "m-a-p/MERT-v1-330M"
    feat = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
    return feat, model


def _embed_wav(feat, mert, path: Path, device: str) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = AF.resample(wav, sr, 24000).squeeze(0).numpy()
    inputs = feat(wav, sampling_rate=24000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mert(**inputs, output_hidden_states=False)
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


def _mpcd(embeds: np.ndarray) -> float:
    """Mean pairwise cosine distance across all C(N, 2) pairs in ``embeds``."""
    embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-10)
    n = embeds.shape[0]
    if n < 2:
        return 0.0
    sims = []
    for i, j in combinations(range(n), 2):
        sims.append(float(np.dot(embeds[i], embeds[j])))
    return 1.0 - float(np.mean(sims))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--N", nargs="+", type=int, default=[1, 8, 16, 32])
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, mert = _load_mert(device)

    rows = []
    print(f"{'Backbone':<28} " + "  ".join(f"N={n:>2}" for n in args.N))
    for bb in args.backbones:
        mpcds = []
        for n in args.N:
            top_dir = args.results_root / f"{bb}_bon100" / f"topN_{n}"
            if not top_dir.exists():
                print(f"  WARNING: {top_dir} missing.")
                mpcds.append(float("nan"))
                continue
            wavs = sorted(top_dir.glob("*.wav"))[:100]
            embeds = np.stack([_embed_wav(feat, mert, p, device) for p in wavs])
            mpcds.append(_mpcd(embeds))
        print(f"{bb:<28} " + "  ".join(f"{m:.4f}" for m in mpcds))
        rows.append({"backbone": bb, **{f"N{n}": v for n, v in zip(args.N, mpcds)}})

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w") as fp:
            header = ["backbone"] + [f"N{n}" for n in args.N]
            fp.write(",".join(header) + "\n")
            for r in rows:
                fp.write(",".join(str(r[c]) for c in header) + "\n")
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
