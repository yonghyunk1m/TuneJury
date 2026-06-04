"""FAD-CLAP and FAD-MERT against SDD-706 (Section 5 / Appendix G).

FAD-X is the Frechet distance between two embedding distributions
under a chosen audio encoder X. We follow the gui2024fad convention:

    - FAD-CLAP: LAION-CLAP-Music embeddings
    - FAD-MERT: MERT-v1-330M embeddings (music-only complement)

SDD-706 is the standard 706-track FAD anchor in text-to-music
evaluation: captions from SDD, audio from MTG-Jamendo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _flatten_embeddings(arr: np.ndarray) -> np.ndarray:
    """Pool a (T, D) embedding array down to (D,) via mean over time."""
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=tuple(range(arr.ndim - 1)))


def _frechet_distance(mu1, cov1, mu2, cov2) -> float:
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    cov_sqrt, _ = sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2 * cov_sqrt))


def _moments(embeddings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([_flatten_embeddings(e) for e in embeddings])
    return X.mean(axis=0), np.cov(X, rowvar=False)


def fad(reference_embeddings, candidate_embeddings) -> float:
    mu_r, cov_r = _moments(reference_embeddings)
    mu_c, cov_c = _moments(candidate_embeddings)
    return _frechet_distance(mu_r, cov_r, mu_c, cov_c)


def _load_clap_audio_embeddings(audio_paths, device):
    import laion_clap
    import torch
    import torchaudio
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    clap.load_ckpt("music_audioset_epoch_15_esc_90.14.pt")  # LAION-CLAP-Music checkpoint
    clap = clap.to(device)
    embeddings = []
    for p in audio_paths:
        wav, sr = torchaudio.load(p)
        if sr != 48_000:
            wav = torchaudio.functional.resample(wav, sr, 48_000)
        wav = wav.mean(dim=0, keepdim=True).cpu().numpy()
        emb = clap.get_audio_embedding_from_data(x=wav, use_tensor=False)
        embeddings.append(emb[0])
    return embeddings


def _load_mert_audio_embeddings(audio_paths, device):
    import torch
    import torchaudio
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device).eval()
    embeddings = []
    with torch.no_grad():
        for p in audio_paths:
            wav, sr = torchaudio.load(p)
            if sr != 24_000:
                wav = torchaudio.functional.resample(wav, sr, 24_000)
            wav = wav.mean(dim=0).cpu().numpy()
            inputs = proc(wav, sampling_rate=24_000, return_tensors="pt").to(device)
            out = model(**inputs, output_hidden_states=True)
            stacked = torch.stack(out.hidden_states).squeeze(1).mean(dim=0)
            embeddings.append(stacked.mean(dim=0).cpu().numpy())
    return embeddings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-dir", required=True, help="SDD-706 audio directory.")
    p.add_argument("--candidate-dir", required=True, help="Generated audio directory.")
    p.add_argument("--encoder", choices=("clap", "mert"), default="clap")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ref_paths = sorted(Path(args.reference_dir).glob("*.wav"))
    cand_paths = sorted(Path(args.candidate_dir).glob("*.wav"))
    load_fn = (
        _load_clap_audio_embeddings if args.encoder == "clap"
        else _load_mert_audio_embeddings
    )
    ref_emb = load_fn(ref_paths, args.device)
    cand_emb = load_fn(cand_paths, args.device)
    score = fad(ref_emb, cand_emb)
    name = "FAD-CLAP" if args.encoder == "clap" else "FAD-MERT"
    print(json.dumps({name: score}, indent=2))


if __name__ == "__main__":
    main()
