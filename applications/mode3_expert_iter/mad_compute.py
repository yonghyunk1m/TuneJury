"""Compute MAD (MAUVE Audio Divergence) for Mode 3 expert iteration.

Mode 3 post-trains FluxAudio-S (~120M rectified-flow DiT) with
TuneJury-driven expert iteration (paper Section 5.3). Across a
single-round LR sweep (1e-6 / 5e-6 / 1e-5), the TuneJury reward
goes up from a baseline of -0.262 by +0.166 / +0.369 / +0.416,
with MAD as the distributional side metric showing deltas
+0.293 / +0.284 / +0.669 (positive ΔMAD = drift away from SDD-706,
since lower MAD means closer). The most aggressive 1e-5 setting
yields the largest reward gain alongside the largest MAD regression,
a textbook reward-exploitation signature.

MAD provides a distributional view of how far the post-trained
checkpoint has drifted from the SDD-706 reference distribution.
We compute MAD on MERT-v1-330M embeddings (1024-d, mean-pooled
over time) using the `mauve-text` library, which matches the
audio-feature MAUVE protocol of Huang et al. 2025.

Usage
-----
$ python mad_compute.py \\
      --baseline_dir   <100 wav from pretrained FluxAudio-S> \\
      --after_ft_dir   <100 wav from expert-iter checkpoint> \\
      --sdd706_dir     <706 SDD reference wav> \\
      --out_json       mad_mode3_summary.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline_dir", type=Path, required=True)
    p.add_argument("--after_ft_dir", type=Path, required=True)
    p.add_argument("--sdd706_dir", type=Path, required=True)
    p.add_argument("--out_json", type=Path, default=Path("mad_mode3_summary.json"))
    return p.parse_args()


def load_mert():
    from transformers import AutoModel, AutoFeatureExtractor
    proc = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, proc


@torch.no_grad()
def mert_embed(model, proc, wav_path: Path) -> np.ndarray:
    wav_np, sr = sf.read(str(wav_path), always_2d=False)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    wav = torch.from_numpy(wav_np).float().unsqueeze(0)
    wav_24k = torchaudio.functional.resample(wav, sr, 24000).squeeze(0).cpu().numpy()
    inputs = proc(wav_24k, sampling_rate=24000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = model(**inputs).last_hidden_state
    return out.mean(dim=1).squeeze(0).cpu().numpy()


def embed_dir(model, proc, dir_path: Path, tag: str, max_n=None):
    wavs = sorted(dir_path.glob("*.wav"))
    if max_n:
        wavs = wavs[:max_n]
    embs, t0 = [], time.time()
    for i, w in enumerate(wavs):
        try:
            embs.append(mert_embed(model, proc, w))
        except Exception as e:
            print(f"  [{tag}] skip {w.name}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] {i+1}/{len(wavs)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    return np.stack(embs).astype(np.float64)


def main():
    import mauve
    args = parse_args()
    model, proc = load_mert()
    print(f"loaded MERT on {DEVICE}", flush=True)

    b_emb = embed_dir(model, proc, args.baseline_dir, "baseline")
    a_emb = embed_dir(model, proc, args.after_ft_dir, "after-FT")
    s_emb = embed_dir(model, proc, args.sdd706_dir, "SDD-706", max_n=706)
    print(f"shapes: baseline {b_emb.shape}, after-FT {a_emb.shape}, SDD-706 {s_emb.shape}",
          flush=True)

    # Paper protocol: mauve default num_buckets='auto' (resolves to max(2, n//10) =
    # 10 for n=100, ~73 for n=30) with seed=42 for cross-cell reproducibility. The
    # equivalent num_buckets=10 only matches 'auto' for the n=100 Mode 3 / TangoFlux
    # cells; SAO (n=30) requires 'auto' to reproduce the paper +0.500.
    mauve_base = mauve.compute_mauve(
        p_features=b_emb, q_features=s_emb,
        featurize_model_name=None, num_buckets='auto', seed=42, verbose=False
    ).mauve
    mauve_after = mauve.compute_mauve(
        p_features=a_emb, q_features=s_emb,
        featurize_model_name=None, num_buckets='auto', seed=42, verbose=False
    ).mauve
    # MAD = -ln(MAUVE) per Huang et al. 2025: lower = closer to reference.
    mad_base = float(-np.log(mauve_base))
    mad_after = float(-np.log(mauve_after))

    out = {
        "mode3_baseline_to_sdd706": float(mad_base),
        "mode3_after_ft_to_sdd706": float(mad_after),
        "delta_after_minus_base":   float(mad_after - mad_base),
        "n_baseline": int(b_emb.shape[0]),
        "n_after_ft": int(a_emb.shape[0]),
        "n_sdd706":   int(s_emb.shape[0]),
    }
    print(json.dumps(out, indent=2))
    args.out_json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
