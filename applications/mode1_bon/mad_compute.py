"""Compute MAD (MAUVE Audio Divergence) for Mode 1 BoN.

For each of the four Mode 1 backbones (MusicGen-medium 1.5B AR /
MusicGen-large 3.3B AR / AudioLDM2-music 1.1B latent diffusion /
ACE-Step v1.5 Turbo Continuous 2.4B continuous-latent DiT) we compute MAD
between

    (i)  the N=1 distribution (single-seed baseline; first candidate c0
         per prompt across 100 SDD prompts), and
    (ii) the N=16 BoN-selected distribution (top-1 of c0..c15 by
         frozen TuneJury reward),

each compared against the SDD-706 reference set. MAD adds a third
distributional view (alongside FAD-CLAP and FAD-MERT) to the paper's
'distributional metrics disagree across encoders' analysis (Section 5
Mode 1 BoN; full sweep with N=32 decline in Appendix G).

MAD is computed on MERT-v1-330M audio embeddings (1024-d, mean-pooled
over time) using the `mauve-text` library's `compute_mauve(p, q)` with
k-means clustering and KL divergence (the audio-feature variant of
MAUVE introduced for music by Huang et al. 2025).

Usage
-----
$ python mad_compute.py \\
      --bon_results_root  /path/to/results \\
      --sdd706_dir        /path/to/SDD-706/audio_wav_torch \\
      --backbones         musicgen_medium musicgen_large audioldm2_music acestep

Output JSON schema (`--out_json`)
---------------------------------
Top-level: ``{backbone_name: per_backbone_dict}`` where each
``per_backbone_dict`` has:

  - ``n_prompts`` (int): number of SDD prompts retained for this backbone
    after filtering (requires c0 wav, all 16 candidate scores, top-c wav).
  - ``mad_n1`` (float): MAD between the N=1 distribution and SDD-706,
    defined as -ln(MAUVE) per Huang et al. 2025 (MusicPrefs);
    range [0, inf), lower = closer to reference.
  - ``mad_n16`` (float): MAD between the N=16 BoN-selected distribution
    and SDD-706 (same scale).
  - ``delta`` (float): ``mad_n16 - mad_n1``; negative = BoN moves the
    distribution closer to SDD-706, positive = distributional drift.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bon_results_root", type=Path, required=True,
                   help="Dir containing {backbone}_bon100 subdirs.")
    p.add_argument("--sdd706_dir", type=Path, required=True,
                   help="Dir with the 706 SDD reference wav files.")
    p.add_argument("--prompts", type=Path, required=True,
                   help="JSON list/list-of-dicts with 100 SDD prompts (matches BoN ordering).")
    p.add_argument("--backbones", nargs="+", required=True)
    p.add_argument("--out_json", type=Path, default=Path("mad_mode1_summary.json"))
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
    out = model(**inputs).last_hidden_state  # (1, T, 1024)
    return out.mean(dim=1).squeeze(0).cpu().numpy()


def load_scores(bon_dir: Path) -> dict[int, list[float]]:
    csv_p = None
    for fname in [
        "tunejury_per_candidate.csv",
        "mrn_per_candidate_proper.csv",
        "mrn_per_candidate.csv",
    ]:
        cand = bon_dir / fname
        if cand.exists():
            csv_p = cand
            break
    if csv_p is None:
        raise FileNotFoundError(
            f"No per-candidate score CSV found in {bon_dir}; expected one of "
            "tunejury_per_candidate.csv, mrn_per_candidate_proper.csv, mrn_per_candidate.csv"
        )
    out: dict[int, list[float]] = {}
    with open(csv_p) as f:
        for row in csv.DictReader(f):
            pid = int(row.get("prompt_id") or row.get("pid"))
            cand = int(row.get("candidate") or row.get("c"))
            s = row.get("tunejury_score") or row.get("mrn_score") or row.get("score") or row.get("reward")
            if s is None:
                continue
            score = float(s)
            out.setdefault(pid, [None] * 16)
            if cand < 16:
                out[pid][cand] = score
    return out


def embed_paths(model, proc, paths, tag):
    embs, t0 = [], time.time()
    for i, p in enumerate(paths):
        try:
            embs.append(mert_embed(model, proc, p))
        except Exception as e:
            print(f"  [{tag}] skip {p.name}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  [{tag}] {i+1}/{len(paths)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    return np.stack(embs).astype(np.float64)


def main():
    import mauve  # imported here to allow --help without the dependency
    args = parse_args()
    model, proc = load_mert()
    print(f"loaded MERT on {DEVICE}", flush=True)

    sdd_wavs = sorted(args.sdd706_dir.glob("*.wav"))[:706]
    sdd_features = embed_paths(model, proc, sdd_wavs, "SDD-706")
    print(f"SDD-706 features: {sdd_features.shape}", flush=True)

    summary = {}
    for name in args.backbones:
        bon_dir = args.bon_results_root / f"{name}_bon100"
        scores = load_scores(bon_dir)
        n1_paths, n16_paths = [], []
        for pid in range(100):
            n1 = bon_dir / f"{pid}_c0.wav"
            if not n1.exists() or pid not in scores:
                continue
            ss = [x for x in scores[pid] if x is not None]
            if len(ss) < 16:
                continue
            top_c = int(np.argmax(scores[pid]))
            n16 = bon_dir / f"{pid}_c{top_c}.wav"
            if not n16.exists():
                continue
            n1_paths.append(n1)
            n16_paths.append(n16)
        print(f"[{name}] N=1: {len(n1_paths)}  N=16: {len(n16_paths)}", flush=True)

        n1_embs = embed_paths(model, proc, n1_paths, f"{name}-N1")
        n16_embs = embed_paths(model, proc, n16_paths, f"{name}-N16")

        # Paper protocol: num_buckets='auto', seed=42 (matches Mode 2 + Mode 3
        # mad_compute pipeline). For n=100, 'auto' resolves to 10, so numerically
        # equivalent to explicit num_buckets=10 but uses the same canonical
        # config across all modes.
        mauve_n1 = mauve.compute_mauve(
            p_features=n1_embs, q_features=sdd_features,
            featurize_model_name=None, num_buckets='auto', seed=42, verbose=False
        ).mauve
        mauve_n16 = mauve.compute_mauve(
            p_features=n16_embs, q_features=sdd_features,
            featurize_model_name=None, num_buckets='auto', seed=42, verbose=False
        ).mauve
        # MAD = -ln(MAUVE), per Huang et al. 2025 (MusicPrefs):
        # lower MAD = closer to reference, aligning with FAD's direction.
        mad_n1 = float(-np.log(mauve_n1))
        mad_n16 = float(-np.log(mauve_n16))

        summary[name] = {
            "n_prompts": len(n1_paths),
            "mad_n1": float(mad_n1),
            "mad_n16": float(mad_n16),
            "delta": float(mad_n16 - mad_n1),
        }
        print(f"[{name}] MAD N=1: {mad_n1:.4f}  N=16: {mad_n16:.4f}  "
              f"delta: {mad_n16 - mad_n1:+.4f}", flush=True)

    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
