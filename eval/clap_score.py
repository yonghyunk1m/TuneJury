"""Per-clip text-audio CLAP cosine similarity.

The "CLAP score" used throughout Section 5 and Appendix G. It
is the per-clip cosine similarity between the LAION-CLAP-Music audio
embedding of a generated clip and the LAION-CLAP-Music text
embedding of its prompt. No reference set required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio


CLAP_CKPT_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    "music_audioset_epoch_15_esc_90.14.pt"
)
CLAP_CKPT_NAME = "music_audioset_epoch_15_esc_90.14.pt"


def _resolve_clap_ckpt(ckpt_dir: str | None = None) -> str:
    """Locate the LAION-CLAP-Music checkpoint, downloading if missing.

    Mirrors ``tunejury.score._resolve_clap_ckpt`` so this script is
    self-contained. ``ckpt_dir`` defaults to the current working directory.
    """
    ckpt_dir = ckpt_dir or "."
    path = os.path.join(ckpt_dir, CLAP_CKPT_NAME)
    if os.path.exists(path):
        return path
    print(f"[clap_score] LAION-CLAP-Music checkpoint not found at {path}, downloading...")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.hub.download_url_to_file(CLAP_CKPT_URL, path)
    return path


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def score(
    audio_path: str,
    prompt: str,
    *,
    device: str = "cuda",
    clap_ckpt_path: str | None = None,
) -> float:
    import laion_clap
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    clap.load_ckpt(ckpt=clap_ckpt_path or _resolve_clap_ckpt())
    clap.to(device).eval()
    wav, sr = torchaudio.load(audio_path)
    if sr != 48_000:
        wav = torchaudio.functional.resample(wav, sr, 48_000)
    wav = wav.mean(dim=0, keepdim=True).cpu().numpy()
    audio_emb = clap.get_audio_embedding_from_data(x=wav, use_tensor=False)[0]
    text_emb = clap.get_text_embedding([prompt], use_tensor=False)[0]
    return _cosine(audio_emb, text_emb)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, help="JSON with prompt_id and prompt.")
    p.add_argument("--candidates_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--clap_ckpt",
        default=None,
        help=(
            "Path to the LAION-CLAP-Music checkpoint "
            f"({CLAP_CKPT_NAME}). Downloaded into the current "
            "directory if not provided."
        ),
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    cand_dir = Path(args.candidates_dir)

    rows = []
    for entry in prompts:
        pid = entry["prompt_id"]
        clip = cand_dir / f"{pid}.wav"
        if not clip.exists():
            print(f"[clap_score] missing: {clip}")
            continue
        s = score(
            str(clip),
            entry["prompt"],
            device=args.device,
            clap_ckpt_path=args.clap_ckpt,
        )
        rows.append({"prompt_id": pid, "clap_score": s})

    mean = float(np.mean([r["clap_score"] for r in rows]))
    out = {"per_clip": rows, "mean": mean}
    print(json.dumps({"mean_clap_score": mean}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
