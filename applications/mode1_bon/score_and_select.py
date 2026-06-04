"""Score and select best-of-N candidates with the released TuneJury.

Scores every candidate produced by one of the ``generate_*.py`` scripts
with the frozen TuneJury head, then for each ``N`` in ``--N`` takes the
top-1 by reward (paper Section 5.1 / Appendix G BoN protocol). Writes
per-candidate scores to ``--per_cand_csv`` and per-N top-1 selections
to ``--out_csv``.

Usage:
    python score_and_select.py \\
        --ckpt ../../checkpoints/tunejury.pt \\
        --prompts ../../eval/prompts/sdd100.json \\
        --candidates_dir results/musicgen_medium_bon100 \\
        --n_candidates 32 --N 1 2 4 8 16 32 \\
        --per_cand_csv results/musicgen_medium_bon100/tunejury_per_candidate.csv \\
        --out_csv results/musicgen_medium_bon100/results.csv
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from tunejury import TuneJury  # noqa: E402


def load_models(device, reward_ckpt: Path):
    import laion_clap
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    clap_ckpt = REPO_ROOT / "checkpoints" / "music_audioset_epoch_15_esc_90.14.pt"
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    clap.load_ckpt(ckpt=str(clap_ckpt))
    torch.load = _orig_load
    clap.eval()

    mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device).eval()
    proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

    rm = TuneJury().to(device)
    state = torch.load(reward_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    rm.load_state_dict(state)
    rm.eval()
    return clap, mert, proc, rm


def score_one(audio_path: str, prompt: str, clap, mert, proc, rm, device) -> float:
    import librosa

    with torch.no_grad():
        audio_emb = clap.get_audio_embedding_from_filelist([audio_path], use_tensor=True).cpu().numpy()[0]
        text_emb = clap.get_text_embedding([prompt], use_tensor=True).cpu().numpy()[0]

    waveform, sr = librosa.load(audio_path, sr=24000, mono=True)
    inputs = proc(waveform, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mert(**inputs, output_hidden_states=True)
    mert_emb = out.last_hidden_state.mean(dim=1).cpu().numpy()[0]

    feat = np.concatenate([audio_emb, mert_emb, text_emb], axis=0)
    x = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return rm(x).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--candidates_dir", type=str, required=True)
    ap.add_argument("--N", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n_candidates", type=int, default=32,
                    help="how many candidates per prompt to score (must be >= max N)")
    ap.add_argument("--per_cand_csv", type=str, required=True,
                    help="output CSV with one row per (prompt_id, c) pair")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="output CSV with top-1 per N selection")
    ap.add_argument("--prompt_prefix", type=str, default="high quality instrumental music, ")
    ap.add_argument("--ckpt", type=str,
                    default=str(REPO_ROOT / "checkpoints" / "tunejury.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cand_dir = Path(args.candidates_dir)
    prompts = json.loads(Path(args.prompts).read_text())

    print(f"[score] reward ckpt: {args.ckpt}", flush=True)
    print(f"[score] candidates_dir: {cand_dir}", flush=True)
    print(f"[score] {len(prompts)} prompts x {args.n_candidates} candidates", flush=True)

    print(f"[score] loading models ...", flush=True)
    t0 = time.time()
    clap, mert, proc, rm = load_models(device, Path(args.ckpt))
    print(f"[score] models loaded in {time.time() - t0:.1f}s")

    per_cand_csv = Path(args.per_cand_csv)
    per_cand_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(prompts) * args.n_candidates
    n_done = 0
    t_start = time.time()

    # Buffer scores
    all_scores = {}  # pid -> list[float]
    with per_cand_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "c", "tunejury_score", "wav"])
        for entry in prompts:
            pid = entry["prompt_id"]
            prompt = args.prompt_prefix + entry["prompt"]
            scores = []
            for c in range(args.n_candidates):
                wav = cand_dir / f"{pid}_c{c}.wav"
                if not wav.exists():
                    print(f"[score] MISSING {wav.name}", flush=True)
                    scores.append(float("-inf"))
                    continue
                s = score_one(str(wav), prompt, clap, mert, proc, rm, device)
                scores.append(s)
                writer.writerow([pid, c, s, str(wav)])
                n_done += 1
                if n_done % 20 == 0 or n_done == n_total:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed
                    eta = (n_total - n_done) / rate if rate > 0 else 0
                    print(f"[score] {n_done}/{n_total}  rate={rate:.2f}/s  eta={eta/60:.1f}min", flush=True)
            all_scores[pid] = scores

    # Track per-N top-1 picks so we can materialize topN_{n}/ directories
    # consumed downstream by eval.distributional, eval.clap_score, and
    # applications/mode1_bon/mpcd_diagnostic.py (Appendix G MPCD table).
    topn_picks: dict[int, list[tuple[str, int, Path]]] = {n: [] for n in args.N}
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "prompt", "N", "best_c", "best_score", "selected_path"])
        for entry in prompts:
            pid = entry["prompt_id"]
            scores = all_scores[pid]
            for n in args.N:
                top_c = max(range(n), key=lambda i: scores[i])
                src_wav = cand_dir / f"{pid}_c{top_c}.wav"
                writer.writerow([
                    pid, entry["prompt"], n, top_c, f"{scores[top_c]:+.4f}",
                    str(src_wav),
                ])
                topn_picks[n].append((pid, top_c, src_wav))

    # Materialize results/{bb}_bon100/topN_{n}/{pid}_c{best_c}.wav as symlinks
    # (falling back to copy on filesystems that do not support symlinks). These
    # directories are required by the Mode 1 distributional + MPCD pipeline.
    for n, picks in topn_picks.items():
        top_dir = cand_dir / f"topN_{n}"
        top_dir.mkdir(parents=True, exist_ok=True)
        for pid, best_c, src_wav in picks:
            dst = top_dir / f"{pid}_c{best_c}.wav"
            if not src_wav.exists():
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                # Use relative symlink so the tree stays portable across moves.
                rel_src = os.path.relpath(src_wav.resolve(), start=dst.parent.resolve())
                os.symlink(rel_src, dst)
            except (OSError, NotImplementedError):
                shutil.copyfile(src_wav, dst)
        print(f"[score] materialized {len(picks)} picks -> {top_dir}", flush=True)

    print(f"[score] DONE  per-cand -> {per_cand_csv}  per-N -> {out_csv}")


if __name__ == "__main__":
    main()
