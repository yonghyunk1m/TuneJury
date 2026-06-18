#!/usr/bin/env python3
"""Re-score a random sample and confirm it matches a released CSV.

Because the scoring protocol is deterministic (fixed center CLAP window, full
MERT, zero text), re-scoring any released track reproduces its CSV value to
floating-point tolerance. This script picks N random rows, re-scores them, and
reports the worst absolute difference.

Usage:
  python verify_release_scores.py --csv release_scores/openmic_scores.csv \
      --manifest openmic_tracks.jsonl --n 16
"""
import argparse
import csv
import json
import random

import torch
import torchaudio

CLAP_SR, MERT_SR = 48000, 24000
CLAP_LEN = 10 * CLAP_SR
MERT_CHUNK = 300 * MERT_SR

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--manifest", required=True, help="JSONL {track_id, path} covering the CSV ids")
ap.add_argument("--repo-root", default=".")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
import sys
sys.path.insert(0, args.repo_root)
from tunejury import Scorer

sc = Scorer.from_pretrained(f"{args.repo_root}/checkpoints/tunejury.pt")
device = sc.device
path_of = {json.loads(l)["track_id"]: json.loads(l)["path"] for l in open(args.manifest)}


@torch.no_grad()
def score_path(path):
    raw, sr = torchaudio.load(path)
    raw = raw.mean(dim=0, keepdim=True)
    wav = raw if sr == CLAP_SR else torchaudio.functional.resample(raw, sr, CLAP_SR)
    if wav.shape[-1] > CLAP_LEN:
        s = (wav.shape[-1] - CLAP_LEN) // 2
        wav = wav[..., s:s + CLAP_LEN]
    clap_a = torch.tensor(sc.clap_model.get_audio_embedding_from_data(
        x=wav.cpu().numpy(), use_tensor=False)[0], dtype=torch.float32).flatten()
    wavm = (raw if sr == MERT_SR else torchaudio.functional.resample(raw, sr, MERT_SR)).squeeze(0)
    T = wavm.shape[-1]
    bounds = list(range(0, T, MERT_CHUNK)) + [T]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < MERT_SR:
        bounds.pop(-2)
    acc, n = torch.zeros(1024), 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        inp = sc.mert_processor(wavm[a:b].cpu().numpy(), sampling_rate=MERT_SR,
                                return_tensors="pt").to(device)
        f = sc.mert_model(**inp, output_hidden_states=False).last_hidden_state.squeeze(0)
        acc += f.sum(dim=0).float().cpu()
        n += f.shape[0]
    feats = torch.cat([clap_a, acc / n, torch.zeros(512)]).unsqueeze(0)
    return sc.head(feats.to(device)).item()


rows = [r for r in csv.DictReader(open(args.csv)) if r["track_id"] in path_of]
random.seed(args.seed)
worst = 0.0
for r in random.sample(rows, min(args.n, len(rows))):
    s = score_path(path_of[r["track_id"]])
    d = abs(s - float(r["reward_score"]))
    worst = max(worst, d)
    print(f"{r['track_id']}: csv {float(r['reward_score']):+.6f} re {s:+.6f} diff {d:.2e}")
print(f"\nworst diff {worst:.2e} -> {'PASS' if worst < args.tol else 'FAIL'}")
