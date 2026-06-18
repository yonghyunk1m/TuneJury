#!/usr/bin/env python3
"""Score an open-license audio collection with the released TuneJury checkpoint.

This is the exact, deterministic protocol used to produce every CSV in
``release_scores/``. Given a manifest of ``{track_id, path}`` rows (or a
directory glob), it writes ``track_id,reward_score`` and is resume-safe.

Protocol (uniform across all seven released collections):
  * CLAP audio branch: the center 10 s window at 48 kHz mono. Clips of 10 s
    or less are encoded whole. The window is fixed (center), so scoring is
    deterministic, unlike a random-crop encode.
  * MERT audio branch: the full track at 24 kHz mono, frame embeddings
    mean-pooled. Tracks longer than 300 s are encoded in consecutive 300 s
    segments whose frame means are length-weighted averaged (a final segment
    shorter than 1 s is merged into the previous one) to bound peak memory.
  * Text branch: the 512-d zero vector (the empty-prompt release protocol).
  * Head: the released ``checkpoints/tunejury.pt``.

Usage:
  # from a manifest (one JSON object per line: {"track_id": ..., "path": ...})
  python score_release_collection.py --manifest tracks.jsonl --out scores.csv

  # or from a directory of audio files
  python score_release_collection.py --audio-dir /data/openmic --pattern '**/*.ogg' \
      --id-prefix openmic --out openmic_scores.csv

Set ``--shard i --num-shards N`` to split work across N GPUs (run one process
per GPU with the matching ``CUDA_VISIBLE_DEVICES``).
"""
import argparse
import csv
import glob
import json
import os
import sys

import torch
import torchaudio

CLAP_SR, MERT_SR = 48000, 24000
CLAP_LEN = 10 * CLAP_SR
MERT_CHUNK = 300 * MERT_SR


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="JSONL with {track_id, path} per line")
    src.add_argument("--audio-dir", help="Directory to glob for audio files")
    ap.add_argument("--pattern", default="**/*.wav", help="Glob under --audio-dir")
    ap.add_argument("--id-prefix", default="", help="Track-id prefix for --audio-dir mode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=os.environ.get("TUNEJURY_ROOT", "."),
                    help="Path to the tune-jury repo (holds checkpoints/tunejury.pt)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    return ap.parse_args()


def build_rows(args):
    if args.manifest:
        rows = [json.loads(line) for line in open(args.manifest)]
    else:
        rows = []
        for p in sorted(glob.glob(os.path.join(args.audio_dir, args.pattern), recursive=True)):
            stem = os.path.splitext(os.path.basename(p))[0]
            tid = f"{args.id_prefix}_{stem}" if args.id_prefix else stem
            rows.append({"track_id": tid, "path": p})
    return [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]


def main():
    args = parse_args()
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    sys.path.insert(0, args.repo_root)
    from tunejury import Scorer

    sc = Scorer.from_pretrained(os.path.join(args.repo_root, "checkpoints/tunejury.pt"))
    device = sc.device

    def load_mono(path, sr_target, raw=None, raw_sr=None):
        if raw is None:
            raw, raw_sr = torchaudio.load(path)
            raw = raw.mean(dim=0, keepdim=True)
        if raw_sr != sr_target:
            return torchaudio.functional.resample(raw, raw_sr, sr_target), raw, raw_sr
        return raw, raw, raw_sr

    @torch.no_grad()
    def score_path(path):
        raw, raw_sr = torchaudio.load(path)
        raw = raw.mean(dim=0, keepdim=True)
        wav, _, _ = load_mono(path, CLAP_SR, raw, raw_sr)
        T = wav.shape[-1]
        if T > CLAP_LEN:
            s = (T - CLAP_LEN) // 2
            wav = wav[..., s:s + CLAP_LEN]
        emb = sc.clap_model.get_audio_embedding_from_data(x=wav.cpu().numpy(), use_tensor=False)
        clap_a = torch.tensor(emb[0], dtype=torch.float32).flatten()

        wavm, _, _ = load_mono(path, MERT_SR, raw, raw_sr)
        wavm = wavm.squeeze(0)
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

    rows = build_rows(args)
    done = set()
    if os.path.exists(args.out):
        done = {r["track_id"] for r in csv.DictReader(open(args.out))}
    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    w = csv.DictWriter(out, fieldnames=["track_id", "reward_score"])
    if new_file:
        w.writeheader()
    fail = open(args.out + ".fail", "a")

    for r in rows:
        if r["track_id"] in done:
            continue
        try:
            s = score_path(r["path"])
            w.writerow({"track_id": r["track_id"], "reward_score": repr(s)})
            out.flush()
        except Exception as e:  # noqa: BLE001
            fail.write(f"{r['track_id']}\t{type(e).__name__}: {e}\n")
            fail.flush()
            torch.cuda.empty_cache()
    print(f"[shard {args.shard}/{args.num_shards}] done")


if __name__ == "__main__":
    main()
