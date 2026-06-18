#!/usr/bin/env python3
"""Resume MidiCaps rescoring for any track not in the released CSV.

Identifies missing track_ids vs release_scores/midicaps_scores.csv, renders
MIDI via FluidSynth, scores via TuneJury Scorer (last-layer MERT mean +
512-d zero-vector text), and writes the missing rows to --rescored_csv.
The released midicaps_scores.csv already covers the full 5,000-track
canonical release; this script is only needed if rescoring a different
MidiCaps subset or recovering from an interrupted re-score run.
"""
import argparse, csv, json, os, subprocess, sys, tempfile, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--orig_csv", default=str(REPO_ROOT / "release_scores/midicaps_scores.csv"))
ap.add_argument("--rescored_csv", required=True,
                help="Output CSV with newly scored rows appended (will be created if absent).")
ap.add_argument("--meta", required=True, help="Path to MidiCaps train.json metadata")
ap.add_argument("--midi_root", required=True, help="Path to MidiCaps MIDI files root")
ap.add_argument("--soundfont", default="/etc/alternatives/default-GM.sf2")
ap.add_argument("--tunejury_ckpt", default=str(REPO_ROOT / "checkpoints/tunejury.pt"))
ap.add_argument("--max_dur", type=float, default=30.0)
args = ap.parse_args()

import torch
_orig_load = torch.load
def _patched_load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _patched_load

sys.path.insert(0, str(REPO_ROOT))
from tunejury import Scorer

print("[init] loading TuneJury Scorer ...", flush=True)
scorer = Scorer.from_pretrained(args.tunejury_ckpt)
print("[init] loaded.", flush=True)

print(f"[meta] loading {args.meta}", flush=True)
meta_by_id = {}
with open(args.meta) as f:
    for line in f:
        m = json.loads(line)
        tid = os.path.basename(m["location"]).replace(".mid", "")
        meta_by_id[f"midicaps_{tid}"] = m
print(f"[meta] {len(meta_by_id)} entries", flush=True)

done = set()
with open(args.rescored_csv) as f:
    for r in csv.DictReader(f):
        done.add(r["track_id"])
print(f"[resume] {len(done)} already in rescored CSV", flush=True)

orig_ids = []
with open(args.orig_csv) as f:
    for r in csv.DictReader(f):
        orig_ids.append(r["track_id"])
print(f"[orig] {len(orig_ids)} ids in original CSV", flush=True)

missing = [tid for tid in orig_ids if tid not in done]
print(f"[missing] {len(missing)} ids to score", flush=True)

csv_f = open(args.rescored_csv, "a", newline="")
writer = csv.DictWriter(csv_f, fieldnames=["track_id", "reward_score", "tempo", "key", "duration", "genre", "caption"])

ok = fail = 0
t0 = time.time()
with tempfile.TemporaryDirectory() as tmp:
    for i, tid in enumerate(missing):
        m = meta_by_id.get(tid)
        if m is None:
            fail += 1
            continue
        midi_path = os.path.join(args.midi_root, m["location"])
        if not os.path.exists(midi_path):
            fail += 1
            continue
        track_id_short = os.path.basename(m["location"]).replace(".mid", "")
        wav_path = os.path.join(tmp, f"{track_id_short}.wav")
        try:
            subprocess.run(
                ["fluidsynth", "-i", "-ni", "-F", wav_path, "-r", "44100",
                 "-T", "wav", args.soundfont, midi_path],
                check=True, capture_output=True, timeout=120,
            )
        except Exception:
            fail += 1
            continue
        try:
            score = scorer.score(wav_path, prompt="")
            writer.writerow({
                "track_id": tid,
                "reward_score": float(score),
                "tempo": m.get("tempo"),
                "key": m.get("key"),
                "duration": m.get("duration"),
                "genre": ",".join(m.get("genre", [])),
                "caption": m.get("caption", ""),
            })
            csv_f.flush()
            ok += 1
        except Exception:
            fail += 1
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = int((len(missing) - i - 1) / max(rate, 0.001))
            print(f"  [{i+1}/{len(missing)}] ok={ok} fail={fail} rate={rate:.2f}/s ETA={eta}s", flush=True)

csv_f.close()
elapsed = time.time() - t0
print(f"[done] ok={ok} fail={fail} elapsed={elapsed:.1f}s rate={ok/max(elapsed,1):.2f}/s")
