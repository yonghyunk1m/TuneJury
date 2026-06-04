#!/usr/bin/env python3
"""Render MidiCaps MIDI files to audio for scoring.

MidiCaps ships MIDI, not audio, so the rendered waveform (and hence its reward
score) depends on the synthesiser and soundfont. The released MidiCaps scores
use FluidSynth with the FluidR3_GM General MIDI soundfont at 44.1 kHz. Pin the
same soundfont to reproduce the released scores exactly.

  Reference environment:
    fluidsynth 2.3.4
    FluidR3_GM.sf2  md5 af289497caf8c76d97fdc67ec8409f05

Usage:
  python render_midicaps.py --meta train.json --midi-root /data/midicaps \
      --soundfont /usr/share/sounds/sf2/FluidR3_GM.sf2 --out-dir /data/midicaps_wav
"""
import argparse
import json
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True, help="MidiCaps train.json (JSONL with 'location')")
ap.add_argument("--midi-root", required=True)
ap.add_argument("--soundfont", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--rate", type=int, default=44100)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
ok = fail = 0
for line in open(args.meta):
    m = json.loads(line)
    stem = os.path.basename(m["location"]).replace(".mid", "")
    midi = os.path.join(args.midi_root, m["location"])
    wav = os.path.join(args.out_dir, f"midicaps_{stem}.wav")
    if os.path.exists(wav) or not os.path.exists(midi):
        continue
    try:
        subprocess.run(["fluidsynth", "-i", "-ni", "-F", wav, "-r", str(args.rate),
                        "-T", "wav", args.soundfont, midi],
                       check=True, capture_output=True, timeout=600)
        ok += 1
    except Exception:  # noqa: BLE001
        fail += 1
print(f"rendered {ok}, failed {fail}")
