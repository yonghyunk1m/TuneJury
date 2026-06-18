#!/usr/bin/env python3
"""Re-render 300 MidiCaps tracks with FluidR3_GM + TimGM6mb soundfonts and rescore
under TuneJury, for the §A.I MidiCaps soundfont sensitivity probe.
"""
import argparse, csv, json, os, subprocess, sys, tempfile
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--meta", required=True, help="Path to MidiCaps train.json metadata")
ap.add_argument("--midi_root", required=True, help="Path to MidiCaps MIDI files root")
ap.add_argument("--fluidr3", default="/etc/alternatives/default-GM.sf2")
ap.add_argument("--timgm6mb", required=True, help="Path to TimGM6mb.sf2 (e.g. <site-packages>/pretty_midi/TimGM6mb.sf2)")
ap.add_argument("--out", required=True,
                help="Output JSON path for paired soundfont comparison results.")
ap.add_argument("--tunejury_ckpt", default=str(REPO_ROOT / "checkpoints/tunejury.pt"))
args = ap.parse_args()

_orig_load = torch.load
def _patched_load(*a, **k):
    k['weights_only'] = False
    return _orig_load(*a, **k)
torch.load = _patched_load

sys.path.insert(0, str(REPO_ROOT))
from tunejury import Scorer
print("[init] loading TuneJury Scorer ...", flush=True)
scorer = Scorer.from_pretrained(args.tunejury_ckpt)
print("[init] loaded.", flush=True)

print(f"[meta] loading {args.meta}", flush=True)
meta_entries = []
with open(args.meta) as f:
    for line in f:
        meta_entries.append(json.loads(line))
print(f"[meta] {len(meta_entries)} entries; using first {args.n}", flush=True)

import shutil
if not os.path.exists(args.fluidr3):
    print(f"FluidR3 missing: {args.fluidr3}"); sys.exit(1)
if not os.path.exists(args.timgm6mb):
    print(f"TimGM6mb missing: {args.timgm6mb}"); sys.exit(1)


def render(midi_path: str, out_wav: str, sf: str) -> bool:
    try:
        subprocess.run(
            ["fluidsynth", "-i", "-ni", "-F", out_wav, "-r", "44100", "-T", "wav", sf, midi_path],
            check=True, capture_output=True, timeout=120,
        )
        return True
    except Exception:
        return False


results = []
import time
t0 = time.time()
with tempfile.TemporaryDirectory() as tmp:
    for i, m in enumerate(meta_entries[: args.n]):
        midi_path = os.path.join(args.midi_root, m["location"])
        if not os.path.exists(midi_path):
            continue
        track_id = os.path.basename(m["location"]).replace(".mid", "")
        wav_fr = os.path.join(tmp, f"{track_id}_fr.wav")
        wav_tg = os.path.join(tmp, f"{track_id}_tg.wav")
        ok_fr = render(midi_path, wav_fr, args.fluidr3)
        ok_tg = render(midi_path, wav_tg, args.timgm6mb)
        if not (ok_fr and ok_tg):
            continue
        try:
            s_fr = scorer.score(wav_fr, prompt="")
            s_tg = scorer.score(wav_tg, prompt="")
            results.append({
                "track_id": f"midicaps_{track_id}",
                "fluidr3": s_fr,
                "timgm6mb": s_tg,
                "diff": s_tg - s_fr,
            })
        except Exception as e:
            print(f"  scoring fail: {e}", flush=True)
        finally:
            for f_ in (wav_fr, wav_tg):
                if os.path.exists(f_):
                    os.unlink(f_)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = int((args.n - i - 1) / max(rate, 0.001))
            print(f"  [{i+1}/{args.n}] ok={len(results)} rate={rate:.2f}/s ETA={eta}s", flush=True)

print(f"[done] {len(results)} paired scores, elapsed={time.time()-t0:.1f}s", flush=True)

# Aggregate stats
import math
fr_vals = [r["fluidr3"] for r in results]
tg_vals = [r["timgm6mb"] for r in results]
diffs = [r["diff"] for r in results]

def mean(v): return sum(v) / len(v)
def std(v):
    m = mean(v); return math.sqrt(sum((x-m)**2 for x in v) / (len(v)-1))

mfr, mtg, md = mean(fr_vals), mean(tg_vals), mean(diffs)
sd = std(diffs)
# Paired t
t = md / (sd / math.sqrt(len(diffs)))
# Spearman
def spearman(a, b):
    n = len(a)
    ra = sorted(range(n), key=lambda i: a[i])
    rb = sorted(range(n), key=lambda i: b[i])
    rank_a = [0]*n; rank_b = [0]*n
    for r, idx in enumerate(ra): rank_a[idx] = r + 1
    for r, idx in enumerate(rb): rank_b[idx] = r + 1
    mean_ra = (n+1)/2
    num = sum((rank_a[i]-mean_ra)*(rank_b[i]-mean_ra) for i in range(n))
    den_a = math.sqrt(sum((rank_a[i]-mean_ra)**2 for i in range(n)))
    den_b = math.sqrt(sum((rank_b[i]-mean_ra)**2 for i in range(n)))
    return num / (den_a * den_b)

rho = spearman(fr_vals, tg_vals)
# Top-10% overlap
n_top = max(1, int(0.1 * len(results)))
top_fr = set(sorted(range(len(results)), key=lambda i: -fr_vals[i])[:n_top])
top_tg = set(sorted(range(len(results)), key=lambda i: -tg_vals[i])[:n_top])
overlap = len(top_fr & top_tg) / n_top

stats = {
    "n": len(results),
    "fluidr3_mean": mfr, "timgm6mb_mean": mtg,
    "diff_mean": md, "diff_std": sd,
    "paired_t": t,
    "spearman_rho": rho,
    "top10pct_overlap": overlap,
}
print(f"\n=== Stats ===")
for k, v in stats.items(): print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

Path(args.out).write_text(json.dumps({"stats": stats, "per_track": results}, indent=2))
print(f"[saved] {args.out}")
