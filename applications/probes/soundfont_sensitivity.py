"""Soundfont-sensitivity probe for the MidiCaps stream (paper Appendix I).

MidiCaps is symbolic MIDI, so any audio-domain reward (TuneJury included)
also depends on the synthesizer. We re-render the first 300 MidiCaps
tracks under two General-MIDI soundfonts and compare paired rewards.

Two soundfonts (exact files used in the paper):

  1. FluidR3_GM (142 MB) - the canonical FluidSynth GM bank.
     Source: /usr/share/sounds/sf2/FluidR3_GM.sf2
     Install: apt-get install fluid-soundfont-gm

  2. TimGM6mb  (5.7 MB) - a much smaller, lower-fidelity GM bank.
     Source: <site-packages>/pretty_midi/TimGM6mb.sf2
     Install: pip install pretty_midi

Headline (paper Appendix I):

    Aggregate means are close but not identical (paired t = +1.73,
    p approx 0.085), and track-level rankings are noisy
    (cross-soundfont Spearman = +0.69; 33% top-10 overlap).

The paper's takeaway is that distribution-level comparisons are safe
to use across renderers, but track-level rankings should be treated
as (score, renderer) joint quantities and the renderer is documented
in the release metadata.

Prerequisites
-------------
- ``fluidsynth`` on PATH (apt install fluidsynth).
- FluidR3_GM.sf2 (apt install fluid-soundfont-gm) and TimGM6mb.sf2
  (pip install pretty_midi; bundled at
  <site-packages>/pretty_midi/TimGM6mb.sf2).
- The first 300 MidiCaps .mid files. See docs/reproducing.md section
  11 for the MidiCaps download.
- The TuneJury checkpoint (checkpoints/tunejury.pt) and the
  LAION-CLAP-Music + MERT-v1-330M weights (auto-downloaded on first run
  by ``tunejury.Scorer``).

Usage
-----
$ python soundfont_sensitivity.py \\
      --midi_dir   /path/to/midicaps_midi_first_300 \\
      --sf_fluidr3 /usr/share/sounds/sf2/FluidR3_GM.sf2 \\
      --sf_timgm6  <site-packages>/pretty_midi/TimGM6mb.sf2 \\
      --checkpoint checkpoints/tunejury.pt
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, ttest_rel


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--midi_dir", type=Path, required=True,
                   help="Dir with the first 300 MidiCaps .mid files.")
    p.add_argument("--sf_fluidr3", type=Path, required=True,
                   help="Path to FluidR3_GM.sf2 (apt fluid-soundfont-gm).")
    p.add_argument("--sf_timgm6", type=Path, required=True,
                   help="Path to TimGM6mb.sf2 (pip pretty_midi bundle).")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="TuneJury checkpoint (e.g. checkpoints/tunejury.pt).")
    p.add_argument("--clap_ckpt", type=Path, default=None,
                   help="Optional explicit LAION-CLAP-Music checkpoint. If "
                   "omitted, Scorer.from_pretrained auto-resolves from the "
                   "directory of --checkpoint.")
    p.add_argument("--out_root", type=Path,
                   default=Path("applications/probes/results/soundfont_probe"))
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--prompt", type=str, default="",
                   help="Text prompt passed to TuneJury (default empty per OOD "
                   "recommendation in paper Section 4.2).")
    return p.parse_args()


def render_with_sf(midi_path: Path, sf_path: Path, wav_out: Path) -> None:
    """Use FluidSynth CLI to render a MIDI file with a given SoundFont."""
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["fluidsynth", "-ni", "-F", str(wav_out),
           "-T", "wav", str(sf_path), str(midi_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def score_dir(wav_dir: Path, scorer, prompt: str = "") -> dict[str, float]:
    """Score every .wav in ``wav_dir`` with a loaded ``Scorer``.

    Returns ``{wav_stem: reward}``. Uses the public
    ``tunejury.Scorer.score`` entry point, which extracts CLAP + MERT
    features under the same 2048-d input convention as the rest of the
    repo (see tunejury/score.py).
    """
    out: dict[str, float] = {}
    wavs = sorted(wav_dir.glob("*.wav"))
    for i, wav in enumerate(wavs, 1):
        try:
            out[wav.stem] = float(scorer.score(str(wav), prompt=prompt))
        except Exception as exc:
            print(f"[score_dir] {wav.name}: failed ({exc})", flush=True)
            continue
        if i % 25 == 0 or i == len(wavs):
            print(f"  scored {i}/{len(wavs)}", flush=True)
    return out


def main():
    args = parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    fluidr3_wav = args.out_root / "fluidr3_wav"
    timgm6_wav = args.out_root / "timgm6_wav"

    midis = sorted(args.midi_dir.glob("*.mid"))[:args.limit]
    print(f"rendering {len(midis)} MIDIs under both soundfonts...", flush=True)
    for m in midis:
        out_a = fluidr3_wav / f"{m.stem}.wav"
        out_b = timgm6_wav / f"{m.stem}.wav"
        if not out_a.exists():
            render_with_sf(m, args.sf_fluidr3, out_a)
        if not out_b.exists():
            render_with_sf(m, args.sf_timgm6, out_b)

    print("loading TuneJury Scorer (CLAP + MERT + head)...", flush=True)
    # Lazy import: Scorer pulls torchaudio, transformers, laion_clap.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tunejury import Scorer
    scorer = Scorer.from_pretrained(
        head_ckpt_path=str(args.checkpoint),
        clap_ckpt_path=str(args.clap_ckpt) if args.clap_ckpt else None,
    )

    print("scoring FluidR3 renders...", flush=True)
    s_fluid = score_dir(fluidr3_wav, scorer, prompt=args.prompt)
    print("scoring TimGM6mb renders...", flush=True)
    s_tim = score_dir(timgm6_wav, scorer, prompt=args.prompt)

    keys = sorted(set(s_fluid) & set(s_tim))
    f_vals = np.array([s_fluid[k] for k in keys])
    t_vals = np.array([s_tim[k] for k in keys])
    print(f"\nn = {len(keys)}")
    if len(keys) < 2:
        print("Not enough successful scores to compute statistics.")
        return
    print(f"FluidR3 mean:  {f_vals.mean():+.4f}")
    print(f"TimGM6mb mean: {t_vals.mean():+.4f}")
    t_stat = ttest_rel(f_vals, t_vals)
    rho = spearmanr(f_vals, t_vals)
    print(f"paired t: {t_stat}")
    print(f"Spearman: {rho}")
    diffs = f_vals - t_vals
    print(f"|delta| std: {diffs.std():.4f}")

    summary_path = args.out_root / "summary.json"
    import json
    summary = {
        "n": len(keys),
        "fluidr3_mean": float(f_vals.mean()),
        "timgm6_mean": float(t_vals.mean()),
        "paired_t": float(t_stat.statistic),
        "paired_t_p": float(t_stat.pvalue),
        "spearman": float(rho.correlation),
        "spearman_p": float(rho.pvalue),
        "delta_std": float(diffs.std()),
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
