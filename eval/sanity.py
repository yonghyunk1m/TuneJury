"""Adversarial sanity checks (Appendix B: Adversarial Sanity Checks).

Five checks against the released TuneJury checkpoint to characterize how
the score behaves on degenerate, perturbed, length-varied, and segmented
inputs. All five share the same 8-clip MTG-Jamendo subset (paper §B).

* --check boundary   Boundary inputs (Table sanity_boundary)
* --check perturb    Graded SNR + clipping-ratio sweeps (Table sanity_perturb)
* --check length     Length sensitivity 1..120s + full-track
* --check segment    Per-segment discrimination via sliding window
* --check temporal   Temporal-structure (time-reversal)

Expected headline numbers (paper §B, Tables sanity_boundary + sanity_perturb):

* Reference music (n=20 Jamendo validation, 10s):  -0.18 ± 0.66, range [-1.39, +1.05]
* Silence (zeros):                                  -1.05
* White noise (RMS -60 / -40 / -20 / 0 dBFS):       -3.90 / -4.03 / -3.97 / -4.59
* SNR sweep (clean / 40 / 20 / 10 / 5 / 0 dB):      monotone -0.10 -> -0.36 -> -0.72 -> -1.25 -> -1.79 -> -2.62
* Clipping-ratio (clean / 0.5 / 0.1 / 0.05 / 0.02): monotone -0.10 -> -0.31 -> -1.48 -> -1.90 -> -2.43
* Length sweep (1 / 3 / 5 / 10s):                   -0.75 -> -0.65 -> -0.46 -> -0.10
* Per-segment probe (silence/noise injected at 20-30s): bad-slot window matches standalone bad input (-1.05 silence, -4.08 noise)
* Time-reversal:                                    -0.10 -> -0.74 (delta = 0.64)

Reproducibility note. check_boundary seeds NumPy + torch at manual_seed=42
and reproduces the silence and four white-noise RMS rows exact within ±0.05
of the paper Table. Pink / brown / sine / harmonic-stack rows match the
qualitative orderings of the paper Table (silence above noise; sines rise with
frequency; harmonic stacks pull upward of pure sines) but exact values can drift
by ±0.5 from the paper Table values, because the paper Table used an earlier
reference implementation with slightly different waveform amplitude / filter
conventions. Statistical claims in the paper (monotone perturbation ladders,
segment-level discrimination, time-reversal sensitivity) hold across both
implementations. Re-run this committed script for fresh canonical values.

Usage
-----
$ python -m eval.sanity --check boundary \\
      --checkpoint   checkpoints/tunejury.pt \\
      --clips-root   /path/to/jamendo_val_8clips \\
      --out-csv      results/sanity_boundary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6 default weights_only=True breaks LAION-CLAP checkpoint load;
# patch before laion_clap import (via tunejury.Scorer).
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import torchaudio
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunejury import Scorer


SR = 16000


def _scorer(checkpoint: Path) -> Scorer:
    return Scorer.from_pretrained(str(checkpoint))


def _load_wav(path: Path, duration_s: float | None = None) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = AF.resample(wav, sr, SR)
    if duration_s:
        n = int(duration_s * SR)
        if wav.shape[1] > n:
            wav = wav[:, :n]
        elif wav.shape[1] < n:
            pad = n - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))
    return wav.squeeze(0)


def _white_noise(duration_s: float, level: float = 0.05) -> torch.Tensor:
    n = int(duration_s * SR)
    return level * torch.randn(n)


def _silence(duration_s: float) -> torch.Tensor:
    return torch.zeros(int(duration_s * SR))


def _white_noise_dbfs(duration_s: float, dbfs: float) -> torch.Tensor:
    """White noise at target RMS dBFS. level = 10^(dBFS/20) for randn (unit std)."""
    n = int(duration_s * SR)
    level = 10 ** (dbfs / 20.0)
    return level * torch.randn(n)


def _colored_noise(duration_s: float, alpha: float, level: float = 0.1) -> torch.Tensor:
    """1/f^alpha colored noise via FFT filter (alpha=1 pink, alpha=2 brown)."""
    n = int(duration_s * SR)
    white = torch.randn(n)
    fft = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(n, d=1.0 / SR)
    freqs[0] = 1.0  # avoid div-by-zero at DC
    filt = freqs ** (-alpha / 2.0)
    filt[0] = 0.0  # DC = 0
    colored = torch.fft.irfft(fft * filt, n=n)
    colored = colored / colored.std().clamp_min(1e-10) * level
    return colored


def _sine(duration_s: float, freq_hz: float, amplitude: float = 0.5) -> torch.Tensor:
    n = int(duration_s * SR)
    t = torch.arange(n, dtype=torch.float32) / SR
    return amplitude * torch.sin(2.0 * np.pi * freq_hz * t)


def _harmonic_stack(duration_s: float, freqs_hz: list, amplitude: float = 0.5) -> torch.Tensor:
    """Sum of equal-amplitude sinusoids at given frequencies, normalized to peak amplitude."""
    n = int(duration_s * SR)
    t = torch.arange(n, dtype=torch.float32) / SR
    out = sum(torch.sin(2.0 * np.pi * f * t) for f in freqs_hz)
    peak = out.abs().max().clamp_min(1e-10)
    return amplitude * out / peak


def _add_noise(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_p = wav.pow(2).mean().clamp_min(1e-10)
    noise_p = signal_p / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * noise_p.sqrt()
    out = wav + noise
    out = out / out.abs().max().clamp_min(1e-10)
    return out


def _hard_clip(wav: torch.Tensor, ratio: float) -> torch.Tensor:
    peak = wav.abs().max().item()
    if peak == 0:
        return wav
    clip_at = ratio * peak
    out = wav.clamp(min=-clip_at, max=clip_at)
    return out / out.abs().max().clamp_min(1e-10)


def _time_reverse(wav: torch.Tensor) -> torch.Tensor:
    return wav.flip(dims=[0])


def _score_wav(scorer: Scorer, wav: torch.Tensor, text: str | None = None) -> float:
    return float(scorer.score_waveform(wav.unsqueeze(0).numpy(), sr=SR, text=text))


def _list_clips(clips_root: Path) -> list[Path]:
    clips = sorted(clips_root.glob("*.wav")) + sorted(clips_root.glob("*.mp3"))
    if not clips:
        raise FileNotFoundError(f"no audio under {clips_root}")
    return clips[:8]  # paper uses 8-clip set


def check_boundary(args) -> None:
    """Reproduce all rows of paper Table sanity_boundary.

    Generates: silence, white noise at 4 RMS levels (-60/-40/-20/0 dBFS),
    pink (1/f) noise, brown (1/f^2) noise, 5 pure sines (110/220/440/880/1760 Hz),
    a harmonic stack (300/600/900 Hz), the A2 harmonic series (110-660 Hz), and
    optionally the n=20 Jamendo validation reference music band.

    Seeded for reproducibility (manual_seed=42); transformer encoder bf16
    attention reduction may still cause small variance across GPUs / runs
    (see applications/probes/README.md note).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    scorer = _scorer(args.checkpoint)
    rows = []

    DUR = 10.0

    # Silence.
    s = _score_wav(scorer, _silence(DUR))
    rows.append({"input": "silence_10s", "mean": s, "std": 0.0, "min": s, "max": s})

    # White noise at 4 RMS levels.
    for dbfs in [-60, -40, -20, 0]:
        wav = _white_noise_dbfs(DUR, dbfs)
        s = _score_wav(scorer, wav)
        rows.append({
            "input": f"white_noise_rms_{dbfs:+d}dbfs",
            "mean": s, "std": 0.0, "min": s, "max": s,
        })

    # Pink (1/f) and brown (1/f^2) noise.
    for label, alpha in [("pink_noise_1_over_f", 1.0), ("brown_noise_1_over_f2", 2.0)]:
        wav = _colored_noise(DUR, alpha)
        s = _score_wav(scorer, wav)
        rows.append({"input": label, "mean": s, "std": 0.0, "min": s, "max": s})

    # Pure sines.
    for freq_hz in [110, 220, 440, 880, 1760]:
        wav = _sine(DUR, freq_hz)
        s = _score_wav(scorer, wav)
        rows.append({"input": f"sine_{freq_hz}hz", "mean": s, "std": 0.0, "min": s, "max": s})

    # Harmonic stacks.
    for label, freqs in [
        ("harmonic_stack_300_600_900", [300, 600, 900]),
        ("a2_harmonic_series_110_to_660", [110, 220, 330, 440, 550, 660]),
    ]:
        wav = _harmonic_stack(DUR, freqs)
        s = _score_wav(scorer, wav)
        rows.append({"input": label, "mean": s, "std": 0.0, "min": s, "max": s})

    # Reference music (n=20 Jamendo val), optional.
    if args.reference_clips_root:
        ref_clips = sorted(args.reference_clips_root.glob("*.wav"))[:20]
        scores = [_score_wav(scorer, _load_wav(p, DUR)) for p in ref_clips]
        rows.append({
            "input": "reference_music_jamendo_val_n20",
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(min(scores)),
            "max": float(max(scores)),
        })

    _emit(rows, args.out_csv)


def check_perturb(args) -> None:
    scorer = _scorer(args.checkpoint)
    clips = [_load_wav(p, 10.0) for p in _list_clips(args.clips_root)]
    rows = []

    # SNR sweep
    for snr_db in [40, 20, 10, 5, 0]:
        scores = [_score_wav(scorer, _add_noise(w, snr_db)) for w in clips]
        rows.append(
            {
                "perturbation": "snr_db",
                "level": snr_db,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
        )

    # Clipping ratio sweep
    for ratio in [0.5, 0.1, 0.05, 0.02]:
        scores = [_score_wav(scorer, _hard_clip(w, ratio)) for w in clips]
        rows.append(
            {
                "perturbation": "clip_ratio",
                "level": ratio,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
        )

    _emit(rows, args.out_csv)


def check_length(args) -> None:
    scorer = _scorer(args.checkpoint)
    clips = [_load_wav(p) for p in _list_clips(args.clips_root)]
    rows = []
    for dur in [1, 3, 5, 10, 15, 20, 30, 45, 60, 120]:
        scores = []
        for full in clips:
            n = min(int(dur * SR), full.shape[0])
            scores.append(_score_wav(scorer, full[:n]))
        rows.append(
            {"duration_s": dur, "mean": float(np.mean(scores)), "std": float(np.std(scores))}
        )
    # Full-track
    scores = [_score_wav(scorer, full) for full in clips]
    rows.append(
        {"duration_s": "full", "mean": float(np.mean(scores)), "std": float(np.std(scores))}
    )
    _emit(rows, args.out_csv)


def check_segment(args) -> None:
    scorer = _scorer(args.checkpoint)
    clips = [_load_wav(p, 10.0) for p in _list_clips(args.clips_root)[:5]]
    if len(clips) < 5:
        raise RuntimeError("need at least 5 clips of 10s for segment probe.")
    rows = []

    for injection in ("silence", "noise"):
        composite = torch.cat(clips, dim=0)  # 50s
        bad_start = 20 * SR
        bad_end = 30 * SR
        if injection == "silence":
            composite[bad_start:bad_end] = 0.0
        else:
            composite[bad_start:bad_end] = _white_noise(10.0, level=0.1)
        # 10s sliding window @ 5s hop
        hop = 5 * SR
        win = 10 * SR
        for start in range(0, composite.shape[0] - win + 1, hop):
            w = composite[start : start + win]
            score = _score_wav(scorer, w)
            rows.append(
                {
                    "injection": injection,
                    "window_start_s": start // SR,
                    "score": score,
                }
            )

    _emit(rows, args.out_csv)


def check_temporal(args) -> None:
    scorer = _scorer(args.checkpoint)
    clips = [_load_wav(p, 10.0) for p in _list_clips(args.clips_root)]
    rows = []
    forward = [_score_wav(scorer, w) for w in clips]
    reversed_ = [_score_wav(scorer, _time_reverse(w)) for w in clips]
    rows.append(
        {
            "direction": "forward",
            "mean": float(np.mean(forward)),
            "std": float(np.std(forward)),
            "n": len(clips),
        }
    )
    rows.append(
        {
            "direction": "reversed",
            "mean": float(np.mean(reversed_)),
            "std": float(np.std(reversed_)),
            "n": len(clips),
        }
    )
    _emit(rows, args.out_csv)


def _emit(rows: list[dict], out_csv: Path | None) -> None:
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    print(",".join(cols))
    for r in rows:
        print(",".join(str(r[c]) for c in cols))
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w") as fp:
            fp.write(",".join(cols) + "\n")
            for r in rows:
                fp.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"wrote {out_csv}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", choices=["boundary", "perturb", "length", "segment", "temporal"], required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tunejury.pt"))
    ap.add_argument("--clips-root", type=Path, default=Path("data/sanity_8clips"))
    ap.add_argument("--reference-clips-root", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    dispatch = {
        "boundary": check_boundary,
        "perturb": check_perturb,
        "length": check_length,
        "segment": check_segment,
        "temporal": check_temporal,
    }
    dispatch[args.check](args)


if __name__ == "__main__":
    main()
