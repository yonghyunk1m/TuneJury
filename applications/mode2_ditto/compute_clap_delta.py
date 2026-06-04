"""Mode 2 Delta-CLAP: compute LAION-CLAP cosine change between
baseline and DITTO-optimized audio for SAO-small and TangoFlux.

Backbones save audio as `{prompt_id}_baseline.wav` and
`{prompt_id}_ditto.wav` in their per-backbone result dirs (default
naming used by the `tangoflux_ditto.py` / `sao_small.py` generators
in this directory). The accompanying `summary.json` provides
per-prompt reward stats; only `tangoflux_ditto.py` writes the prompt
text inline. For `sao_small.py` the prompt is recovered from the
original prompt JSON (`--prompts_json`), matched by `prompt_id` (or
by index as a fallback).

Paper Mode 2 table (5.2) ΔCLAP headline:

    SAO-small  (n=30)   ΔCLAP -0.008
    TangoFlux  (n=100)  ΔCLAP +0.043

TangoFlux lifts CLAP alignment as a byproduct of reward optimization,
while SAO-small slightly degrades — the classic reward-exploitation
signature surfacing on the backbone with the lower baseline reward
headroom (paper §5.2).

Usage
-----
$ python compute_clap_delta.py \\
      --results_dir  <path with {pid}_baseline.wav and {pid}_ditto.wav> \\
      --json         <path/summary.json or ditto_results.json> \\
      --prompts_json <eval/prompts/sdd100.json>  # required if summary lacks 'prompt'
      --clap_ckpt    <music_audioset_epoch_15_esc_90.14.pt> \\
      [--limit 30]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch

# Older LAION-CLAP checkpoints predate PyTorch 2.6 weights_only=True default.
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

import laion_clap  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=Path, required=True,
                   help="Dir with {pid}_baseline.wav and {pid}_ditto.wav files.")
    p.add_argument("--json", type=Path, required=True,
                   help="DITTO results JSON (per-prompt records with 'id'/'prompt_id'; "
                        "'prompt' field is optional, recovered from --prompts_json if absent).")
    p.add_argument("--prompts_json", type=Path, default=None,
                   help="Original prompts JSON (e.g. eval/prompts/sdd100.json). Used to recover "
                        "the prompt text when the summary records don't carry one "
                        "(sao_small.py case). Matched by prompt_id, "
                        "with positional index as a fallback.")
    p.add_argument("--clap_ckpt", type=Path, required=True,
                   help="LAION-CLAP music checkpoint (music_audioset_epoch_15_esc_90.14.pt).")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_clap(ckpt):
    m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    m.load_ckpt(str(ckpt))
    m.to(DEVICE).eval()
    return m


@torch.no_grad()
def clap_cos(model, audio_path: Path, prompt: str) -> float:
    a = model.get_audio_embedding_from_filelist([str(audio_path)], use_tensor=True)
    t = model.get_text_embedding([prompt], use_tensor=True)
    a = a / a.norm(dim=-1, keepdim=True)
    t = t / t.norm(dim=-1, keepdim=True)
    return float((a * t).sum().item())


def _build_prompt_lookup(prompts_json: Path | None):
    """Return (by_id, ordered_list) lookups from the original prompts JSON.

    `sao_small.py` writes summary records that carry only `prompt_id` /
    reward stats, no prompt text. We recover the prompt by matching on
    the id field; if the summary record lacks an id we fall back to
    positional indexing.
    """
    if prompts_json is None:
        return {}, []
    raw = json.loads(prompts_json.read_text())
    by_id, ordered = {}, []
    for entry in raw:
        if isinstance(entry, str):
            ordered.append(entry)
            continue
        prompt = entry.get("prompt") or entry.get("caption")
        ordered.append(prompt)
        pid = entry.get("prompt_id", entry.get("id"))
        if pid is not None:
            by_id[str(pid)] = prompt
    return by_id, ordered


def main():
    args = parse_args()
    results = json.loads(args.json.read_text())
    # Some DITTO scripts wrap per-prompt records under "per_prompt" or "results";
    # some return a list directly.
    if isinstance(results, dict):
        for key in ("per_prompt", "results"):
            if key in results:
                records = results[key]
                break
        else:
            records = results
    else:
        records = results
    if args.limit:
        records = records[: args.limit]

    prompts_by_id, prompts_ordered = _build_prompt_lookup(args.prompts_json)

    clap = load_clap(args.clap_ckpt)
    base_scores, ditto_scores = [], []
    for idx, r in enumerate(records):
        pid = r.get("id", r.get("prompt_id"))
        prompt = r.get("prompt")
        if prompt is None:
            prompt = prompts_by_id.get(str(pid))
        if prompt is None and idx < len(prompts_ordered):
            prompt = prompts_ordered[idx]
        if prompt is None:
            print(
                f"  skip pid={pid}: no prompt in summary and not recoverable from "
                f"--prompts_json (pass --prompts_json to enable lookup)",
                flush=True,
            )
            continue
        b = args.results_dir / f"{pid}_baseline.wav"
        d = args.results_dir / f"{pid}_ditto.wav"
        if not (b.exists() and d.exists()):
            print(f"  skip pid={pid}: missing audio", flush=True)
            continue
        b_c = clap_cos(clap, b, prompt)
        d_c = clap_cos(clap, d, prompt)
        base_scores.append(b_c)
        ditto_scores.append(d_c)
        print(f"  pid={pid}: base={b_c:.4f} ditto={d_c:.4f} delta={d_c-b_c:+.4f}", flush=True)

    base_scores = np.array(base_scores)
    ditto_scores = np.array(ditto_scores)
    deltas = ditto_scores - base_scores
    print(f"\nN={len(deltas)}")
    print(f"Base mean CLAP:  {base_scores.mean():.4f}")
    print(f"Ditto mean CLAP: {ditto_scores.mean():.4f}")
    print(f"Mean delta CLAP: {deltas.mean():+.4f}")
    print(f"Win (delta>0):   {(deltas > 0).sum()}/{len(deltas)}")


if __name__ == "__main__":
    main()
