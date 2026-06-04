"""Mode 2: DITTO-style inference-time reward optimization on TangoFlux.

Mirrors the SAO-small DITTO setup in ``sao_small.py`` but targets the
declare-lab/TangoFlux base model (515M, flow-matching DiT, 44.1kHz, 10s).
Initial noise latents are wrapped as an ``nn.Parameter`` and optimized
against the negative TuneJury reward with Adam.

Hyper-parameters from Section 5.2 of the paper:
    - 5 outer iterations
    - Adam learning rate 0.05
    - 8 sampler steps (full backprop)

Expected lift: mean reward ``-0.978 -> +0.578`` (``+1.557``) on
TangoFlux with ``100/100`` prompts improved (paper Section 5.2).

Example
-------
$ conda activate tangoflux
$ CUDA_VISIBLE_DEVICES=3 python tangoflux_ditto.py \\
      --prompts ../../eval/prompts/sdd100.json \\
      --checkpoint ../../checkpoints/tunejury.pt \\
      --out_dir results/ditto_tangoflux \\
      --limit 100 \\
      --n_iterations 5 \\
      --steps 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury import DifferentiableScorer


def _load_tangoflux(device: str):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from diffusers import AutoencoderOobleck
    from tangoflux.model import TangoFlux

    paths = snapshot_download(repo_id="declare-lab/TangoFlux")
    with open(f"{paths}/config.json") as f:
        config = json.load(f)
    model = TangoFlux(config)
    weights = load_file(f"{paths}/tangoflux.safetensors")
    model.load_state_dict(weights, strict=False)
    vae = AutoencoderOobleck()
    vae.load_state_dict(load_file(f"{paths}/vae.safetensors"))

    dtype = torch.bfloat16
    model = model.to(device=device, dtype=dtype).eval()
    model.text_encoder = model.text_encoder.to("cpu")
    vae = vae.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    return model, vae


def _ditto_optimize(
    model,
    vae,
    scorer: DifferentiableScorer,
    prompt: str,
    *,
    n_iterations: int = 5,
    lr: float = 0.05,
    n_steps: int = 8,
    seed: int = 42,
    duration_s: float = 10.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    from tangoflux.model import retrieve_timesteps

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dtype = torch.bfloat16

    # Mode 1/2/3 share the same instrumental prompt prefix applied to ALL backbones.
    cond_prompt = "high quality instrumental music, " + prompt
    # TangoFlux is CFG-free by design (paper Section A line 569): we only need the
    # conditional half of the encoder output, and pass guidance=None to the DiT below.
    # encode_text_classifier_free returns a (uncond, cond) pair; we slice the cond row.
    enc_h_pair, enc_mask_pair = model.encode_text_classifier_free([cond_prompt], num_samples_per_prompt=1)
    enc_h = enc_h_pair[1:2].to(device=device, dtype=dtype)
    enc_mask = enc_mask_pair[1:2].to(device)
    masked = torch.where(enc_mask.unsqueeze(-1).expand_as(enc_h), enc_h,
                         torch.tensor(float("nan"), device=device, dtype=dtype))
    pooled = torch.nanmean(masked, dim=1)
    pooled_proj = model.fc(pooled)
    dur = torch.tensor([duration_s], device=device)
    dur_h = model.encode_duration(dur).to(dtype=dtype)
    enc_h_full = torch.cat([enc_h, dur_h], dim=1)
    bsz = 1
    txt_ids = torch.zeros(bsz, enc_h_full.shape[1], 3, device=device, dtype=dtype)
    audio_ids = (torch.arange(model.audio_seq_len, device=device)
                 .unsqueeze(0).unsqueeze(-1).repeat(bsz, 1, 3).to(dtype=dtype))

    scheduler = model.noise_scheduler
    sigmas = np.linspace(1.0, 1 / n_steps, n_steps)

    noise = torch.randn(1, model.audio_seq_len, 64, device=device, dtype=torch.float32) * 1.0
    noise = torch.nn.Parameter(noise.clone(), requires_grad=True)
    optim = torch.optim.Adam([noise], lr=lr)

    base_reward = None
    final_reward = None
    wav_baseline_out = None

    for it in range(n_iterations + 1):
        timesteps, _ = retrieve_timesteps(scheduler, n_steps, device, None, sigmas)
        latents = noise.to(dtype=dtype)
        for t in timesteps:
            noise_pred = model.transformer(
                hidden_states=latents,
                timestep=torch.tensor([t / 1000], device=device, dtype=dtype),
                guidance=None,
                pooled_projections=pooled_proj,
                encoder_hidden_states=enc_h_full,
                txt_ids=txt_ids,
                img_ids=audio_ids,
                return_dict=False,
            )[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        wav_bf16 = vae.decode(latents.transpose(1, 2)).sample
        wav = wav_bf16.float()
        sr_vae = int(vae.config.sampling_rate)
        target_len = int(duration_s * sr_vae)
        wav = wav[..., :target_len]
        wav_mono = wav.mean(dim=1, keepdim=False)
        wav_16k = torchaudio.functional.resample(wav_mono, sr_vae, 16000)
        reward = scorer.score_waveform(wav_16k, sr=16000, text=cond_prompt)

        if it == 0:
            base_reward = reward.detach().item()
            # Snapshot the baseline waveform (pre-optimization) so downstream
            # compute_clap_delta.py / mad_compute.py can compare against the
            # DITTO-optimized output as per the paper's Mode 2 protocol.
            with torch.no_grad():
                wav_baseline_out = wav.detach().float().cpu()[0].clone()
        final_reward = reward.detach().item()

        if it < n_iterations:
            optim.zero_grad()
            (-reward).backward()
            optim.step()

    with torch.no_grad():
        wav_out = wav.detach().float().cpu()[0].clone()
    # Drop everything that might keep autograd graphs / GPU memory alive
    del noise, optim, enc_h, enc_mask, enc_h_full, pooled, pooled_proj, txt_ids, audio_ids
    del wav, wav_bf16, latents, noise_pred, reward
    return wav_out, wav_baseline_out, base_reward, final_reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_iterations", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--seed_base", type=int, default=42)
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TangoFlux...", flush=True)
    model, vae = _load_tangoflux(device)
    print("Loaded.", flush=True)

    print("Loading TuneJury scorer...", flush=True)
    scorer = DifferentiableScorer(reward_ckpt=args.checkpoint, device=device)
    print("Scorer ready.", flush=True)

    with open(args.prompts) as f:
        prompts = json.load(f)
    if args.limit > 0:
        prompts = prompts[: args.limit]

    import gc
    results = []
    t0 = time.time()
    for i, item in enumerate(prompts):
        prompt = item if isinstance(item, str) else (item.get("prompt") or item.get("caption"))
        if isinstance(item, dict):
            pid = item.get("prompt_id", item.get("id", i))
        else:
            pid = i
        seed_idx = i  # always-integer index for deterministic per-prompt seeding
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        try:
            wav, wav_base, base_r, final_r = _ditto_optimize(
                model, vae, scorer, prompt,
                n_iterations=args.n_iterations,
                lr=args.lr,
                n_steps=args.steps,
                seed=args.seed_base + 100 * seed_idx,
                duration_s=args.duration,
                device=device,
            )
            sr_vae = int(vae.config.sampling_rate)
            # Match the {pid}_baseline.wav / {pid}_ditto.wav naming expected by
            # compute_clap_delta.py and the Mode 2 MAD recompute pipeline.
            torchaudio.save(str(out_dir / f"{pid}_baseline.wav"), wav_base, sr_vae)
            torchaudio.save(str(out_dir / f"{pid}_ditto.wav"), wav, sr_vae)
            results.append({
                "id": pid, "prompt_id": pid, "prompt": prompt,
                "base_reward": base_r, "after_reward": final_r,
                "delta": final_r - base_r, "improved": final_r > base_r,
            })
            del wav, wav_base
            print(f"[{i+1}/{len(prompts)}] base={base_r:+.3f} after={final_r:+.3f} d={final_r-base_r:+.3f}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[{i+1}/{len(prompts)}] OOM: {e}", flush=True)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            continue
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "n": len(results),
            "mean_base": sum(r["base_reward"] for r in results) / max(len(results), 1),
            "mean_after": sum(r["after_reward"] for r in results) / max(len(results), 1),
            "mean_delta": sum(r["delta"] for r in results) / max(len(results), 1),
            "win_rate": sum(1 for r in results if r["improved"]) / max(len(results), 1),
            "results": results,
            "config": vars(args),
        }, f, indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m -> {out_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
