"""Generate N candidates per prompt with Stable Audio Open (stable_audio_tools).

Usage:
    conda activate sao
    CUDA_VISIBLE_DEVICES=5 python generate_sao.py \
        --prompts ../../eval/prompts/sdd100.json \
        --out_dir results/sao_bon100 \
        --n_candidates 32 \
        --duration 10
"""
import argparse
import json
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="stabilityai/stable-audio-open-1.0")
    p.add_argument("--n_candidates", type=int, default=32)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=7.0)
    p.add_argument("--sampler_type", type=str, default="dpmpp-3m-sde",
                   help="dpmpp-3m-sde for SAO-1.0; euler/dpmpp/rk4/pingpong for SAO-small (rectified flow)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 74)))
    p.add_argument("--prompt_prefix", type=str, default="high quality instrumental music, ")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import soundfile as sf
    from einops import rearrange
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    print(f"[gen] loading {args.model} ...", flush=True)
    model, model_config = get_pretrained_model(args.model)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    device = "cuda"
    model = model.to(device)

    prompts = json.loads(Path(args.prompts).read_text())
    seeds = args.seeds[: args.n_candidates]
    if len(seeds) < args.n_candidates:
        raise ValueError(f"Need {args.n_candidates} seeds; got {len(seeds)}")

    n_total = len(prompts) * args.n_candidates
    n_done = 0
    t_start = time.time()

    for entry in prompts:
        pid = entry["prompt_id"]
        full_prompt = args.prompt_prefix + entry["prompt"]
        for c, seed in enumerate(seeds):
            target = out / f"{pid}_c{c}.wav"
            if target.exists():
                n_done += 1
                continue

            conditioning = [{"prompt": full_prompt, "seconds_start": 0, "seconds_total": args.duration}]
            t0 = time.time()
            output = generate_diffusion_cond(
                model,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type=args.sampler_type,
                device=device,
                seed=seed,
            )
            elapsed = time.time() - t0

            # output: (1, channels, samples)
            output = rearrange(output, "b d n -> d (b n)")
            output = output.to(torch.float32).clamp(-1, 1).cpu()
            # mono mix
            if output.shape[0] > 1:
                output = output.mean(dim=0, keepdim=True)
            tgt_n = int(args.duration * sample_rate)
            if output.shape[-1] > tgt_n:
                output = output[..., :tgt_n]
            sf.write(str(target), output.squeeze(0).numpy(), samplerate=sample_rate)

            n_done += 1
            print(f"[gen] {n_done}/{n_total}  pid={pid}  c={c}  seed={seed}  ({elapsed:.1f}s)", flush=True)

    total_min = (time.time() - t_start) / 60
    print(f"[gen] DONE  total={total_min:.1f}min")


if __name__ == "__main__":
    main()
