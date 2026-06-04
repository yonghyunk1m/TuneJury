"""Generate N candidates per prompt with AudioLDM2-music (diffusers).

Usage:
    conda activate meanaudio
    CUDA_VISIBLE_DEVICES=4 python generate_audioldm2.py \
        --prompts ../../eval/prompts/sdd100.json \
        --out_dir results/audioldm2_music_bon100 \
        --n_candidates 32 \
        --duration 10
"""
import argparse
import json
import time
from pathlib import Path

# GPT2 monkey-patch for transformers compatibility with diffusers AudioLDM2 (newer
# transformers versions removed _get_initial_cache_position / _update_model_kwargs_for_generation
# from GPT2Model, but diffusers 0.30 AudioLDM2 still calls them).
import torch
from transformers import GPT2Model
def _gpt2_init_cache(self, inputs_embeds, model_kwargs):
    if 'cache_position' not in model_kwargs:
        model_kwargs['cache_position'] = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    return model_kwargs
def _gpt2_update_kwargs(self, outputs, model_kwargs):
    if 'cache_position' in model_kwargs and model_kwargs['cache_position'] is not None:
        model_kwargs['cache_position'] = model_kwargs['cache_position'][-1:] + 1
    return model_kwargs
GPT2Model._get_initial_cache_position = _gpt2_init_cache
GPT2Model._update_model_kwargs_for_generation = _gpt2_update_kwargs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="cvssp/audioldm2-music")
    p.add_argument("--n_candidates", type=int, default=32)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--num_inference_steps", type=int, default=200)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 74)))
    p.add_argument("--prompt_prefix", type=str, default="high quality instrumental music, ")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import scipy.io.wavfile as wavfile
    import numpy as np
    from diffusers import AudioLDM2Pipeline

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pipe = AudioLDM2Pipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

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

            generator = torch.Generator(device="cuda").manual_seed(seed)
            t0 = time.time()
            audio = pipe(
                full_prompt,
                num_inference_steps=args.num_inference_steps,
                audio_length_in_s=args.duration,
                num_waveforms_per_prompt=1,
                generator=generator,
                guidance_scale=args.guidance_scale,
            ).audios[0]
            elapsed = time.time() - t0

            # AudioLDM2 outputs 16 kHz mono float32
            sr = 16000
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(str(target), sr, audio_int16)

            n_done += 1
            print(f"[gen] {n_done}/{n_total}  pid={pid}  c={c}  seed={seed}  ({elapsed:.1f}s)", flush=True)

    total_min = (time.time() - t_start) / 60
    print(f"[gen] DONE  total={total_min:.1f}min")


if __name__ == "__main__":
    main()
