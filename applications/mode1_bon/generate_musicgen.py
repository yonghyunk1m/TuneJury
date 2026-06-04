"""Generate N candidates per prompt with MusicGen / MAGNET (audiocraft).

Usage:
    conda activate musicgen
    CUDA_VISIBLE_DEVICES=3 python generate_musicgen.py \
        --prompts ../../eval/prompts/sdd100.json \
        --model facebook/musicgen-medium \
        --out_dir results/musicgen_medium_bon100 \
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
    p.add_argument("--model", required=True,
                   help="HF id, e.g. facebook/musicgen-medium / facebook/musicgen-large / facebook/magnet-medium-30secs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_candidates", type=int, default=32)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 74)))
    p.add_argument("--prompt_prefix", type=str, default="high quality instrumental music, ")
    p.add_argument("--magnet", action="store_true", help="treat as MAGNET (different generator class)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    if args.magnet or "magnet" in args.model.lower():
        from audiocraft.models import MAGNeT
        model = MAGNeT.get_pretrained(args.model)
        # MAGNET-medium-30secs has fixed 30s duration; we still trim/center later by duration kwarg
        model.set_generation_params(use_sampling=True, top_k=250, top_p=0.0, temperature=1.0,
                                    cfg_coef=10.0, max_cfg_coef=10.0, min_cfg_coef=1.0)
    else:
        from audiocraft.models import MusicGen
        model = MusicGen.get_pretrained(args.model)
        model.set_generation_params(duration=args.duration, use_sampling=True, top_k=250,
                                    top_p=0.0, temperature=1.0, cfg_coef=3.0)

    sr = model.sample_rate
    prompts = json.loads(Path(args.prompts).read_text())
    seeds = args.seeds[: args.n_candidates]
    if len(seeds) < args.n_candidates:
        raise ValueError(f"Need {args.n_candidates} seeds; got {len(seeds)}")

    n_total = len(prompts) * args.n_candidates
    n_done = 0
    t_start = time.time()

    import soundfile as sf
    for entry in prompts:
        pid = entry["prompt_id"]
        full_prompt = args.prompt_prefix + entry["prompt"]
        for c, seed in enumerate(seeds):
            target = out / f"{pid}_c{c}.wav"
            if target.exists():
                n_done += 1
                continue

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            t0 = time.time()
            with torch.no_grad():
                wav = model.generate([full_prompt], progress=False)[0]
            elapsed = time.time() - t0

            # wav: (channels, samples)
            wav = wav.detach().cpu()
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            # trim to duration
            tgt_n = int(args.duration * sr)
            if wav.shape[-1] > tgt_n:
                wav = wav[..., :tgt_n]
            sf.write(str(target), wav.squeeze(0).numpy(), samplerate=sr)

            n_done += 1
            print(f"[gen] {n_done}/{n_total}  pid={pid}  c={c}  seed={seed}  ({elapsed:.1f}s)", flush=True)

    total_min = (time.time() - t_start) / 60
    print(f"[gen] DONE  total={total_min:.1f}min")


if __name__ == "__main__":
    main()
