"""Generate N candidates per prompt with ACE-Step v1.5 Turbo Continuous.

Pure generation; no scoring. Outputs `<out_dir>/{prompt_id}_c{n}.wav` for
n=0..N-1. Score and select with `score_and_select.py` in the
tune-jury env afterwards.

Usage:
    conda activate ace_step
    CUDA_VISIBLE_DEVICES=3 python generate_acestep.py \
        --prompts /tmp/sdd5_prompts.json \
        --out_dir /tmp/acestep_bon \
        --n_candidates 32 \
        --duration 10
"""

import argparse
import json
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_candidates", type=int, default=32)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--infer_step", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 74)))
    p.add_argument("--prompt_prefix", type=str, default="high quality instrumental music, ")
    return p.parse_args()


def build_pipeline():
    from huggingface_hub import snapshot_download
    from acestep.pipeline_ace_step import ACEStepPipeline

    ckpt_dir = Path.home() / ".cache" / "ace-step" / "v15-turbo-continuous"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="ACE-Step/acestep-v15-turbo-continuous",
        local_dir=str(ckpt_dir),
    )
    return ACEStepPipeline(
        checkpoint_dir=str(ckpt_dir),
        device_id=0,
        dtype="bfloat16",
    )


def main():
    import torch
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prompts = json.loads(Path(args.prompts).read_text())
    seeds = args.seeds[: args.n_candidates]
    if len(seeds) < args.n_candidates:
        raise ValueError(f"Need {args.n_candidates} seeds; got {len(seeds)}")

    pipe = build_pipeline()
    print(f"[gen] pipeline ready; {len(prompts)} prompts x {args.n_candidates} candidates")

    t_start = time.time()
    n_done = 0
    n_total = len(prompts) * args.n_candidates

    for entry in prompts:
        pid = entry["prompt_id"]
        prompt = args.prompt_prefix + entry["prompt"]
        for c, seed in enumerate(seeds):
            target = out / f"{pid}_c{c}.wav"
            if target.exists():
                n_done += 1
                continue
            t0 = time.time()
            pipe(
                format="wav",
                audio_duration=args.duration,
                prompt=prompt,
                lyrics="",
                infer_step=args.infer_step,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=[seed],
                save_path=str(target),
                batch_size=1,
            )
            n_done += 1
            elapsed = time.time() - t0
            print(f"[gen] {n_done}/{n_total}  pid={pid}  c={c}  seed={seed}  "
                  f"({elapsed:.1f}s)  -> {target.name}")

    total_min = (time.time() - t_start) / 60
    print(f"[gen] DONE  total={total_min:.1f}min")


if __name__ == "__main__":
    main()
