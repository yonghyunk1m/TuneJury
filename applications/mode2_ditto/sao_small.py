"""Mode 2: DITTO-style inference-time reward optimization on SAO-small.

Routes the frozen TuneJury reward back into a rectified-flow sampler
via full sampler backprop. We initialize a random noise tensor, run
the 8-step rectified-flow ODE forward to produce audio, score the
audio with :class:`tunejury.DifferentiableScorer`, and use Adam on
the noise tensor against the negative reward. Base weights stay
frozen.

Hyper-parameters from §5.2 of the paper:
    - 5 outer iterations
    - Adam learning rate 0.05
    - 8 sampler steps (full backprop)

Expected lift: mean reward +0.159 -> +0.404 (+0.245) on
SAO-small with 19/30 prompts improved (paper Section 5.2).

Example
-------
$ conda activate sao
$ python sao_small.py \\
      --prompts ../../eval/prompts/sdd100.json \\
      --checkpoint ../../checkpoints/tunejury.pt \\
      --out_dir results/ditto_sao_small \\
      --limit 30

Dependencies
------------
- ``stable_audio_tools`` (for SAO-small).
- ``laion_clap`` + ``transformers`` (for the differentiable reward).
- The TuneJury head checkpoint at ``--checkpoint``.
"""

from __future__ import annotations

# Determinism env vars MUST precede the torch import (CUBLAS reads its workspace
# config at first allocation). Matches the protocol in applications/probes/
# (Demucs, per-system PAM, SVCC, SingMOS) where we verified bit-identical reruns.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

# Bit-identical determinism for the full DITTO loop (channels + sampler
# + Adam updates + scoring). cuDNN/cuBLAS pinned to deterministic
# kernels and TF32 disabled; Flash/Memory-Efficient SDPA backward
# kernels use atomic reductions and are NOT bit-identical across runs,
# so we force the SDP backend to math-only (slower but deterministic).
# torch.use_deterministic_algorithms enforces deterministic kernels on
# every op that PyTorch tracks; warn_only=True keeps the run alive if
# a non-tracked op slips through, but on this code path we have not
# observed a warning. Combined with manual_seed in _ditto_optimize,
# two reruns produce bit-identical scores on the same GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.use_deterministic_algorithms(True, warn_only=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury import DifferentiableScorer, Scorer

# Paper §5.2 / §5.1: the instrumental prompt prefix applied to all Mode 1
# backbones is also used on the Mode 2 conditioning input (and matched on
# the reward side so the scorer sees the same caption).
PROMPT_PREFIX = "high quality instrumental music, "


def _install_ditto_patch() -> None:
    """Two monkey-patches required for DITTO over stable_audio_tools 0.0.19:

    (1) ``generate_diffusion_cond`` does not accept ``init_noise=``; it
        allocates the initial noise internally with ``torch.randn`` and has
        no hook for a caller-supplied tensor. DITTO requires backprop through
        that tensor, so we wrap the function, pop ``init_noise`` from kwargs,
        and temporarily monkey-patch ``torch.randn`` within the generation
        module to return our tensor for the single initial-noise allocation.

    (2) ``sample_discrete_euler`` (the rectified-flow Euler sampler used by
        ``sample_rf`` → ``generate_diffusion_cond``) is wrapped with
        ``@torch.no_grad()``, which blocks gradient flow through the sampler.
        We unwrap the decorator via ``__wrapped__`` so the underlying loop
        runs with autograd enabled, letting backprop reach the noise tensor.

    Both patches are idempotent and side-effect-free outside their wrapped calls.
    """
    import stable_audio_tools.inference.generation as _gen
    import stable_audio_tools.inference.sampling as _sampling

    if getattr(_gen.generate_diffusion_cond, "_ditto_patched", False):
        return

    # Patch (2): unwrap @torch.no_grad() on sample_discrete_euler so gradients
    # flow through the rectified-flow sampler. Must be applied at module level
    # since generate_diffusion_cond resolves sample_discrete_euler via module
    # attribute lookup at call time.
    _orig_sde = _sampling.sample_discrete_euler
    if hasattr(_orig_sde, "__wrapped__"):
        _sampling.sample_discrete_euler = _orig_sde.__wrapped__

    # Patch (1): init_noise injection via torch.randn substitution.
    _original = _gen.generate_diffusion_cond

    def _patched(*args, **kwargs):
        init_noise = kwargs.pop("init_noise", None)
        if init_noise is None:
            return _original(*args, **kwargs)
        _real_randn = _gen.torch.randn
        _calls = {"n": 0}

        def _fake_randn(*a, **kw):
            if _calls["n"] == 0:
                _calls["n"] += 1
                return init_noise
            return _real_randn(*a, **kw)

        _gen.torch.randn = _fake_randn
        try:
            return _original(*args, **kwargs)
        finally:
            _gen.torch.randn = _real_randn

    _patched._ditto_patched = True  # type: ignore[attr-defined]
    _gen.generate_diffusion_cond = _patched


def _load_sao_small(device: str):
    """Load Stable Audio Open-small via ``stable_audio_tools``."""
    _install_ditto_patch()
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
    model = model.to(device).eval()
    return model, model_config, generate_diffusion_cond


def _ditto_optimize(
    sao_model,
    sao_config,
    sao_generate,
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
    """Run DITTO optimization for a single prompt.

    Returns a 4-tuple ``(final_audio, baseline_audio, baseline_reward,
    final_reward)``.
    """
    # Seeding strategy: seed once at entry so the initial noise tensor
    # is reproducible per prompt. The outer DITTO loop is deterministic
    # given that noise (Adam updates a single parameter, no fresh RNG
    # draws inside the loop), so re-seeding per iteration is not needed.
    # ``cudnn.deterministic=True`` is set globally to keep the
    # rectified-flow forward pass bitwise reproducible across runs.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sample_rate = sao_config["sample_rate"]
    sample_size = int(duration_s * sample_rate)

    # ``generate_diffusion_cond`` consumes the injected ``init_noise`` AFTER
    # the upstream code rescales the per-sample length to the latent grid
    # (``sample_size // model.pretransform.downsampling_ratio``). The
    # diffusion model operates on the LATENT channels (``model.io_channels``,
    # e.g., 64 for SAO-small), not the pretransform's audio channels
    # (``pretransform.io_channels``, e.g., 2 for stereo). Allocate the
    # optimization variable at latent shape so the patched function consumes
    # it without a shape mismatch.
    if sao_model.pretransform is not None:
        latent_channels = sao_model.io_channels
        latent_length = sample_size // sao_model.pretransform.downsampling_ratio
    else:
        latent_channels = sao_model.io_channels
        latent_length = sample_size

    noise = torch.randn(1, latent_channels, latent_length, device=device)
    noise = torch.nn.Parameter(noise.clone(), requires_grad=True)
    optim = torch.optim.Adam([noise], lr=lr)

    prefixed_prompt = PROMPT_PREFIX + prompt
    conditioning = [{"prompt": prefixed_prompt, "seconds_total": duration_s}]

    def _render(latent_noise: torch.Tensor) -> torch.Tensor:
        # Paper §5.2 specifies the DITTO knobs (5 outer iterations,
        # 8-step sampler-noise backprop, Adam lr 0.05) but does not
        # override CFG. We therefore keep the SAO-small inference
        # default (cfg_scale=6 in ``generate_diffusion_cond``); the
        # Mode 1 standard of CFG 4.5 is a Mode 1 backbone setting and
        # does not apply to the differentiable Mode 2 sampler.
        return sao_generate(
            sao_model,
            conditioning=conditioning,
            steps=n_steps,
            init_noise=latent_noise,
            sample_size=sample_size,
            device=device,
        )

    # Baseline reward at N=1.
    with torch.no_grad():
        audio_0 = _render(noise.detach())
        baseline_reward = scorer.score_waveform(
            _to_score_input(audio_0, sample_rate),
            sr=16_000,
            text=prefixed_prompt,
        ).item()

    # DITTO loop.
    final_reward = baseline_reward
    for _ in range(n_iterations):
        optim.zero_grad()
        audio = _render(noise)
        reward = scorer.score_waveform(
            _to_score_input(audio, sample_rate),
            sr=16_000,
            text=prefixed_prompt,
        )
        loss = -reward
        loss.backward()
        optim.step()
        final_reward = reward.detach().item()

    with torch.no_grad():
        final_audio = _render(noise.detach())
    return final_audio.detach().cpu(), audio_0.detach().cpu(), baseline_reward, final_reward


def _to_score_input(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Convert sampler output to a (T,) waveform at 16 kHz."""
    if audio.dim() == 3:
        audio = audio[0]
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    if sample_rate != 16_000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16_000)
    return audio


def _save(audio: torch.Tensor, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.squeeze().cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate=sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="results/ditto_sao_small")
    parser.add_argument("--n_iterations", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration_s", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=30,
                        help="Number of prompts to run (paper uses 30).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = json.loads(Path(args.prompts).read_text())[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Mode 2] Loading SAO-small ...")
    sao_model, sao_config, sao_generate = _load_sao_small(device)
    print("[Mode 2] Loading differentiable TuneJury scorer (DITTO loop objective) ...")
    diff_scorer = DifferentiableScorer(reward_ckpt=args.checkpoint, device=device)
    # The Frozen Scorer is the paper's reported scoring path (CLAP numpy branch,
    # MERT-processor branch). The DifferentiableScorer uses a tensor-path CLAP
    # branch (gradient-preserving) which introduces a small but systematic
    # offset (~0.5 reward units) vs the frozen path. We optimize the loop with
    # the differentiable path (gradient flows through CLAP+MERT to noise) and
    # REPORT with the frozen path so the numbers are consistent with paper
    # §4 / §A internal-test reward distributions.
    print("[Mode 2] Loading frozen TuneJury scorer (paper reporting) ...")
    frozen_scorer = Scorer.from_pretrained(args.checkpoint, device=device)

    summary = []
    for entry in prompts:
        prompt = entry["prompt"]
        pid = entry.get("prompt_id", entry.get("id", "p"))
        prefixed_prompt = PROMPT_PREFIX + prompt
        audio, audio_baseline, base_diff, final_diff = _ditto_optimize(
            sao_model, sao_config, sao_generate, diff_scorer, prompt,
            n_iterations=args.n_iterations,
            lr=args.lr,
            n_steps=args.n_steps,
            seed=args.seed,
            duration_s=args.duration_s,
            device=device,
        )
        baseline_path = out_dir / f"{pid}_baseline.wav"
        ditto_path = out_dir / f"{pid}_ditto.wav"
        _save(audio_baseline, sao_config["sample_rate"], baseline_path)
        _save(audio, sao_config["sample_rate"], ditto_path)
        # Re-score the saved audio with the frozen Scorer for paper-consistent
        # baseline / DITTO rewards.
        base_frozen = float(frozen_scorer.score(str(baseline_path), prompt=prefixed_prompt))
        final_frozen = float(frozen_scorer.score(str(ditto_path), prompt=prefixed_prompt))
        improved = int(final_frozen > base_frozen)
        summary.append({
            "prompt_id": pid,
            "baseline_reward": base_frozen,
            "ditto_reward": final_frozen,
            "baseline_reward_diff": base_diff,
            "ditto_reward_diff": final_diff,
            "improved": improved,
        })
        print(
            f"[Mode 2] {pid}: frozen {base_frozen:+.3f} -> {final_frozen:+.3f} "
            f"(diff {base_diff:+.3f} -> {final_diff:+.3f}, "
            f"{'improved' if improved else 'no change'})"
        )

    base = sum(s["baseline_reward"] for s in summary) / len(summary)
    final = sum(s["ditto_reward"] for s in summary) / len(summary)
    improved = sum(s["improved"] for s in summary)
    print("=" * 60)
    print(f"[Mode 2] Frozen-scorer mean baseline reward: {base:+.4f}")
    print(f"[Mode 2] Frozen-scorer mean DITTO    reward: {final:+.4f}")
    print(f"[Mode 2] Mean lift: {final-base:+.4f}")
    print(f"[Mode 2] Improved (frozen-scorer): {improved}/{len(summary)} prompts")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
