"""Mode 3: Expert-iteration post-training driven by TuneJury.

Reproduces Section 5.3 of the paper. The offline loop is:

    1. Generate N candidates with the current checkpoint.
    2. Score every candidate with frozen TuneJury.
    3. Filter the top reward decile (90 of 900).
    4. Fine-tune on those 90 expert samples alone for 5,000 iterations
       (AdamW, LR 1e-5, batch 16; EMA snapshot at iter 5K used for
       inference). Data-free fine-tune:
       the only training data at this step are the 90 model-generated
       expert samples (no MTG-Jamendo audio is mixed in). The pretrained
       FluxAudio-S starting weights were of course trained on
       MTG-Jamendo, so the model still carries that distributional
       prior into the loop. The fine-tune LR (1e-5) is one order of
       magnitude below the pretraining LR (1e-4), and the only new
       signal beyond the pretrained weights is the reward filter itself.

Steps 2 (score) and 3 (top-decile filter) are backbone-agnostic and
fully implemented in this script. Steps 1 and 4 (generate and
fine-tune the ~120M rectified-flow DiT) require the backbone
codebase. The paper uses the MeanAudio framework
(https://github.com/xiquan-li/MeanAudio); for a different backbone,
plug in your own generation and fine-tuning entry points where
indicated below.

Headline result (paper §5.3, SDD-100 prompts; FluxAudio-S inference
settings: prefix "high quality instrumental music, ", CFG 4.5,
25 Euler steps, no post-processing): mean per-prompt reward gain
+0.416 at the aggressive LR (1e-5). Side-metric drift:
ΔMAD against SDD-706 +0.669 (lower MAD = closer to SDD-706, so
positive ΔMAD is drift away from the reference), while ΔCLAP score stays
approximately flat at a small positive offset (+0.023). The reward gain
paired with MAD drift is the classic reward-exploitation pattern; see
paper §5.3 for the Pareto-frontier framing and mitigation directions. The conservative
LR (1e-6) yields +0.166 with smaller side-metric regressions; 5e-6
is the most favorable trade-off among the three swept rates
(not necessarily the global Pareto optimum; only 3 LR points tested).

Example
-------
$ python run.py \\
      --candidates_dir results/round1/generated \\
      --checkpoint ../../checkpoints/tunejury.pt \\
      --out_dir results/round1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury import Scorer


REPO_ROOT = Path(__file__).resolve().parents[2]


def step_2_score(args: argparse.Namespace, candidate_dir: Path) -> Path:
    """Score every candidate with frozen TuneJury."""
    print(f"[Mode 3] Step 2/4: scoring {candidate_dir} with frozen TuneJury ...")
    scorer = Scorer.from_pretrained(args.checkpoint)
    rewards: dict[str, float] = {}
    wavs = sorted(candidate_dir.glob("*.wav")) + sorted(candidate_dir.glob("*.flac"))
    for wav in wavs:
        rewards[wav.stem] = scorer.score(str(wav), prompt="")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "step2_scores.json"
    scores_path.write_text(json.dumps(rewards, indent=2))
    print(f"[Mode 3] Wrote {len(rewards)} scores to {scores_path}")
    return scores_path


def step_3_filter_top_decile(
    args: argparse.Namespace, scores_path: Path, candidate_dir: Path,
) -> Path:
    """Select the top-K reward fraction (default top decile = 10%)."""
    print(f"[Mode 3] Step 3/4: filtering top {int(args.top_pct)}% by reward ...")
    scores = json.loads(scores_path.read_text())
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n_keep = max(1, int(len(ranked) * args.top_pct / 100))
    top = ranked[:n_keep]
    print(f"[Mode 3] Keeping {n_keep} of {len(ranked)} samples.")

    out_dir = Path(args.out_dir)
    filtered_dir = out_dir / "step3_top_decile"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    # Symlink the top files into the filtered directory.
    for stem, reward in top:
        for ext in (".wav", ".flac"):
            src = candidate_dir / f"{stem}{ext}"
            if src.exists():
                dst = filtered_dir / src.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
                break

    # Write the filtered manifest with rewards.
    (filtered_dir / "manifest.json").write_text(json.dumps(dict(top), indent=2))
    print(f"[Mode 3] Top-decile samples in {filtered_dir}/")
    return filtered_dir


def _instructions_steps_1_and_4(args: argparse.Namespace) -> None:
    """Print the generate / fine-tune entry points users must adapt from MeanAudio.

    The paper's Mode 3 result uses the MeanAudio rectified-flow DiT
    (https://github.com/xiquan-li/MeanAudio). MeanAudio's training entry
    points are bash wrappers around its own training stack (see
    scripts/flowmatching/train_flowmatching.sh in that repo). tune-jury
    owns only the score + filter steps (this script). For Step 4 the user
    must adapt MeanAudio's training script with the paper §5.3 hyperparams
    listed below — exact reproduction requires writing a small dataset
    adapter that points MeanAudio's training data loader at this script's
    step3_top_decile/ output.
    """
    print()
    print("=" * 60)
    print("[Mode 3] Steps 1 (generate) and 4 (fine-tune) require MeanAudio:")
    print("[Mode 3]   git clone https://github.com/xiquan-li/MeanAudio")
    print()
    print(f"  # Step 1: generate {args.n_candidates} candidates with the current FluxAudio-S")
    print(f"  # checkpoint. Follow MeanAudio's inference scripts (scripts/sample_*) at:")
    print(f"  #   CFG = 4.5, Euler steps = 25, no post-processing,")
    print(f"  #   prompt prefix = 'high quality instrumental music, ',")
    print(f"  #   {args.n_candidates // 100} noise seeds per SDD-100 prompt.")
    print()
    print(f"  # (then) point this script's --candidates_dir at the generation output,")
    print(f"  #        run steps 2-3 here to produce step3_top_decile/, then:")
    print()
    print(f"  # Step 4: fine-tune FluxAudio-S on the top-decile set.")
    print(f"  # Paper §5.3 hyperparams to override in MeanAudio's training script:")
    print(f"  #   optimizer = AdamW, learning rate = {args.lr}")
    print(f"  #   max iterations = {args.iterations}")
    print(f"  #   batch size = {args.batch_size}")
    print(f"  #   EMA decay = 0.999 (snapshot at iter {args.iterations} used for inference)")
    print(f"  #   data source = <out_dir>/step3_top_decile/ (90 expert wavs, no MTG-Jamendo mix-in)")
    print(f"  #")
    print(f"  # Paper §5.3 LR sweep: 1e-6 (conservative, ΔRwd +0.166),")
    print(f"  # 5e-6 (most favorable swept rate, ΔRwd +0.369),")
    print(f"  # 1e-5 (aggressive, ΔRwd +0.416).")
    print(f"  #")
    print(f"  # NOTE: paper §5.3 uses reward-only top-decile filtering (this script).")
    print(f"  # If MeanAudio ships an expert-iteration wrapper that blends reward with")
    print(f"  # CLAP under an alpha sweep, that is a DIFFERENT protocol from the paper")
    print(f"  # — use this script's step3_top_decile/ as the data source instead.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates_dir", required=True,
        help="Directory of generated audio (.wav or .flac) from step 1."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="TuneJury reward checkpoint."
    )
    parser.add_argument("--out_dir", default="results/mode3_expert_iter")
    parser.add_argument(
        "--n_candidates", type=int, default=900,
        help="Candidates generated upstream (paper uses 900).",
    )
    parser.add_argument(
        "--top_pct", type=float, default=10.0,
        help="Top-K percent to keep (paper uses top decile = 10).",
    )
    parser.add_argument(
        "--iterations", type=int, default=5000,
        help="Fine-tuning iterations (paper uses 5K; EMA snapshot at iter 5K used for inference).",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Aggressive endpoint of the paper's LR sweep (1e-6 / 5e-6 / 1e-5).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (paper §A.H: 16).")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (override if needed).")
    args = parser.parse_args()

    candidate_dir = Path(args.candidates_dir)
    if not candidate_dir.exists():
        raise FileNotFoundError(
            f"{candidate_dir} does not exist. Generate candidates first "
            "(step 1 of the loop; see the MeanAudio instructions below)."
        )

    scores_path = step_2_score(args, candidate_dir)
    filtered_dir = step_3_filter_top_decile(args, scores_path, candidate_dir)
    print(f"[Mode 3] Top-decile dataset ready at {filtered_dir}.")
    _instructions_steps_1_and_4(args)


if __name__ == "__main__":
    main()
