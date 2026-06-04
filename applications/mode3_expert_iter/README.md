# Mode 3: Expert-iteration post-training

Reproduces Section 5.3 of the paper.

We post-train a ~120M Jamendo-trained rectified-flow DiT with TuneJury-driven expert iteration. The offline loop is:

1. Generate 900 candidates with the current checkpoint.
2. Score every candidate with frozen TuneJury.
3. Filter the top reward decile (90 expert samples).
4. Fine-tune on those 90 expert samples alone for 5K iterations (AdamW, LR `1e-5`, batch 16; EMA snapshot at iter 5K used for inference). Data-free fine-tune: the only training data at this step are the 90 model-generated expert samples (no MTG-Jamendo audio is mixed in). The pretrained FluxAudio-S starting weights were trained on MTG-Jamendo, so the model still carries that distributional prior into the loop. The fine-tune LR (1e-5) is one order of magnitude below the pretraining LR (1e-4); the only new signal beyond the pretrained weights is the reward filter itself.

We choose expert iteration over diffusion-style policy-gradient methods (DDPO, GRPO) because the offline generate, score, filter, and finetune loop avoids the per-step log-probability tracking through the multi-step denoising chain that online policy gradient requires.

## Headline result

On the SDD-100 prompt set, with the FluxAudio-S inference settings (prompt prepended by `"high quality instrumental music, "`, CFG 4.5, 25 Euler steps, no post-processing), the paper reports a three-LR sweep mapping the reward-fidelity Pareto frontier (Table 5 bottom panel):

| Checkpoint | Reward | MAD | CLAP score | Win |
|---|---|---|---|---|
| FluxAudio-S baseline | $-0.262$ | $1.758$ | $0.0921$ | -- |
| lr $10^{-6}$ (conservative) | $-0.096$ ($+0.166$) | $2.051$ ($+0.293$) | $0.1109$ ($+0.019$) | 67/100 |
| lr $5{\times}10^{-6}$ | $+0.107$ ($+0.369$) | $2.041$ ($+0.284$) | $0.1195$ ($+0.027$) | 73/100 |
| lr $10^{-5}$ (aggressive) | $+0.154$ ($+0.416$) | $2.427$ ($+0.669$) | $0.1155$ ($+0.023$) | 75/100 |

Parenthesized values are the change from the baseline, computed before rounding (matching paper Table 5).

MAD here is $-\ln(\text{MAUVE})$ on MERT-v1-330M embeddings against SDD-706 (lower means closer to the reference, aligning with FAD's direction), so positive $\Delta$MAD indicates drift away from SDD-706, paired with the reward lift along the sweep. Headline aggressive-LR result: mean reward gain ${+}0.416$ (75/100 improved), MAD drift ${+}0.669$, CLAP score approximately flat at a small positive offset (${+}0.023$). The divergence between reward and distributional fidelity (MAD) along the LR sweep is the classic reward-exploitation signature (Gao 2023). CLAP scores use the unprefixed SDD captions, matching the Mode 2 `compute_clap_delta.py` convention. Among the swept rates, $5{\times}10^{-6}$ is the most favorable trade-off: more than 2× reward lift over $10^{-6}$ at essentially the same MAD cost. We do not claim this is the global Pareto optimum, only the best of the three swept points. Practitioners using Mode 3 should layer a distributional-fidelity or alignment side metric on top of the reward signal.

## Running

```bash
python run.py \
    --candidates_dir /path/to/generated_900_candidates \
    --checkpoint ../../checkpoints/tunejury.pt \
    --out_dir results/round1
```

`run.py` orchestrates the four-step loop. Steps 2 (score) and 3 (top-decile filter) are backbone-agnostic and run as-is from this script. Steps 1 (generate) and 4 (fine-tune) live in the backbone codebase. The paper's Mode 3 result uses the MeanAudio rectified-flow DiT framework (`https://github.com/xiquan-li/MeanAudio`). For a different backbone, run the equivalent generate / fine-tune entry points in your own codebase. `run.py` prints the exact MeanAudio commands at the end of each invocation for convenience.

## Hyperparameters

| Hyperparameter        | Value     | Notes                                    |
|-----------------------|-----------|------------------------------------------|
| Candidate count       | 900       | 9 per prompt × 100 prompts               |
| Top-decile size       | 90        | Top 10% of 900 candidates                |
| Fine-tune set         | 90 only   | Data-free: only the 90 expert samples (no MTG-Jamendo added) |
| Fine-tune iterations  | 5,000     | EMA snapshot at iter 5K used for inference |
| Learning rate         | $10^{-5}$ | 10× below pretraining (`1e-4`) for fine-tune |
| Batch size            | 16        |                                          |
| Sampler               | Euler     | 25 steps, CFG 4.5                        |
| Prompt prefix         | `"high quality instrumental music, "` | Applied to both baseline and post-trained outputs |
| Post-processing       | None      | Raw generator output (no demucs / LUFS)  |

## MAD (MAUVE Audio Divergence)

To add a third distributional view alongside FAD-CLAP and FAD-MERT, compute MAD on MERT-v1-330M embeddings against the SDD-706 reference set, for both the pretrained baseline checkpoint and the expert-iteration post-trained checkpoint:

```bash
python mad_compute.py \
    --baseline_dir /path/to/pretrained_FluxAudio_S/sdd100_outputs \
    --after_ft_dir /path/to/expert_iter_FluxAudio_S/sdd100_outputs \
    --sdd706_dir   /path/to/SDD-706/audio_wav_torch \
    --out_json     mad_mode3_summary.json
```

Requires `pip install mauve-text`. Each `--*_dir` should contain ~100 wav files (one per SDD-100 prompt, generated with CFG 4.5 + 25 Euler steps, no post-processing) consistent with the inference settings used to produce the Table 5 numbers.
