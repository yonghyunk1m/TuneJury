"""Derive paper Table 11 (pairwise accuracy on PAM and MusicEval clip pairs).

Computes per-clip TuneJury scores on PAM (500 clips) and MusicEval (413 clips)
from the CMI-RewardBench manifest, then computes pairwise accuracy on every
distinct clip pair by comparing score ordering against musicality MOS ordering.

Companion to eval/cmi_rewardbench.py (which reports SRCC for the same splits).

Usage:
    python eval/derive_table11.py \
        --checkpoint checkpoints/tunejury.pt \
        --bench-root /tmp/bench_fixed \
        --out /tmp/derived_table11_main.json

Output JSON:
    {
      "PAM_pw_acc":      0.7193,   # paper §A Table 11 PAM column
      "MusicEval_pw_acc": 0.7521,  # paper §A Table 11 MusicEval column
      "n_pairs_PAM":     121016,
      "n_pairs_ME":      79754,
      "n_clips_PAM":     500,
      "n_clips_ME":      413
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunejury import Scorer


def _pairwise_accuracy_from_clips(clips: list[dict], scorer: Scorer,
                                  prompt_protocol: str, bench_root: Path) -> tuple[float, int]:
    """Score every clip, form every distinct pair, count correct orderings.

    For pair (a, b), correct order = sign(reward_a - reward_b) == sign(mos_a - mos_b).
    Ties in either reward or MOS are excluded from the count.
    """
    rewards = []
    mos = []
    for clip in clips:
        prompt = clip.get("prompt", "") if prompt_protocol == "with" else ""
        audio = bench_root / clip["audio_path"] if not Path(clip["audio_path"]).is_absolute() \
                else Path(clip["audio_path"])
        rewards.append(scorer.score(str(audio), prompt=prompt))
        mos.append(clip["musicality_mos"])

    n_correct = 0
    n_pairs = 0
    for i, j in combinations(range(len(clips)), 2):
        d_reward = rewards[i] - rewards[j]
        d_mos = mos[i] - mos[j]
        if d_reward == 0 or d_mos == 0:
            continue
        n_pairs += 1
        if (d_reward > 0) == (d_mos > 0):
            n_correct += 1

    return n_correct / n_pairs if n_pairs else 0.0, n_pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bench-root", required=True, type=Path,
                   help="CMI-RewardBench root with manifest.json")
    p.add_argument("--prompt-protocol", choices=("with", "empty"), default="with")
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSON path")
    args = p.parse_args()

    bench_root = args.bench_root
    manifest = json.load((bench_root / "manifest.json").open())

    scorer = Scorer.from_pretrained(args.checkpoint)

    print(f"[Table 11 derive] PAM: scoring {len(manifest['pam'])} clips ...", file=sys.stderr)
    pam_acc, pam_pairs = _pairwise_accuracy_from_clips(
        manifest["pam"], scorer, args.prompt_protocol, bench_root,
    )
    print(f"[Table 11 derive] MusicEval: scoring {len(manifest['musiceval'])} clips ...",
          file=sys.stderr)
    me_acc, me_pairs = _pairwise_accuracy_from_clips(
        manifest["musiceval"], scorer, args.prompt_protocol, bench_root,
    )

    out = {
        "PAM_pw_acc": pam_acc,
        "MusicEval_pw_acc": me_acc,
        "n_pairs_PAM": pam_pairs,
        "n_pairs_ME": me_pairs,
        "n_clips_PAM": len(manifest["pam"]),
        "n_clips_ME": len(manifest["musiceval"]),
        "checkpoint": args.checkpoint,
        "prompt_protocol": args.prompt_protocol,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
