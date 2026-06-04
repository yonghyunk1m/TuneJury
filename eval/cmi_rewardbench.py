"""Reproduce CMI-RewardBench head-to-head (Section 4.2 / Table 4).

Evaluates TuneJury on the four CMI-RewardBench test splits:

    - PAM        (500 clips,   musicality SRCC)
    - MusicEval  (413 samples, musicality SRCC)
    - CMI-Pref   (500 pairs,   pairwise accuracy)
    - Music Arena (1340 pairs, pairwise accuracy)

External data dependency
------------------------
CMI-RewardBench (Ma et al. 2026, arXiv:2603.00610) is NOT shipped with
this repository. Download the upstream release from:

    Paper:  https://arxiv.org/abs/2603.00610
    Code:   https://github.com/Haiwen-Xia/CMI-RewardBench
    Data:   https://huggingface.co/datasets/HaiwenXia/cmi-pref
            https://huggingface.co/datasets/HaiwenXia/cmi-pref-pseudo

See docs/reproducing.md section 5.0 for the full preparation walkthrough.

Manifest format
---------------
``--bench-root`` should point to a directory containing
``manifest.json`` with four keys, each a list:

    {
      "pam":        [{"audio_path": "...", "prompt": "...", "musicality_mos": 4.2}, ...],
      "musiceval":  [{"audio_path": "...", "prompt": "...", "musicality_mos": 3.8}, ...],
      "cmi_pref":   [{"audio_a": "...", "audio_b": "...", "prompt": "...", "winner": "a"}, ...],
      "music_arena":[{"audio_a": "...", "audio_b": "...", "prompt": "...", "winner": "b"}, ...]
    }

``audio_path`` / ``audio_a`` / ``audio_b`` may be absolute or relative
to ``<bench-root>``. ``winner`` is ``"a"`` or ``"b"``. ``musicality_mos``
is a float (1-5 scale in the upstream release).

Convert the upstream CMI-RewardBench release to this layout before
running this script.

Fairness disclosure: the Music Arena split is fair OOD for our
released checkpoint because the 1,340 bench MA battle_uuids were
removed from our entire MA pool (train, validation, held-out test)
before training. Item-level disjointness is verified at file
identifier / prompt / byte-MD5 levels (Section A.D).

The companion ``--prompt-protocol empty`` flag reproduces the
empty-prompt OOD release recommendation from Section 4.2 (text
branch fed an empty string).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunejury import Scorer


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-9))


_CMI_DOWNLOAD_HELP = (
    "CMI-RewardBench is an external dataset and must be downloaded separately.\n"
    "  Paper: https://arxiv.org/abs/2603.00610\n"
    "  Code:  https://github.com/Haiwen-Xia/CMI-RewardBench\n"
    "  Data:  https://huggingface.co/datasets/HaiwenXia/cmi-pref\n"
    "         https://huggingface.co/datasets/HaiwenXia/cmi-pref-pseudo\n"
    "See docs/reproducing.md section 5.0 for the manifest.json layout this "
    "script expects."
)

_REQUIRED_KEYS = ("pam", "musiceval", "cmi_pref", "music_arena")


def _check_bench_root(bench_root: Path) -> dict:
    """Validate --bench-root and return the parsed manifest.

    Exits with a clear, actionable error message if the directory or
    manifest is missing/malformed, including the upstream download URL.
    """
    if not bench_root.exists():
        sys.stderr.write(
            f"ERROR: --bench-root '{bench_root}' does not exist.\n\n"
            + _CMI_DOWNLOAD_HELP
            + "\n"
        )
        sys.exit(2)
    if not bench_root.is_dir():
        sys.stderr.write(
            f"ERROR: --bench-root '{bench_root}' is not a directory.\n\n"
            + _CMI_DOWNLOAD_HELP
            + "\n"
        )
        sys.exit(2)
    manifest_path = bench_root / "manifest.json"
    if not manifest_path.exists():
        sys.stderr.write(
            f"ERROR: --bench-root '{bench_root}' does not contain "
            f"manifest.json.\n\n"
            + _CMI_DOWNLOAD_HELP
            + "\n"
        )
        sys.exit(2)
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        sys.stderr.write(
            f"ERROR: failed to parse '{manifest_path}': {e}\n\n"
            + _CMI_DOWNLOAD_HELP
            + "\n"
        )
        sys.exit(2)
    if not isinstance(manifest, dict):
        sys.stderr.write(
            f"ERROR: '{manifest_path}' must be a JSON object with keys "
            f"{list(_REQUIRED_KEYS)}.\n\n"
            + _CMI_DOWNLOAD_HELP
            + "\n"
        )
        sys.exit(2)
    missing = [k for k in _REQUIRED_KEYS if k not in manifest]
    if missing:
        sys.stderr.write(
            f"ERROR: '{manifest_path}' is missing required keys: {missing}.\n"
            f"Expected all four of {list(_REQUIRED_KEYS)}.\n\n"
            + _CMI_DOWNLOAD_HELP
            + "\n"
        )
        sys.exit(2)
    return manifest


def _resolve(bench_root: Path, audio_path: str) -> str:
    """Resolve a manifest audio_path against bench_root if not absolute."""
    p = Path(audio_path)
    if p.is_absolute():
        return str(p)
    return str(bench_root / p)


def _pairwise_accuracy(
    pairs: list[dict], scorer: Scorer, prompt_protocol: str, bench_root: Path,
) -> float:
    correct = 0
    for pair in pairs:
        prompt = pair.get("prompt", "") if prompt_protocol == "with" else ""
        s_a = scorer.score(_resolve(bench_root, pair["audio_a"]), prompt=prompt)
        s_b = scorer.score(_resolve(bench_root, pair["audio_b"]), prompt=prompt)
        pred = "a" if s_a > s_b else "b"
        if pred == pair["winner"]:
            correct += 1
    return correct / max(len(pairs), 1)


def _musicality_srcc(
    clips: list[dict], scorer: Scorer, prompt_protocol: str, bench_root: Path,
) -> float:
    rewards = []
    mos = []
    for clip in clips:
        prompt = clip.get("prompt", "") if prompt_protocol == "with" else ""
        rewards.append(
            scorer.score(_resolve(bench_root, clip["audio_path"]), prompt=prompt)
        )
        mos.append(clip["musicality_mos"])
    return _spearman(np.array(rewards), np.array(mos))


def evaluate(args: argparse.Namespace) -> dict:
    bench_root = Path(args.bench_root)
    bench = _check_bench_root(bench_root)
    scorer = Scorer.from_pretrained(args.checkpoint)

    results = {}
    results["PAM"] = _musicality_srcc(
        bench["pam"], scorer, args.prompt_protocol, bench_root,
    )
    results["MusicEval"] = _musicality_srcc(
        bench["musiceval"], scorer, args.prompt_protocol, bench_root,
    )
    results["CMI-Pref"] = _pairwise_accuracy(
        bench["cmi_pref"], scorer, args.prompt_protocol, bench_root,
    )
    results["MA bench"] = _pairwise_accuracy(
        bench["music_arena"], scorer, args.prompt_protocol, bench_root,
    )
    # Single-line JSON so downstream parsers (eval/cmi_rewardbench_sweep.py)
    # can `json.loads(out.strip().splitlines()[-1])` without breaking.
    print(json.dumps(results))
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--bench-root", required=True,
        help="Path to the local copy of CMI-RewardBench (with manifest.json)."
    )
    p.add_argument(
        "--prompt-protocol", choices=("with", "empty"), default="with",
        help="With-prompt = released T+A protocol (default); "
             "empty = OOD-prompt-empty recommendation from §4.2."
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
