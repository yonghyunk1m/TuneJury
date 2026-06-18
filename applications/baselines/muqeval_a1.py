"""Head-to-head baseline: MuQ-Eval-A1 vs TuneJury on CMI-RewardBench 4 splits.

Paper anchor: Section 4.2, Table head_to_head (row MuQ-Eval-A1).

MuQ-Eval (Zhu & Li, arXiv:2603.22677, 2026-03-24) is a concurrent
frozen-MuQ + MLP regressor trained on MusicEval via MSE on per-clip MOS
ratings. We benchmark its public A1 checkpoint (HuggingFace
zhudi2825/MuQ-Eval-A1, MIT) on the four CMI-RewardBench test splits
(PAM, MusicEval, CMI-Pref, Music Arena) using the official all_test.jsonl.

Data layout
-----------
Requires CMI-RewardBench bench root (containing all_test.jsonl and the
four audio source subdirs). Get it via the CMI-RewardBench release.

External code dependency
------------------------
MuQ-Eval is not on PyPI; the script clones the upstream repo at runtime
into --muqeval_repo (default /tmp/MuQ-Eval) for the model class. The MuQ
audio encoder is installed via `pip install muq` (one-time).

Usage
-----
Paper-exact reproduction (default; reads the committed
results/muqeval_a1/summary.json):

    $ python applications/baselines/muqeval_a1.py

Fresh re-inference (downloads weights, scores 2,753 clips on GPU,
overwrites the committed summary):

    $ python applications/baselines/muqeval_a1.py \\
          --reinfer --bench_root /path/to/CMI-RewardBench/data \\
          --device cuda:0

Reproduces paper Table head_to_head MuQ-Eval-A1 row: PAM SRCC 0.4995,
MusicEval SRCC 0.8089 (in-distribution, italicized in paper), CMI-Pref
pairwise accuracy 0.6600, Music Arena pairwise accuracy 0.6761.
"""
from __future__ import annotations

# PyTorch 2.6 default weights_only=True breaks the MuQ-Eval-A1 checkpoint load.
import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

# Deterministic cuDNN for repeatable inference.
_torch.backends.cudnn.deterministic = True
_torch.backends.cudnn.benchmark = False
_torch.manual_seed(42)

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as ss
import torch
import torchaudio

DEFAULT_SUMMARY = Path(__file__).parent / "results" / "muqeval_a1" / "summary.json"
SR = 24000
CLIP_SAMPLES = SR * 10  # 10s clip per MuQ-Eval config


def _ensure_muqeval_repo(repo_dir: Path) -> None:
    """Clone the upstream MuQ-Eval repo if not already present."""
    if (repo_dir / "src" / "model.py").exists():
        return
    print(f"[muqeval] cloning upstream repo to {repo_dir}", flush=True)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "git", "clone", "https://github.com/dgtql/MuQ-Eval.git", str(repo_dir),
    ])


def _load_model(repo_dir: Path, device: str):
    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(repo_dir))
    from src.model import MusicQualityModel  # type: ignore

    config_path = hf_hub_download("zhudi2825/MuQ-Eval-A1", "config.yaml")
    model_path = hf_hub_download("zhudi2825/MuQ-Eval-A1", "model_state_dict.pt")
    base_cfg = OmegaConf.load(repo_dir / "configs" / "base.yaml")
    a1_cfg = OmegaConf.load(config_path)
    if "defaults" in a1_cfg:
        del a1_cfg["defaults"]
    cfg = OmegaConf.merge(base_cfg, a1_cfg)
    model = MusicQualityModel(cfg).to(device)
    state = torch.load(model_path, map_location=device)
    miss = model.load_state_dict(state, strict=False)
    assert not miss.missing_keys and not miss.unexpected_keys, (
        f"state_dict mismatch: missing={len(miss.missing_keys)}, "
        f"unexpected={len(miss.unexpected_keys)}"
    )
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[muqeval] loaded; trainable head+pooling params: {trainable:,}", flush=True)
    return model


def _load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    wav = wav.squeeze(0)
    if wav.shape[0] > CLIP_SAMPLES:
        wav = wav[:CLIP_SAMPLES]
    return wav


@torch.no_grad()
def _score(model, audio_path: Path, device: str) -> dict:
    wav = _load_audio(audio_path).unsqueeze(0).to(device)
    out = model(wav)
    return {
        k: float(v.item()) for k, v in out.items()
        if isinstance(v, torch.Tensor) and v.numel() == 1
    }


def _infer(args) -> dict:
    repo_dir = Path(args.muqeval_repo).expanduser().resolve()
    _ensure_muqeval_repo(repo_dir)
    model = _load_model(repo_dir, args.device)

    bench_root = Path(args.bench_root).expanduser().resolve()
    jsonl = bench_root / "all_test.jsonl"
    if not jsonl.exists():
        sys.exit(f"all_test.jsonl not found at {jsonl}")

    rows = [json.loads(l) for l in open(jsonl)]
    print(f"[muqeval] {len(rows)} test rows", flush=True)

    results, fails = [], 0
    t0 = time.time()
    for i, row in enumerate(rows):
        try:
            a_path = bench_root / row["audio-path"]
            entry = {
                "source": row.get("source", "?"),
                "audio_a": row.get("audio-path", ""),
                "scores_a": _score(model, a_path, args.device),
            }
            audio2 = row.get("audio2") or ""
            if audio2:
                entry["audio_b"] = audio2
                entry["scores_b"] = _score(model, bench_root / audio2, args.device)
                # cmi-arena uses preference-musicality (model_a/model_b);
                # Music Arena uses preference (A/B). Capture both.
                entry["pref_musicality"] = row.get("preference-musicality", "")
                entry["preference"] = row.get("preference", "")
            else:
                entry["musicality_mos"] = row.get("musicality", "")
                entry["alignment_mos"] = row.get("text-music alignment", "")
            results.append(entry)
        except Exception as e:
            fails += 1
            if fails <= 3:
                print(f"[muqeval] FAIL row {i}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            rate = (i + 1) / max(time.time() - t0, 0.01)
            eta = (len(rows) - i - 1) / max(rate, 0.01)
            print(
                f"[muqeval] {i+1}/{len(rows)} ({rate:.1f}/s, ETA {eta/60:.1f} min)",
                flush=True,
            )
    return _aggregate(results, fails, time.time() - t0)


def _aggregate(results: list[dict], fails: int = 0, elapsed_s: float = 0.0) -> dict:
    """Compute the four CMI-RewardBench head-to-head numbers from raw scores."""
    by_source = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)

    summary = {
        "n_rows": len(results),
        "n_fails": fails,
        "elapsed_s": elapsed_s,
        "per_split": {},
    }

    # PAM and MusicEval: musicality SRCC per-clip
    for src_key, label in [("PAM (music)", "PAM"), ("MusicEval", "MusicEval")]:
        rs = by_source.get(src_key, [])
        mi, ta, mos = [], [], []
        for r in rs:
            try:
                m = float(r.get("musicality_mos", ""))
            except (TypeError, ValueError):
                continue
            mi.append(r["scores_a"]["MI"])
            ta.append(r["scores_a"]["TA"])
            mos.append(m)
        if not mi:
            summary["per_split"][label] = {"n": 0}
            continue
        rho_mi, _ = ss.spearmanr(mi, mos)
        rho_ta, _ = ss.spearmanr(ta, mos)
        summary["per_split"][label] = {
            "n": len(mi),
            "srcc_mi_vs_musicality": float(rho_mi),
            "srcc_ta_vs_musicality": float(rho_ta),
        }

    # Pairwise sources
    for src_key, label in [
        ("cmi-arena-annotation", "CMI-Pref"),
        # Music Arena source string is verbose; match by substring
        ("Music Arena", "Music Arena"),
    ]:
        rs = [
            r for k, rs0 in by_source.items() for r in rs0 if src_key in k
        ] if label == "Music Arena" else by_source.get(src_key, [])
        correct = total = 0
        for r in rs:
            winner_raw = r.get("pref_musicality") or r.get("preference") or ""
            if winner_raw in ("model_a", "model_b"):
                winner = winner_raw
            elif winner_raw in ("A", "B"):
                winner = "model_a" if winner_raw == "A" else "model_b"
            else:
                continue
            mi_a = r["scores_a"]["MI"]
            mi_b = r["scores_b"]["MI"]
            pred = "model_a" if mi_a > mi_b else "model_b"
            if pred == winner:
                correct += 1
            total += 1
        summary["per_split"][label] = {
            "n": total,
            "pairwise_accuracy_mi": float(correct / total) if total else None,
        }

    return {"summary": summary, "rows": results}


def _print(d: dict) -> None:
    s = d.get("summary", d)
    print(f"\n=== MuQ-Eval-A1 on CMI-RewardBench 4 splits ===")
    print(f"  rows: {s.get('n_rows', '?')}  fails: {s.get('n_fails', '?')}")
    for label, st in s.get("per_split", {}).items():
        if "srcc_mi_vs_musicality" in st:
            print(
                f"  {label:<12} n={st['n']:<4} "
                f"SRCC(MI)={st['srcc_mi_vs_musicality']:+.4f} "
                f"SRCC(TA)={st['srcc_ta_vs_musicality']:+.4f}"
            )
        else:
            acc = st.get("pairwise_accuracy_mi")
            acc_s = f"{acc:.4f}" if acc is not None else "n/a"
            print(f"  {label:<12} n={st['n']:<4} pairwise(MI)={acc_s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench_root", default=None,
                    help="CMI-RewardBench dataset root (required with --reinfer)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--muqeval_repo", default="/tmp/MuQ-Eval",
                    help="Where to clone the upstream MuQ-Eval repo (default /tmp/MuQ-Eval)")
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--reinfer", action="store_true",
                    help="Re-run audio inference. Default: read committed summary.json (paper-exact).")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not args.reinfer:
        if not summary_path.exists():
            sys.exit(
                f"Committed {summary_path} not found. Use --reinfer with "
                "--bench_root to regenerate from raw audio."
            )
        d = json.loads(summary_path.read_text())
        print(f"[muqeval] loaded committed summary from {summary_path}")
        _print(d)
        return

    if not args.bench_root:
        sys.exit("--bench_root required with --reinfer")
    d = _infer(args)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(d))
    print(f"\nwrote {summary_path}")
    _print(d)


if __name__ == "__main__":
    main()
