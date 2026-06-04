"""Decomposition probe (Section 3): can TuneJury's features support a
musicality/alignment decomposition?

Defines two derived scores on top of the released single-scalar TuneJury:

    TuneJury_musicality(a)    := TuneJury(a, empty)              # audio-only score
    TuneJury_alignment(a, t)  := TuneJury(a, t) - TuneJury(a, empty)   # marginal text contribution

Four-stage probe on CMI-RewardBench PAM (n=500) and MusicEval (n=413) per-axis MOS:

    (i)   Post-hoc decomposition: does the audio-only / delta split
          recover musicality vs alignment from the released composite head?
    (ii)  Cross-distribution supervised: train a fresh MLP head on alignment
          MOS from one of {PAM, MusicEval}, test on the other.
    (iii) Stratified combined: 80/20 stratified split over the combined 913
          clips, 20-seed mean + 95% CI for alignment SRCC, partial SRCC
          (controlling for musicality), and residual SRCC (training on
          alignment minus linear musicality).
    (iv)  Data scaling: 5%, 10%, 25%, 50%, 75%, 100% of stratified train pool
          per 5 seeds, partial-SRCC scaling curve.

Reproduces (paper Section 3, Decomposition probe paragraph + Figure):

    Stage (i) [single-seed, our measurements]:
        PAM:       composite musicality SRCC +0.59, alignment SRCC +0.26
                   audio-only musicality   +0.67, alignment       +0.33
                   delta (post-hoc) vs PAM alignment MOS:  SRCC  -0.30
        MusicEval: composite musicality SRCC +0.67, alignment SRCC +0.47
                   audio-only musicality   +0.66, alignment       +0.48
                   delta (post-hoc) vs MusicEval alignment MOS: SRCC +0.02

    Stage (ii) [cross-distribution supervised]:
        PAM -> MusicEval alignment SRCC:  +0.18
        MusicEval -> PAM alignment SRCC:  -0.41

    Stage (iii) [stratified combined, 20-seed]:
        alignment SRCC mean +0.630  std 0.047  95% CI [+0.608, +0.651]
        partial   SRCC mean +0.305  std 0.074  95% CI [+0.271, +0.340]
        residual  SRCC mean +0.444  std 0.058  95% CI [+0.417, +0.472]

    Stage (iv) [scaling, 5-seed]:
        partial SRCC at n=36 (5%)   mean +0.085 (95% CI ±0.084)
        partial SRCC at n=72 (10%)  mean +0.123 (95% CI ±0.079)
        partial SRCC at n=182 (25%) mean +0.185 (95% CI ±0.071)
        partial SRCC at n=364 (50%) mean +0.226 (95% CI ±0.062)
        partial SRCC at n=546 (75%) mean +0.283 (95% CI ±0.063)
        partial SRCC at n=728 (100%) mean +0.318 (95% CI ±0.065)
        (monotone ascent, no plateau)

External data dependency
------------------------
CMI-RewardBench (Ma et al. 2026) PAM + MusicEval per-clip MOS:
    Paper: https://arxiv.org/abs/2603.00610
    Code:  https://github.com/Haiwen-Xia/CMI-RewardBench
Provide --cmi-root pointing at the unpacked release directory whose
``all_test.jsonl`` contains rows tagged ``PAM`` / ``MusicEval`` with
both ``musicality`` and ``text-music alignment`` MOS.

Usage
-----
$ python -m eval.decomposition_probe \\
      --checkpoint     checkpoints/tunejury.pt \\
      --cmi-root       /path/to/CMI-RewardBench/data \\
      --stage          all     # all | post_hoc | crossdist | stratified | scaling
      --n-seeds-stab   20      # stage (iii)
      --n-seeds-scale  5       # stage (iv)
      --output-json    decomposition_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as t_dist

# Make the script runnable both as ``python -m eval.decomposition_probe`` and
# as ``python eval/decomposition_probe.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunejury import Scorer  # noqa: E402


# --------------------------------------------------------------------------
# Data loading + feature extraction
# --------------------------------------------------------------------------

def _load_cmi_pam_me(cmi_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read PAM and MusicEval rows with both musicality + alignment MOS.

    Expects ``cmi_root / all_test.jsonl`` from the CMI-RewardBench release.
    """
    all_test_path = cmi_root / "all_test.jsonl"
    if not all_test_path.exists():
        sys.stderr.write(
            f"ERROR: missing {all_test_path}.\n"
            "Provide --cmi-root pointing at the unpacked CMI-RewardBench data dir.\n"
        )
        sys.exit(2)
    rows = []
    with open(all_test_path) as f:
        for line in f:
            rows.append(json.loads(line))

    def keep(row, tag):
        if tag not in str(row.get("source", "")):
            return False
        if row.get("text-music alignment") in (None, ""):
            return False
        ap = cmi_root / row.get("audio-path", "")
        return ap.exists()

    pam = [r for r in rows if keep(r, "PAM")]
    me = [r for r in rows if keep(r, "MusicEval")]
    return pam, me


def _extract_features(
    scorer: Scorer, clips: List[Dict[str, Any]], cmi_root: Path, tag: str,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Extract 2048-d (CLAP_audio + MERT + CLAP_text) features + MOS arrays."""
    feats, mus, aln = [], [], []
    for i, c in enumerate(clips):
        audio = str(cmi_root / c["audio-path"])
        prompt = (c.get("prompt") or "").strip()
        try:
            with torch.no_grad():
                f = scorer._extract_features(audio, prompt).detach().cpu().squeeze(0)
        except Exception as exc:
            print(f"  [{tag}] skip {i}: {type(exc).__name__}", flush=True)
            torch.cuda.empty_cache()
            continue
        feats.append(f)
        mus.append(
            float(c["musicality"])
            if c.get("musicality") not in (None, "")
            else float("nan")
        )
        aln.append(float(c["text-music alignment"]))
        if (i + 1) % 100 == 0:
            print(f"  [{tag}] {i+1}/{len(clips)} extracted", flush=True)
            torch.cuda.empty_cache()
    return torch.stack(feats), np.array(mus), np.array(aln)


# --------------------------------------------------------------------------
# Stage helpers
# --------------------------------------------------------------------------

class _Head(torch.nn.Module):
    def __init__(self, in_dim: int = 2048, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):  # noqa: D401
        return self.net(x).squeeze(-1)


def _stratified_split(y: np.ndarray, seed: int, frac_train: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """5-bin stratified split (deterministic for given seed)."""
    bins = np.percentile(y, [20, 40, 60, 80])
    bin_idx = np.digitize(y, bins)
    rng = np.random.RandomState(seed)
    tr, vl = [], []
    for b in range(5):
        idxs = np.where(bin_idx == b)[0]
        rng.shuffle(idxs)
        cut = int(frac_train * len(idxs))
        tr.extend(idxs[:cut])
        vl.extend(idxs[cut:])
    return np.array(tr), np.array(vl)


def _partial_spearman(pred: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> float:
    """Spearman partial correlation r_{pred, y | z} via rank residualization."""
    pr = rankdata(pred[mask])
    ar = rankdata(y[mask])
    mr = rankdata(z[mask])

    def resid(yv, xv):
        b = np.cov(yv, xv)[0, 1] / np.var(xv)
        return yv - b * xv

    return float(spearmanr(resid(pr, mr), resid(ar, mr)).correlation)


def _train_head_alignment(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    seed: int,
    n_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[float, np.ndarray]:
    """Train a fresh head; return best val SRCC + best-epoch predictions."""
    torch.manual_seed(seed)
    head = _Head().to(train_X.device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_srcc, best_pred = -float("inf"), None
    val_y_np = val_y.cpu().numpy()
    for ep in range(n_epochs):
        head.train()
        pred = head(train_X)
        loss = torch.nn.functional.mse_loss(pred, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            head.eval()
            with torch.no_grad():
                vp = head(val_X).cpu().numpy()
            srcc = spearmanr(vp, val_y_np).correlation
            if srcc > best_srcc:
                best_srcc, best_pred = float(srcc), vp.copy()
    return best_srcc, best_pred


# --------------------------------------------------------------------------
# Stage implementations
# --------------------------------------------------------------------------

def stage_post_hoc(scorer: Scorer, pam_clips, me_clips, cmi_root, device) -> Dict[str, Any]:
    """Stage (i): post-hoc decomposition via the released composite head."""
    print("\n=== Stage (i): Post-hoc decomposition ===", flush=True)
    out = {}
    for tag, clips in [("PAM", pam_clips), ("MusicEval", me_clips)]:
        composite, audio_only, mus_mos, aln_mos = [], [], [], []
        for i, c in enumerate(clips):
            audio = str(cmi_root / c["audio-path"])
            prompt = (c.get("prompt") or "").strip()
            with torch.no_grad():
                s_full = float(scorer.score(audio, prompt=prompt))
                s_empty = float(scorer.score(audio, prompt=""))
            composite.append(s_full)
            audio_only.append(s_empty)
            mus_mos.append(float(c["musicality"]) if c.get("musicality") not in (None, "") else float("nan"))
            aln_mos.append(float(c["text-music alignment"]))
            if (i + 1) % 100 == 0:
                print(f"  [{tag}] {i+1}/{len(clips)}", flush=True)
        composite = np.array(composite)
        audio_only = np.array(audio_only)
        delta = composite - audio_only
        mus_mos = np.array(mus_mos)
        aln_mos = np.array(aln_mos)
        m_mask = ~np.isnan(mus_mos)
        out[tag] = {
            "n": int(len(clips)),
            "composite_vs_musicality": float(spearmanr(composite[m_mask], mus_mos[m_mask]).correlation),
            "composite_vs_alignment": float(spearmanr(composite, aln_mos).correlation),
            "audio_only_vs_musicality": float(spearmanr(audio_only[m_mask], mus_mos[m_mask]).correlation),
            "audio_only_vs_alignment": float(spearmanr(audio_only, aln_mos).correlation),
            "delta_vs_musicality": float(spearmanr(delta[m_mask], mus_mos[m_mask]).correlation),
            "delta_vs_alignment": float(spearmanr(delta, aln_mos).correlation),
        }
        print(
            f"  {tag} (n={out[tag]['n']}): "
            f"composite mus {out[tag]['composite_vs_musicality']:+.4f} aln {out[tag]['composite_vs_alignment']:+.4f}; "
            f"audio-only mus {out[tag]['audio_only_vs_musicality']:+.4f} aln {out[tag]['audio_only_vs_alignment']:+.4f}; "
            f"delta mus {out[tag]['delta_vs_musicality']:+.4f} aln {out[tag]['delta_vs_alignment']:+.4f}",
            flush=True,
        )
    return out


def stage_crossdist(pam_X, pam_aln, me_X, me_aln, device) -> Dict[str, Any]:
    """Stage (ii): cross-distribution supervised."""
    print("\n=== Stage (ii): Cross-distribution supervised ===", flush=True)
    pam_X_t = pam_X.to(device); pam_y_t = torch.tensor(pam_aln, dtype=torch.float32).to(device)
    me_X_t = me_X.to(device); me_y_t = torch.tensor(me_aln, dtype=torch.float32).to(device)
    p2m, _ = _train_head_alignment(pam_X_t, pam_y_t, me_X_t, me_y_t, seed=42)
    m2p, _ = _train_head_alignment(me_X_t, me_y_t, pam_X_t, pam_y_t, seed=42)
    print(f"  PAM -> MusicEval alignment SRCC: {p2m:+.4f}", flush=True)
    print(f"  MusicEval -> PAM alignment SRCC: {m2p:+.4f}", flush=True)
    return {"PAM_to_MusicEval": p2m, "MusicEval_to_PAM": m2p}


def stage_stratified(all_X, all_mus, all_aln, n_seeds, device) -> Dict[str, Any]:
    """Stage (iii): stratified combined, multi-seed mean + 95% CI."""
    print(f"\n=== Stage (iii): Stratified combined, {n_seeds}-seed ===", flush=True)
    all_aln_t = torch.tensor(all_aln, dtype=torch.float32)
    aln_list, par_list, res_list = [], [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        tr_idx, vl_idx = _stratified_split(all_aln, seed)
        train_X = all_X[tr_idx].to(device)
        val_X = all_X[vl_idx].to(device)
        train_y = all_aln_t[tr_idx].to(device)
        val_y = all_aln_t[vl_idx].to(device)
        val_mus = all_mus[vl_idx]
        v_mask = ~np.isnan(val_mus)

        # alignment-supervised head
        aln_srcc, best_pred = _train_head_alignment(train_X, train_y, val_X, val_y, seed=seed)
        par_srcc = _partial_spearman(best_pred, val_y.cpu().numpy(), val_mus, v_mask)

        # residual-supervised head
        train_mus = all_mus[tr_idx]
        tm_mask = ~np.isnan(train_mus)
        aln_tr = all_aln[tr_idx][tm_mask]
        mus_tr = train_mus[tm_mask]
        beta = np.cov(aln_tr, mus_tr)[0, 1] / np.var(mus_tr)
        alpha = aln_tr.mean() - beta * mus_tr.mean()
        train_resid = aln_tr - (alpha + beta * mus_tr)
        train_X_r = all_X[tr_idx[tm_mask]].to(device)
        train_y_r = torch.tensor(train_resid, dtype=torch.float32).to(device)
        val_resid = all_aln[vl_idx][v_mask] - (alpha + beta * val_mus[v_mask])
        val_y_r = torch.tensor(val_resid, dtype=torch.float32).to(device)
        val_X_r_mask = val_X[torch.tensor(v_mask, device=device)]
        res_srcc, _ = _train_head_alignment(train_X_r, train_y_r, val_X_r_mask, val_y_r, seed=seed + 1000)

        aln_list.append(aln_srcc); par_list.append(par_srcc); res_list.append(res_srcc)
        print(f"  seed {seed:>2}: aln {aln_srcc:+.4f}, partial {par_srcc:+.4f}, residual {res_srcc:+.4f}", flush=True)

    def stats(arr):
        arr = np.array(arr)
        n = len(arr)
        m = float(arr.mean())
        s = float(arr.std(ddof=1)) if n > 1 else 0.0
        se = s / np.sqrt(n) if n > 1 else 0.0
        t_crit = float(t_dist.ppf(0.975, df=max(n - 1, 1)))
        return {
            "values": [float(x) for x in arr], "mean": m, "std": s,
            "ci_low": m - t_crit * se, "ci_high": m + t_crit * se, "n": n,
        }

    return {"alignment": stats(aln_list), "partial": stats(par_list), "residual": stats(res_list)}


def stage_scaling(all_X, all_mus, all_aln, n_seeds, device) -> Dict[str, Any]:
    """Stage (iv): data scaling, 5-seed default."""
    print(f"\n=== Stage (iv): Data scaling, {n_seeds}-seed per size ===", flush=True)
    all_aln_t = torch.tensor(all_aln, dtype=torch.float32)
    fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    out = {}
    for frac in fractions:
        aln_list, par_list = [], []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            tr_idx, vl_idx = _stratified_split(all_aln, seed)
            rng = np.random.RandomState(seed)
            tr_shuffled = tr_idx.copy()
            rng.shuffle(tr_shuffled)
            n = max(int(frac * len(tr_shuffled)), 10)
            sub_idx = tr_shuffled[:n]
            train_X = all_X[sub_idx].to(device)
            train_y = all_aln_t[sub_idx].to(device)
            val_X = all_X[vl_idx].to(device)
            val_y = all_aln_t[vl_idx].to(device)
            val_mus = all_mus[vl_idx]
            v_mask = ~np.isnan(val_mus)
            aln_srcc, best_pred = _train_head_alignment(train_X, train_y, val_X, val_y, seed=seed * 100 + int(frac * 100))
            par_srcc = _partial_spearman(best_pred, val_y.cpu().numpy(), val_mus, v_mask)
            aln_list.append(aln_srcc); par_list.append(par_srcc)
        out[str(frac)] = {"alignment": aln_list, "partial": par_list, "n_train": int(frac * 728)}
        print(
            f"  frac {int(frac*100):>3}% (n={int(frac*728)}): "
            f"alignment {np.mean(aln_list):+.4f}, partial {np.mean(par_list):+.4f}",
            flush=True,
        )
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="TuneJury checkpoint .pt")
    parser.add_argument("--cmi-root", required=True, type=Path, help="CMI-RewardBench data dir (contains all_test.jsonl)")
    parser.add_argument("--stage", default="all",
                        choices=["all", "post_hoc", "crossdist", "stratified", "scaling"])
    parser.add_argument("--n-seeds-stab", type=int, default=20, help="Seeds for stage (iii) stratified")
    parser.add_argument("--n-seeds-scale", type=int, default=5, help="Seeds for stage (iv) scaling")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", default="decomposition_probe.json")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)

    scorer = Scorer.from_pretrained(args.checkpoint, device=args.device)
    pam_clips, me_clips = _load_cmi_pam_me(args.cmi_root)
    print(f"PAM clips with both MOS: {len(pam_clips)}, MusicEval clips: {len(me_clips)}", flush=True)

    results: Dict[str, Any] = {"n_pam": len(pam_clips), "n_musiceval": len(me_clips)}

    if args.stage in ("all", "post_hoc"):
        results["post_hoc"] = stage_post_hoc(scorer, pam_clips, me_clips, args.cmi_root, args.device)

    # Feature extraction needed for stages ii–iv
    if args.stage in ("all", "crossdist", "stratified", "scaling"):
        print("\nExtracting features for supervised probes...", flush=True)
        pam_X, pam_mus, pam_aln = _extract_features(scorer, pam_clips, args.cmi_root, "PAM")
        me_X, me_mus, me_aln = _extract_features(scorer, me_clips, args.cmi_root, "MusicEval")
        all_X = torch.cat([pam_X, me_X], dim=0)
        all_mus = np.concatenate([pam_mus, me_mus])
        all_aln = np.concatenate([pam_aln, me_aln])
        # free GPU before training
        del scorer
        torch.cuda.empty_cache()
        if args.stage in ("all", "crossdist"):
            results["crossdist"] = stage_crossdist(pam_X, pam_aln, me_X, me_aln, args.device)
        if args.stage in ("all", "stratified"):
            results["stratified"] = stage_stratified(all_X, all_mus, all_aln, args.n_seeds_stab, args.device)
        if args.stage in ("all", "scaling"):
            results["scaling"] = stage_scaling(all_X, all_mus, all_aln, args.n_seeds_scale, args.device)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
