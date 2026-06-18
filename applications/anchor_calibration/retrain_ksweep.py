"""Retrain K-sweep against the released checkpoint for the post-cutoff slice.

For each seed in [0, 1, 2, 3, 4] and each K in K_VALUES, trains a TuneJury
head from scratch on:
    bench_clean MA train (571 pairs)  ∪  train_uuids[:K]  (post-cutoff)
and evaluates on the same 299-pair held-out test split that the anchor
calibration K-sweep uses. Outputs `results/ood_retrain.json` with the
per-seed accuracy lists, matching the schema of `results/ood_repro.json`
(produced by `run_experiment.py`).

This is the "retraining (R)" row in paper §A.D Table `ood_scaling`.

Concat order matches the released-ckpt eval path: [clap_audio, mert_audio,
text_emb].
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUES = [0, 3, 10, 30, 100, 250]
SEEDS = [0, 1, 2, 3, 4]


# ---- Concat order [clap, mert, text] matches released-ckpt eval path ----
def cat_input(feat: dict, side: str) -> torch.Tensor:
    return torch.cat(
        [
            feat[f"clap_{side}"].float(),
            feat[f"mert_{side}"].float(),
            feat["text_emb"].float(),
        ]
    )


def winner_to_target(winner: str) -> float:
    return {"model_a": 1.0, "model_b": 0.0, "tie": 0.5}[winner]


def load_pair_cache(feat_dir: Path, uuids: list[str], target_fn) -> list[dict]:
    """Return list of records with x_a, x_b, target tensors on DEVICE."""
    out = []
    for u in uuids:
        d = torch.load(feat_dir / f"{u}.pt", map_location="cpu", weights_only=False)
        target = target_fn(u, d)
        if target is None:
            continue
        x_a = cat_input(d, "a")
        x_b = cat_input(d, "b")
        out.append({"x_a": x_a, "x_b": x_b, "target": torch.tensor(target, dtype=torch.float32)})
    return out


def train_one_run(
    train_recs: list[dict],
    val_recs: list[dict],
    test_recs: list[dict],
    max_epochs: int = 100,
    patience: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    init_seed: int = 0,
) -> float:
    """Train TuneJury head from scratch, return test accuracy at best-val epoch."""
    g = torch.Generator(device="cpu").manual_seed(init_seed)
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    model = TuneJury(input_dim=2048).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def batched(recs):
        idx = torch.randperm(len(recs), generator=g).tolist()
        for i in range(0, len(idx), batch_size):
            chunk = [recs[j] for j in idx[i : i + batch_size]]
            x_a = torch.stack([r["x_a"] for r in chunk]).to(DEVICE)
            x_b = torch.stack([r["x_b"] for r in chunk]).to(DEVICE)
            tgt = torch.stack([r["target"] for r in chunk]).to(DEVICE)
            yield x_a, x_b, tgt

    def evaluate(recs):
        model.eval()
        with torch.no_grad():
            corr = n = 0
            for r in recs:
                if float(r["target"]) == 0.5:
                    continue
                sa = float(model(r["x_a"].unsqueeze(0).to(DEVICE)).item())
                sb = float(model(r["x_b"].unsqueeze(0).to(DEVICE)).item())
                pred = 1 if sa > sb else 0
                if pred == int(r["target"]):
                    corr += 1
                n += 1
        return corr / max(n, 1)

    best_val = -1.0
    best_test = 0.5
    bad = 0
    for epoch in range(max_epochs):
        model.train()
        for x_a, x_b, tgt in batched(train_recs):
            sa = model(x_a).squeeze(-1)
            sb = model(x_b).squeeze(-1)
            logits = sa - sb
            loss = F.binary_cross_entropy_with_logits(logits, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()

        v = evaluate(val_recs) if val_recs else 0.0
        if v > best_val:
            best_val = v
            best_test = evaluate(test_recs)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best_test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat-dir", required=True, help="MusicArena features cache (shared by bench-clean and post-cutoff)")
    ap.add_argument("--bench-split", required=True, help="MusicArena-random_split_bench_clean.json")
    ap.add_argument("--increment-split", required=True, help="MusicArena_v2_increment.json")
    ap.add_argument("--label-csv", required=True, help="ma_postcut_scored.csv")
    ap.add_argument("--out", default="results/ood_retrain.json")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--max-epochs", type=int, default=100)
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    bench = json.loads(Path(args.bench_split).read_text())
    increment = json.loads(Path(args.increment_split).read_text())
    csv_rows = list(csv.DictReader(open(args.label_csv)))
    csv_label = {r["battle_id"]: r["preference"] for r in csv_rows}

    bench_train_uuids = bench["train"]
    bench_val_uuids = bench["valid"]
    universe = [
        u for u in increment["new_uuids"]
        if u in csv_label and (feat_dir / f"{u}.pt").exists()
    ]
    print(f"bench train: {len(bench_train_uuids)}, bench val: {len(bench_val_uuids)}, universe: {len(universe)}", flush=True)

    # Bench-clean: target from feature winner
    bench_train_recs = load_pair_cache(
        feat_dir, bench_train_uuids,
        lambda u, d: winner_to_target(d["winner"]) if "winner" in d else None,
    )
    bench_val_recs = load_pair_cache(
        feat_dir, bench_val_uuids,
        lambda u, d: winner_to_target(d["winner"]) if "winner" in d else None,
    )
    print(f"bench train recs: {len(bench_train_recs)} (after tie filter or missing winner)", flush=True)
    print(f"bench val recs:   {len(bench_val_recs)}", flush=True)

    # Post-cutoff: target from CSV preference
    postcut_targets = {u: (1.0 if csv_label[u] == "A" else 0.0) for u in universe}

    results: dict[int, list[float]] = defaultdict(list)
    for K in K_VALUES:
        for seed in SEEDS:
            rng = random.Random(seed)
            shuf = universe.copy()
            rng.shuffle(shuf)
            train_uuids = shuf[:299]
            test_uuids = shuf[299:598]

            # Build training pool
            train_recs = list(bench_train_recs)
            if K > 0:
                for u in train_uuids[:K]:
                    d = torch.load(feat_dir / f"{u}.pt", map_location="cpu", weights_only=False)
                    train_recs.append({
                        "x_a": cat_input(d, "a"),
                        "x_b": cat_input(d, "b"),
                        "target": torch.tensor(postcut_targets[u], dtype=torch.float32),
                    })

            # Test pool (always the same 299 post-cutoff held-out)
            test_recs = []
            for u in test_uuids:
                d = torch.load(feat_dir / f"{u}.pt", map_location="cpu", weights_only=False)
                test_recs.append({
                    "x_a": cat_input(d, "a"),
                    "x_b": cat_input(d, "b"),
                    "target": torch.tensor(postcut_targets[u], dtype=torch.float32),
                })

            acc = train_one_run(
                train_recs,
                bench_val_recs,
                test_recs,
                max_epochs=args.max_epochs,
                patience=args.patience,
                init_seed=seed,
            )
            results[K].append(acc)
            print(f"K={K:>3d}  seed={seed}  acc={acc:.4f}", flush=True)

    out_obj = {"retrain": {str(k): v for k, v in results.items()}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2))

    print("\nSUMMARY (mean ± std, 5 seeds):", flush=True)
    print(f"{'K':>5s}  {'retrain':>22s}", flush=True)
    for K in K_VALUES:
        a = np.array(results[K])
        print(f"{K:>5d}  {a.mean()*100:5.2f} ± {a.std()*100:4.2f}", flush=True)


if __name__ == "__main__":
    main()
