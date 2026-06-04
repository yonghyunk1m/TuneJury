"""Per-system retrain + 'no catastrophic forgetting' probe (Paper Appendix §A.D).

Trains a TuneJury-style head from scratch on
   bench_clean.train (canonical 571 MA pairs)
   plus Feb-Mar (598)
   plus April (397)
and evaluates on three held-out partitions: bench_clean.test, Feb-Mar
held-out, April held-out. Used to back the 'no catastrophic forgetting'
and 'why anchor calibration outperforms naive retraining' paragraphs.

Inputs:
  --feat-dir-fm        Feb-Mar (incl. bench-clean) feature cache
  --feat-dir-apr       April feature cache
  --bench-split        bench_clean split json (train/valid/test UUIDs)
  --increment-split    v2 increment split json (Feb-Mar post-cutoff UUIDs)
  --fm-label-csv       CSV labels for Feb-Mar
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunejury.model import TuneJury  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# torch.load weights_only=False shim
_orig_load = torch.load
torch.load = lambda *a, **kw: (_orig_load(*a, **{**kw, "weights_only": False})
                                if "weights_only" not in kw else _orig_load(*a, **kw))


def to1d(t):
    return t.float().reshape(-1)


class _DS(Dataset):
    def __init__(self, cache: dict, uuids: list[str], target_of):
        self.cache = cache
        self.uuids = [u for u in uuids if u in cache]
        self.target_of = target_of

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, i):
        u = self.uuids[i]
        c = self.cache[u]
        text = c["text"]
        x_a = torch.cat([c["clap_a"], text, c["mert_a"]])
        x_b = torch.cat([c["clap_b"], text, c["mert_b"]])
        return {"x_a": x_a, "x_b": x_b,
                "target": torch.tensor([self.target_of(u)], dtype=torch.float32)}


def load_pt(p: Path) -> dict:
    d = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "text": to1d(d["text_emb"]),
        "clap_a": to1d(d["clap_a"]), "clap_b": to1d(d["clap_b"]),
        "mert_a": to1d(d["mert_a"]), "mert_b": to1d(d["mert_b"]),
        "pt_winner": str(d.get("winner", "tie")).strip().lower(),
    }


def train_one(model, train_loader, val_loader, epochs=50, lr=1e-4, wd=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x_a = batch["x_a"].to(DEVICE)
            x_b = batch["x_b"].to(DEVICE)
            t = batch["target"].to(DEVICE).view(-1)
            logit = model(x_a, x_b).view(-1)
            loss = F.binary_cross_entropy_with_logits(logit, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # val
        model.eval()
        v, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x_a = batch["x_a"].to(DEVICE)
                x_b = batch["x_b"].to(DEVICE)
                t = batch["target"].to(DEVICE).view(-1)
                logit = model(x_a, x_b).view(-1)
                v += F.binary_cross_entropy_with_logits(logit, t, reduction="sum").item()
                n += t.numel()
        if v / n < best - 1e-5:
            best = v / n
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


def eval_acc(model, cache, uuids, target_of):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for u in uuids:
            if u not in cache:
                continue
            t = target_of(u)
            if t == 0.5:
                continue  # exclude ties from binary accuracy
            c = cache[u]
            text = c["text"]
            x_a = torch.cat([c["clap_a"], text, c["mert_a"]]).unsqueeze(0).to(DEVICE)
            x_b = torch.cat([c["clap_b"], text, c["mert_b"]]).unsqueeze(0).to(DEVICE)
            logit = model(x_a, x_b).item()
            pred = 1.0 if logit > 0 else 0.0
            if pred == t:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feat-dir-fm", required=True)
    parser.add_argument("--feat-dir-apr", required=True)
    parser.add_argument("--bench-split", required=True)
    parser.add_argument("--increment-split", required=True)
    parser.add_argument("--fm-label-csv", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    bc = json.loads(Path(args.bench_split).read_text())
    incr = json.loads(Path(args.increment_split).read_text())
    fm_rows = list(csv.DictReader(open(args.fm_label_csv)))
    fm_label = {r["battle_id"]: (1.0 if r["preference"] == "A" else 0.0) for r in fm_rows}

    feat_fm = Path(args.feat_dir_fm)
    feat_apr = Path(args.feat_dir_apr)
    cache: dict[str, dict] = {}
    for u in bc["train"] + bc["valid"] + bc["test"]:
        p = feat_fm / f"{u}.pt"
        if p.exists():
            cache[u] = load_pt(p)
    for u in incr["new_uuids"]:
        p = feat_fm / f"{u}.pt"
        if p.exists():
            cache[u] = load_pt(p)
    apr_uuids = []
    for p in sorted(feat_apr.glob("*.pt")):
        cache[p.stem] = load_pt(p)
        apr_uuids.append(p.stem)
    print(f"cache: {len(cache)} pairs")

    fm_in_v2 = [u for u in incr["new_uuids"] if u in fm_label and u in cache]
    print(f"Feb-Mar in v2 universe: {len(fm_in_v2)}")
    print(f"April features: {len(apr_uuids)}")

    def target_of(u: str) -> float:
        if u in fm_label:
            return fm_label[u]
        c = cache[u]
        if c["pt_winner"] == "model_a":
            return 1.0
        if c["pt_winner"] == "model_b":
            return 0.0
        return 0.5

    conditions = {
        "A: bench_clean only (571)":
            lambda: bc["train"],
        "B: bench_clean + Feb-Mar (571+598)":
            lambda: bc["train"] + fm_in_v2,
        "C: bench_clean + April only (571+397)":
            lambda: bc["train"] + apr_uuids,
        "D: bench_clean + FM + Apr (571+995)":
            lambda: bc["train"] + fm_in_v2 + apr_uuids,
    }
    test_sets = {
        "bench_clean.test": bc["test"],
        "Feb-Mar (598)": fm_in_v2,
        "April (397)": apr_uuids,
    }

    all_results: dict[tuple[str, str], list[float]] = defaultdict(list)
    for name, train_fn in conditions.items():
        print(f"\n=== {name} ===")
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = TuneJury(input_dim=2048).to(DEVICE)
            tl = DataLoader(_DS(cache, train_fn(), target_of), batch_size=32, shuffle=True)
            vl = DataLoader(_DS(cache, bc["valid"], target_of), batch_size=32, shuffle=False)
            train_one(model, tl, vl)
            line = f"  seed {seed}: "
            for tname, tuuids in test_sets.items():
                acc = eval_acc(model, cache, tuuids, target_of)
                all_results[(name, tname)].append(acc)
                line += f"{tname}={acc:.3f}  "
            print(line)

    print("\n=== Summary (mean ± std over seeds) ===")
    hdr = f'{"condition":40s}  ' + "  ".join(f"{t:>17s}" for t in test_sets)
    print(hdr)
    for cond in conditions:
        row = [
            f"{np.mean(all_results[(cond, t)]):.3f}±{np.std(all_results[(cond, t)]):.3f}"
            for t in test_sets
        ]
        print(f"{cond:40s}  " + "  ".join(f"{r:>17s}" for r in row))


if __name__ == "__main__":
    main()
