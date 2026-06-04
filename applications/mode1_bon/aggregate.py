"""Aggregate per-N statistics from a Best-of-N results CSV.

Usage:
    python aggregate.py results/acestep_bon100/results.csv
"""

import csv
import statistics
import sys
from collections import defaultdict


def main(csv_path: str):
    rows_by_n = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows_by_n[int(row["N"])].append(float(row["best_score"]))

    print(f"=== Aggregate from {csv_path} ===")
    print(f"{'N':>4} {'count':>6} {'mean':>9} {'median':>9} {'std':>8} {'min':>9} {'max':>9}")
    for n in sorted(rows_by_n):
        scores = rows_by_n[n]
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{n:>4} {len(scores):>6} {mean:>+9.4f} {median:>+9.4f} {std:>8.4f} {min(scores):>+9.4f} {max(scores):>+9.4f}")

    print()
    if 1 in rows_by_n and 4 in rows_by_n:
        delta = statistics.mean(rows_by_n[4]) - statistics.mean(rows_by_n[1])
        print(f"Reward gain N=1 -> N=4: {delta:+.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: aggregate.py <results.csv>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
