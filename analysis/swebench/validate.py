#!/usr/bin/env python3
"""Independent checks on the two claims that matter. Standard library only.

itemwise's own statistics come from classical test theory. If a conclusion only
holds under that model, it is not worth publishing - so both headline claims are
re-derived here by methods that share none of its assumptions.

  1. BACKWARDS ITEMS. itemwise 0.1 reported 37 broken graders among the top 25
     using a fixed r_pb < -0.05 cutoff. This shuffles the matrix while holding
     both item difficulties and system totals fixed (a curveball / checkerboard
     randomisation) and counts how many backwards items that produces by chance.
     If the observed count sits inside the null, the finding was noise.

  2. THE RANKING. Resample the 500 instances with replacement and re-rank. If a
     leaderboard position is real it should survive; if it is noise, the order
     will scramble. This makes no measurement-error assumption at all.

    python3 validate.py                 # ~1-2 minutes
    python3 validate.py --reps 50 --boot 500    # quicker, noisier
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_THRESHOLD = -0.05


def point_biserial(x: list[int], totals: list[float]) -> float | None:
    """Corrected (item-removed) point-biserial - what itemwise 0.1 flagged on."""
    n = len(x)
    if 0 == sum(x) or sum(x) == n:
        return None
    rest = [t - xi for t, xi in zip(totals, x)]
    mx = sum(x) / n
    mr = sum(rest) / n
    num = sum((a - mx) * (b - mr) for a, b in zip(x, rest))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dr = math.sqrt(sum((b - mr) ** 2 for b in rest))
    if dx == 0 or dr == 0:
        return None
    return num / (dx * dr)


def count_backwards(matrix: list[list[int]]) -> int:
    totals = [float(sum(col)) for col in zip(*matrix)]
    n = 0
    for row in matrix:
        r = point_biserial(row, totals)
        if r is not None and r < OLD_THRESHOLD:
            n += 1
    return n


def curveball(matrix: list[list[int]], swaps: int, rng: random.Random) -> list[list[int]]:
    """Shuffle preserving every row sum AND every column sum.

    Both margins fixed means the null keeps each instance exactly as hard as it
    is and each system exactly as good as it is, and destroys only the pairing
    between them - which is precisely the thing 'backwards' claims to detect.
    """
    sets = [set(i for i, v in enumerate(row) if v) for row in matrix]
    n_rows = len(sets)
    for _ in range(swaps):
        i, j = rng.randrange(n_rows), rng.randrange(n_rows)
        if i == j:
            continue
        only_i = list(sets[i] - sets[j])
        only_j = list(sets[j] - sets[i])
        k = min(len(only_i), len(only_j))
        if k == 0:
            continue
        rng.shuffle(only_i)
        rng.shuffle(only_j)
        for a, b in zip(only_i[: rng.randint(1, k)], only_j):
            sets[i].discard(a); sets[i].add(b)
            sets[j].discard(b); sets[j].add(a)
    width = len(matrix[0])
    return [[1 if c in s else 0 for c in range(width)] for s in sets]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=int, default=25)
    ap.add_argument("--reps", type=int, default=200, help="curveball randomisations")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads((HERE / "matrix.json").read_text())
    scores, names = data["scores"], data.get("names", {})
    totals = {s: sum(v) for s, v in scores.items()}
    top = sorted(totals, key=lambda s: -totals[s])[: args.cohort]
    n_items = len(data["item_ids"])
    matrix = [[scores[m][i] for m in top] for i in range(n_items)]
    rng = random.Random(args.seed)

    # ------------------------------------------------------------- check 1
    observed = count_backwards(matrix)
    print(f"CHECK 1 - backwards items, top {args.cohort}")
    print(f"  observed under the old r_pb < {OLD_THRESHOLD} rule: {observed}")
    print(f"  building null from {args.reps} curveball randomisations...")
    null = []
    for r in range(args.reps):
        null.append(count_backwards(curveball(matrix, 20_000, rng)))
        if (r + 1) % 50 == 0:
            print(f"    {r + 1}/{args.reps}")
    mean = sum(null) / len(null)
    sd = math.sqrt(sum((v - mean) ** 2 for v in null) / len(null))
    p = sum(1 for v in null if v >= observed) / len(null)
    print(f"  null: mean {mean:.1f}, sd {sd:.1f}, range {min(null)}-{max(null)}")
    print(f"  p(null >= observed) = {p:.4f}")
    print("  -> " + ("NOT distinguishable from chance. The finding was noise."
                     if p > 0.05 else "beyond chance - real signal."))

    # ------------------------------------------------------------- check 2
    print(f"\nCHECK 2 - ranking stability, {args.boot} bootstrap resamples over items")
    wins: Counter[str] = Counter()
    cols = list(zip(*matrix))                     # one tuple per system
    for _ in range(args.boot):
        idx = [rng.randrange(n_items) for _ in range(n_items)]
        best, best_score = None, -1
        for m, col in zip(top, cols):
            s = sum(col[i] for i in idx)
            if s > best_score:
                best, best_score = m, s
        wins[best] += 1
    print("  P(this system is genuinely #1):")
    for m, w in wins.most_common(8):
        print(f"    {w / args.boot:6.1%}  {totals[m] / (n_items / 100):5.1f}%  "
              f"{names.get(m, m)[:48]}")
    print("\n  A leaderboard position that only holds for one particular draw of")
    print("  500 instances is a property of the sample, not of the system.")


if __name__ == "__main__":
    main()
