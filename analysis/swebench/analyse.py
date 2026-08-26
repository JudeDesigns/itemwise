#!/usr/bin/env python3
"""Run itemwise over the SWE-bench Verified leaderboard.

Reproduces every number in FINDINGS.md. Reads matrix.json (committed; rebuild it
with fetch.py) and needs nothing but the standard library and itemwise itself.

    python3 analyse.py
    python3 analyse.py --cohort 25 --report top25.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))   # the itemwise checkout itself

from itemwise import RunResult, Suite, analyze, to_html  # noqa: E402

COHORTS = (5, 10, 25, 50, 100, 134)


def load() -> tuple[list[str], dict[str, list[int]], dict[str, str]]:
    data = json.loads((HERE / "matrix.json").read_text())
    return data["item_ids"], data["scores"], data.get("names", {})


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    rank_a = {k: i for i, k in enumerate(sorted(a, key=lambda k: -a[k]))}
    rank_b = {k: i for i, k in enumerate(sorted(b, key=lambda k: -b[k]))}
    n = len(a)
    d2 = sum((rank_a[k] - rank_b[k]) ** 2 for k in a)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=int, default=25,
                    help="how many top systems to examine in detail (default 25)")
    ap.add_argument("--report", help="also write an HTML report to this path")
    args = ap.parse_args()

    item_ids, scores, names = load()
    suite = Suite.from_dicts([{"id": i} for i in item_ids])
    totals = {s: sum(v) for s, v in scores.items()}
    ranked = sorted(totals, key=lambda s: -totals[s])
    label = lambda s: names.get(s, s)

    print(f"SWE-bench Verified: {len(item_ids)} instances, {len(scores)} submissions\n")

    # ---------------------------------------------------------------- finding 1
    print("=" * 78)
    print("FINDING 1 - how much of the suite still measures anything")
    print("=" * 78)
    print(f"{'cohort':>7} {'dead':>10} {'%':>7} {'everyone':>9} {'nobody':>7} "
          f"{'SEM':>6} {'min real gap':>13}")
    for n in COHORTS:
        if n > len(ranked):
            continue
        rep = analyze(RunResult(suite, {m: scores[m] for m in ranked[:n]}))
        dead = rep.dead_items()
        everyone = sum(1 for s in dead if s.difficulty == 1.0)
        nobody = sum(1 for s in dead if s.difficulty == 0.0)
        print(f"{n:>7} {len(dead):>7}/{len(item_ids)} {len(dead)/len(item_ids):>6.1%} "
              f"{everyone:>9} {nobody:>7} {rep.sem:>6.2f} "
              f"{rep.min_real_gap()/(len(item_ids)/100):>11.2f} pp")

    top = ranked[: args.cohort]
    rep = analyze(RunResult(suite, {m: scores[m] for m in top}))
    full = {m: totals[m] for m in top}
    print()
    for name, verdicts in (("non-dead", ("weak", "acceptable", "strong")),
                           ("informative (r_pb >= 0.20)", ("acceptable", "strong"))):
        idx = [s.index for s in rep.stats if s.verdict in verdicts]
        trimmed = {m: sum(scores[m][i] for i in idx) for m in top}
        print(f"  keeping only {name}: {len(idx)} items -> "
              f"same top-{args.cohort} ordering at Spearman {spearman(full, trimmed):+.3f}")

    # ---------------------------------------------------------------- finding 2
    print()
    print("=" * 78)
    print(f"FINDING 2 - is the top-{args.cohort} ordering real?")
    print("=" * 78)
    gap = rep.min_real_gap()
    print(f"a gap must exceed {gap:.2f} items "
          f"({gap/(len(item_ids)/100):.2f} pp) to survive 95% confidence\n")
    indistinct = 0
    for a, b in zip(top, top[1:]):
        d = totals[a] - totals[b]
        real = d > gap
        indistinct += not real
        print(f"  {totals[a]/(len(item_ids)/100):5.1f}% vs {totals[b]/(len(item_ids)/100):5.1f}%"
              f"  gap {d:3}  {'real' if real else 'inside the noise':<16}"
              f"  {label(a)[:30]:32} / {label(b)[:30]}")
    print(f"\n  {indistinct}/{len(top)-1} adjacent pairs are statistically indistinguishable")
    tied = [m for m in top[1:] if totals[top[0]] - totals[m] <= gap]
    print(f"  {len(tied)+1} systems are tied for #1")

    # ---------------------------------------------------------------- finding 3
    print()
    print("=" * 78)
    print("FINDING 3 - broken graders")
    print("=" * 78)
    for n in COHORTS:
        if n > len(ranked):
            continue
        r = analyze(RunResult(suite, {m: scores[m] for m in ranked[:n]}))
        print(f"  top {n:>3}: {len(r.backwards_items()):>2} backwards "
              f"(detectable={r.backwards_detectable()}), "
              f"{len(r.suspicious_items()):>3} point the wrong way without reaching significance")
    print("\n  itemwise 0.1 reported 37 of these for the top 25 on a fixed threshold.")
    print("  A difficulty- and total-preserving null produces 38.2 +/- 4.2 by chance;")
    print("  run validate.py to reproduce that. All 37 were noise. See CHANGELOG.md.")

    if args.report:
        Path(args.report).write_text(
            to_html(rep, title=f"SWE-bench Verified - top {args.cohort}"))
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
