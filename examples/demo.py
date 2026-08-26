"""End-to-end demo on a synthetic suite that looks like a real one.

Run:  PYTHONPATH=src python3 examples/demo.py
"""

import random

from itemwise import RunResult, Suite, analyze, to_html, to_text

random.seed(7)

# Twelve models, not five. That is deliberate: the backwards test is exact, and
# with five models the most extreme possible split is still one of only ten, so
# no item can ever clear a 0.05 bar. A suite is only as diagnosable as the
# number of models you ran it on. See Report.backwards_detectable().
ABILITY = {
    "frontier-a": 0.94, "frontier-b": 0.91, "frontier-c": 0.88,
    "strong-a": 0.82, "strong-b": 0.78, "mid-tier-a": 0.71,
    "mid-tier-b": 0.66, "mid-tier-c": 0.59, "small-open-a": 0.51,
    "small-open-b": 0.45, "tiny-open-a": 0.36, "tiny-open-b": 0.28,
}
MODELS = list(ABILITY)


def build_suite() -> tuple[Suite, dict[str, list[int]]]:
    items: list[dict] = []
    scores: dict[str, list[int]] = {m: [] for m in MODELS}

    def add(item_id: str, tags: list[str], outcome) -> None:
        items.append({"id": item_id, "prompt": f"<{item_id}>", "tags": tags})
        for m in MODELS:
            scores[m].append(outcome(m))

    # 12 items every model passes - the sanity checks nobody ever deleted.
    for i in range(12):
        add(f"smoke-{i:02d}", ["smoke"], lambda m: 1)

    # 5 items nothing passes - aspirational cases, or a broken grader.
    for i in range(5):
        add(f"frontier-{i:02d}", ["frontier"], lambda m: 0)

    # 20 well-behaved items: pass probability tracks model ability.
    for i in range(20):
        threshold = 0.25 + i * 0.03
        add(f"reasoning-{i:02d}", ["reasoning"],
            lambda m, t=threshold: int(ABILITY[m] > t))

    # 3 broken items: the grader rewards a shortcut smaller models take.
    for i in range(3):
        add(f"format-{i:02d}", ["formatting"],
            lambda m: int(ABILITY[m] < 0.6))

    return Suite.from_dicts(items), scores


def main() -> None:
    suite, scores = build_suite()
    report = analyze(RunResult(suite, scores))

    print(to_text(report, max_items=12))
    print()
    print(f"Suite size:            {report.n_items} items")
    print(f"Dead weight:           {report.wasted_fraction:.0%}")
    print(f"Backwards items:       {[s.item.id for s in report.backwards_items()]}")
    print(f"Backwards test usable: {report.backwards_detectable()} "
          f"({report.n_models} models)")
    print(f"Min. real gap (95%):   {report.min_real_gap():.2f} items")

    keep = [s for s in report.stats if s.verdict in ("acceptable", "strong")]
    saved = 1 - len(keep) / report.n_items
    print()
    print(f"Keeping only informative items: {len(keep)} of {report.n_items} "
          f"({saved:.0%} fewer calls per run, same conclusions).")

    with open("report.html", "w", encoding="utf-8") as fh:
        fh.write(to_html(report, title="Demo suite report"))
    print("Wrote report.html")


if __name__ == "__main__":
    main()
