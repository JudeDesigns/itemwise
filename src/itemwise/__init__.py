"""itemwise - find the eval cases that actually tell you something.

Most LLM eval suites are full of dead weight: cases every model passes, cases
every model fails, and cases where the grader is quietly broken. You pay to
run all of them on every commit, and they move no decision.

itemwise applies classical test theory - the same item analysis used to build
professional certification exams - to your eval suite, and tells you which
cases carry signal and which are costing you money to learn nothing.

    from itemwise import Suite, RunResult, analyze

    suite = Suite.from_jsonl("evals.jsonl")
    result = RunResult(suite, {"gpt-x": [...], "claude-y": [...]})
    report = analyze(result)

    print(f"{report.wasted_fraction:.0%} of this suite is dead weight")
    for s in report.backwards_items():
        print(s.item.id, s.diagnosis)
"""

from .analysis import ItemStats, Report, analyze
from .models import Item, RunResult, Suite
from .report import to_html, to_text
from .stats import (
    cronbach_alpha,
    difficulty,
    discrimination_index,
    point_biserial,
    standard_error_of_measurement,
)

__version__ = "0.1.0"

__all__ = [
    "Item",
    "ItemStats",
    "Report",
    "RunResult",
    "Suite",
    "analyze",
    "cronbach_alpha",
    "difficulty",
    "discrimination_index",
    "point_biserial",
    "standard_error_of_measurement",
    "to_html",
    "to_text",
]
