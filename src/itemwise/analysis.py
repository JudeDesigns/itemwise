"""The analysis entry point: turn a RunResult into an actionable Report."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .models import Item, RunResult
from .stats import (
    cronbach_alpha,
    difficulty,
    discrimination_index,
    point_biserial,
    standard_error_of_measurement,
    variance,
)

Verdict = Literal["dead", "backwards", "weak", "acceptable", "strong"]

# Two-tailed normal deviates. Enumerated rather than pulled from scipy so the
# package stays dependency-free and the numbers are auditable.
_Z = {0.68: 1.0, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


@dataclass(frozen=True)
class ItemStats:
    """Diagnostics for a single eval case."""

    item: Item
    index: int
    difficulty: float
    discrimination: float
    point_biserial: float
    n_pass: int
    n_models: int

    @property
    def verdict(self) -> Verdict:
        if self.difficulty in (0.0, 1.0):
            return "dead"
        if self.point_biserial < -0.05:
            return "backwards"
        if self.point_biserial < 0.20:
            return "weak"
        if self.point_biserial < 0.40:
            return "acceptable"
        return "strong"

    @property
    def diagnosis(self) -> str:
        """One sentence a human can act on."""
        if self.difficulty == 1.0:
            return (
                "Every model passes. This item costs tokens on every run and "
                "tells you nothing. Retire it, or make it harder."
            )
        if self.difficulty == 0.0:
            return (
                "No model passes. Either the case is genuinely beyond current "
                "models - in which case park it in a separate frontier suite - "
                "or the grader is wrong. Check the grader first; it usually is."
            )
        if self.verdict == "backwards":
            return (
                "Weaker models pass this more often than stronger ones. That "
                "almost always means a broken grader, an ambiguous prompt, or a "
                "case rewarding a shortcut. Investigate before trusting it."
            )
        if self.verdict == "weak":
            return (
                "Barely separates strong models from weak ones. Low priority to "
                "fix, but it is padding your suite rather than informing it."
            )
        if self.verdict == "acceptable":
            return "Contributing real signal. Fine as it is."
        return "Discriminates cleanly between strong and weak models. Keep."


@dataclass(frozen=True)
class Report:
    """Suite-level and item-level diagnostics."""

    stats: list[ItemStats]
    alpha: float
    sem: float
    models: list[str]
    model_totals: dict[str, int]

    # ---- item selections -------------------------------------------------

    def dead_items(self) -> list[ItemStats]:
        """Items every model passes or every model fails. Zero information."""
        return [s for s in self.stats if s.verdict == "dead"]

    def backwards_items(self) -> list[ItemStats]:
        """Items where weaker models outperform stronger ones. Usually bugs."""
        return [s for s in self.stats if s.verdict == "backwards"]

    def weak_items(self) -> list[ItemStats]:
        return [s for s in self.stats if s.verdict == "weak"]

    def strong_items(self) -> list[ItemStats]:
        return [s for s in self.stats if s.verdict == "strong"]

    def ranked(self) -> list[ItemStats]:
        """Every item, most informative first."""
        return sorted(self.stats, key=lambda s: s.point_biserial, reverse=True)

    # ---- suite-level -----------------------------------------------------

    @property
    def n_items(self) -> int:
        return len(self.stats)

    @property
    def signal_ratio(self) -> float:
        """Fraction of items carrying usable signal (not dead, not backwards)."""
        if not self.stats:
            return 0.0
        useful = sum(1 for s in self.stats if s.verdict in ("weak", "acceptable", "strong"))
        return useful / len(self.stats)

    @property
    def wasted_fraction(self) -> float:
        """Fraction of every run spent on items that inform nothing."""
        if not self.stats:
            return 0.0
        return (len(self.dead_items()) + len(self.backwards_items())) / len(self.stats)

    def min_real_gap(self, confidence: float = 0.95) -> float:
        """Smallest score difference that is not plausibly measurement noise.

            SED = SEM * sqrt(2)          standard error of a difference
            gap = z(confidence) * SED

        Any two models closer together than this are, on the evidence of this
        suite, tied - however confidently a leaderboard orders them.
        """
        if confidence not in _Z:
            raise ValueError(
                f"confidence must be one of {sorted(_Z)}, got {confidence!r}"
            )
        return _Z[confidence] * self.sem * math.sqrt(2)

    def separates(self, model_a: str, model_b: str,
                  confidence: float = 0.95) -> bool:
        """Is the gap between two models bigger than the measurement noise?

        Defaults to 95% confidence. If this returns False, the ordering
        between these two models is not supported by this suite - you have
        measured the same thing twice and drawn a line between the readings.
        """
        for m in (model_a, model_b):
            if m not in self.model_totals:
                raise ValueError(f"unknown model {m!r}")
        gap = abs(self.model_totals[model_a] - self.model_totals[model_b])
        return gap > self.min_real_gap(confidence)

    def alpha_verdict(self) -> str:
        a = self.alpha
        if a >= 0.90:
            return "very high - check for near-duplicate items"
        if a >= 0.80:
            return "good"
        if a >= 0.70:
            return "acceptable"
        if a >= 0.60:
            return "questionable"
        return "low - this suite may be measuring several unrelated things"


def analyze(result: RunResult) -> Report:
    """Compute item and suite diagnostics for a completed run."""
    if len(result.models) < 2:
        raise ValueError(
            "item analysis needs at least 2 models to compare - with one model "
            "every item looks the same and no item can discriminate"
        )
    if len(result.suite) < 2:
        raise ValueError("need at least 2 items to analyse a suite")

    totals = result.total_scores()
    matrix = result.item_matrix()

    stats: list[ItemStats] = []
    for i, item in enumerate(result.suite.items):
        row = matrix[i]
        stats.append(
            ItemStats(
                item=item,
                index=i,
                difficulty=difficulty(row),
                discrimination=discrimination_index(row, totals),
                point_biserial=point_biserial(row, totals, corrected=True),
                n_pass=sum(row),
                n_models=len(row),
            )
        )

    return Report(
        stats=stats,
        alpha=cronbach_alpha(matrix),
        sem=standard_error_of_measurement(matrix),
        models=result.models,
        model_totals={m: int(sum(result.scores[m])) for m in result.models},
    )
