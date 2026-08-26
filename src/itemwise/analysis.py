"""The analysis entry point: turn a RunResult into an actionable Report."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .models import Item, RunResult
from .stats import (
    backwards_p_value,
    benjamini_hochberg,
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
    backwards_p: float | None = None
    backwards_significant: bool = False

    @property
    def verdict(self) -> Verdict:
        if self.difficulty in (0.0, 1.0):
            return "dead"
        if self.backwards_significant:
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
            p = "" if self.backwards_p is None else f" (p={self.backwards_p:.4f})"
            return (
                "Weaker models pass this more often than stronger ones, by more "
                f"than chance explains{p}. That usually means a broken grader, an "
                "ambiguous prompt, or a case rewarding a shortcut. Investigate "
                "before trusting it."
            )
        if self.verdict == "weak" and self.point_biserial < 0.0:
            p = "" if self.backwards_p is None else f" p={self.backwards_p:.2f}"
            return (
                "Points slightly the wrong way, but not by more than noise would "
                f"produce on its own{p}. Not evidence of a broken grader. It is "
                "still carrying no usable signal."
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
    backwards_fdr: float = 0.05

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

    @property
    def n_models(self) -> int:
        return len(self.models)

    def backwards_detectable(self) -> bool:
        """Could a backwards item be established at all, with this many models?

        The backwards test is exact: with n models and k passing an item, the
        smallest p-value reachable is 1 / C(n, k), because the observed split is
        one of that many equally likely splits. Below roughly eight models even a
        perfectly inverted item cannot clear the significance bar - not because
        the suite is clean, but because there is not enough evidence to say so.

        When this returns False, an empty backwards list means "cannot tell",
        not "nothing wrong". Add models before drawing a conclusion.
        """
        n = self.n_models
        return any(
            1.0 / math.comb(n, k) <= self.backwards_fdr for k in range(1, n)
        )

    def suspicious_items(self) -> list[ItemStats]:
        """Items pointing the wrong way that the evidence cannot yet convict.

        These are not findings. They are the shortlist to re-examine if you add
        more models, ordered by how unlikely they already look.
        """
        out = [
            s for s in self.stats
            if not s.backwards_significant
            and s.backwards_p is not None
            and s.point_biserial < 0.0
        ]
        return sorted(out, key=lambda s: s.backwards_p or 1.0)

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


def analyze(result: RunResult, backwards_fdr: float = 0.05) -> Report:
    """Compute item and suite diagnostics for a completed run.

    ``backwards_fdr`` is the false-discovery rate allowed when calling items
    backwards. Every item is tested against the null that it is unrelated to
    model ability, and the resulting p-values are corrected across the whole
    suite - so the expected share of mistakes among the items reported as
    backwards is at most this. Raise it to widen the net, lower it to be
    stricter. Setting it high enough to catch everything will also catch noise;
    that is the trade, and it is now yours to make explicitly rather than a
    hard-coded threshold making it for you.
    """
    if not 0.0 < backwards_fdr < 1.0:
        raise ValueError(f"backwards_fdr must be between 0 and 1, got {backwards_fdr}")
    if len(result.models) < 2:
        raise ValueError(
            "item analysis needs at least 2 models to compare - with one model "
            "every item looks the same and no item can discriminate"
        )
    if len(result.suite) < 2:
        raise ValueError("need at least 2 items to analyse a suite")

    totals = result.total_scores()
    matrix = result.item_matrix()

    p_values = [backwards_p_value(row, totals) for row in matrix]
    flagged = benjamini_hochberg(p_values, q=backwards_fdr)

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
                backwards_p=p_values[i],
                backwards_significant=flagged[i] and point_biserial(
                    row, totals, corrected=True
                ) < 0.0,
            )
        )

    return Report(
        stats=stats,
        alpha=cronbach_alpha(matrix),
        sem=standard_error_of_measurement(matrix),
        models=result.models,
        model_totals={m: int(sum(result.scores[m])) for m in result.models},
        backwards_fdr=backwards_fdr,
    )
