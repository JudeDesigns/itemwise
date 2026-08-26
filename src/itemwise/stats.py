"""Classical test theory, applied to LLM evaluation suites.

Every function here is deliberately dependency-free and operates on plain
lists of 0/1 scores, so the maths can be read and checked by eye.

Vocabulary map, for anyone arriving from ML rather than psychometrics:

    examinee / respondent  ->  a model (or a model configuration) under test
    item                   ->  a single eval case
    item score             ->  1 if that model passed that case, else 0
    total score            ->  how many cases that model passed overall
"""

from __future__ import annotations

import math
from typing import Sequence

Scores = Sequence[int]


def _validate(scores: Scores, name: str = "scores") -> list[int]:
    out = list(scores)
    if not out:
        raise ValueError(f"{name} is empty")
    bad = {s for s in out if s not in (0, 1)}
    if bad:
        raise ValueError(f"{name} must contain only 0 or 1, found {sorted(bad)!r}")
    return out


def difficulty(item_scores: Scores) -> float:
    """Difficulty index ``p`` - the proportion of models that passed this item.

    Confusingly but conventionally, a *higher* p means an *easier* item.
    p = 1.0 means every model passed; p = 0.0 means every model failed.
    Both extremes carry zero information about relative model quality.
    """
    s = _validate(item_scores, "item_scores")
    return sum(s) / len(s)


def variance(values: Sequence[float]) -> float:
    """Population variance (denominator N, not N-1).

    Population rather than sample variance is the classical-test-theory
    convention for item statistics, and it is what keeps Cronbach's alpha
    and the point-biserial consistent with each other.
    """
    vals = list(values)
    if not vals:
        raise ValueError("values is empty")
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def discrimination_index(item_scores: Scores, total_scores: Sequence[float],
                         group_fraction: float = 0.27) -> float:
    """Kelley's discrimination index ``D``.

    Rank models by total score, take the top and bottom ``group_fraction``,
    and subtract the bottom group's pass rate from the top group's.

    D = p_upper - p_lower, bounded [-1, 1].

    0.27 is Kelley's classic choice: it maximises the difference between the
    groups relative to sampling error for a normal distribution.

    Interpretation, by long convention:
        D >= 0.40   excellent
        0.30-0.39   good
        0.20-0.29   marginal, worth revising
        < 0.20      poor - the item is not separating strong from weak
        negative    the item is backwards; weaker models pass it more often
    """
    s = _validate(item_scores, "item_scores")
    totals = list(total_scores)
    if len(s) != len(totals):
        raise ValueError("item_scores and total_scores must be the same length")
    if not 0 < group_fraction <= 0.5:
        raise ValueError("group_fraction must be in (0, 0.5]")

    n = len(s)
    k = max(1, int(round(n * group_fraction)))

    order = sorted(range(n), key=lambda i: totals[i])
    lower = order[:k]
    upper = order[-k:]

    p_upper = sum(s[i] for i in upper) / k
    p_lower = sum(s[i] for i in lower) / k
    return p_upper - p_lower


def point_biserial(item_scores: Scores, total_scores: Sequence[float],
                   corrected: bool = True) -> float:
    """Point-biserial correlation between one item and the total score.

    This is the continuous cousin of the discrimination index and is the more
    trustworthy of the two, because it uses every model rather than only the
    extremes.

        r_pb = (M1 - M0) / SD_total * sqrt(p * q)

    where M1 and M0 are mean total scores among models that passed and failed
    the item, p is the item's difficulty and q = 1 - p.

    ``corrected=True`` (the default, and the honest choice) subtracts the
    item's own contribution from the total before correlating. Without that
    correction every item is partly correlated with itself, which inflates
    the statistic - badly so on short suites.

    Returns 0.0 when the item has no variance (everyone passed or everyone
    failed): there is genuinely no correlation to measure, and the item
    carries no information.
    """
    s = _validate(item_scores, "item_scores")
    totals = [float(t) for t in total_scores]
    if len(s) != len(totals):
        raise ValueError("item_scores and total_scores must be the same length")

    if corrected:
        totals = [t - si for t, si in zip(totals, s)]

    p = sum(s) / len(s)
    q = 1.0 - p
    if p in (0.0, 1.0):
        return 0.0

    passed = [t for t, si in zip(totals, s) if si == 1]
    failed = [t for t, si in zip(totals, s) if si == 0]

    sd = math.sqrt(variance(totals))
    if sd == 0.0:
        return 0.0

    m1 = sum(passed) / len(passed)
    m0 = sum(failed) / len(failed)
    return (m1 - m0) / sd * math.sqrt(p * q)


def cronbach_alpha(item_matrix: Sequence[Scores]) -> float:
    """Cronbach's alpha - internal consistency reliability of the suite.

    ``item_matrix`` is a list of items, each a list of per-model 0/1 scores.

        alpha = k/(k-1) * (1 - sum(item variances) / total variance)

    Rough reading for an eval suite:
        >= 0.90   very high (and possibly redundant - are items duplicates?)
        0.80-0.89 good
        0.70-0.79 acceptable
        0.60-0.69 questionable
        < 0.60    the suite is not measuring one coherent thing

    A low alpha is not automatically a fault. A suite that deliberately spans
    several unrelated capabilities *should* score low; that is a signal to
    split it into sub-suites and report them separately, not to bin items.
    """
    items = [_validate(row, "item row") for row in item_matrix]
    if not items:
        raise ValueError("item_matrix is empty")
    k = len(items)
    if k < 2:
        raise ValueError("Cronbach's alpha needs at least 2 items")

    n_models = len(items[0])
    if any(len(row) != n_models for row in items):
        raise ValueError("every item must have a score for every model")

    totals = [sum(item[m] for item in items) for m in range(n_models)]
    total_var = variance(totals)
    if total_var == 0.0:
        return 0.0

    sum_item_var = sum(variance(row) for row in items)
    return (k / (k - 1)) * (1 - sum_item_var / total_var)


def standard_error_of_measurement(item_matrix: Sequence[Scores]) -> float:
    """SEM - how much of an observed score is plausibly noise.

        SEM = SD_total * sqrt(1 - alpha)

    Use it to decide whether two models actually differ. If model A scores 71
    and model B scores 73 with an SEM of 3, you have not measured a
    difference; you have measured the same thing twice.
    """
    items = [_validate(row, "item row") for row in item_matrix]
    n_models = len(items[0])
    totals = [sum(item[m] for item in items) for m in range(n_models)]
    sd = math.sqrt(variance(totals))
    alpha = cronbach_alpha(items)
    return sd * math.sqrt(max(0.0, 1.0 - alpha))


def backwards_p_value(item_scores: Scores, total_scores: Sequence[float]) -> float | None:
    """How surprising it would be for this item to look this backwards by chance.

    An item is "backwards" when the models that pass it are the ones that do
    *worse* on the rest of the suite. A raw negative correlation is not enough
    to conclude that: with a handful of models, or with an item almost everyone
    passes, negative correlations arise constantly from noise alone. Acting on
    them means chasing graders that were never broken.

    So we test it. Under the null hypothesis "this item is unrelated to ability",
    the k models that passed it are just a uniformly random subset of the n
    models. The correlation with the rest-score is a strictly increasing
    function of

        S = sum of the rest-scores of the models that passed,

    so P(correlation this low or lower) is exactly P(S <= observed S) - the
    lower tail of a sum drawn without replacement from the rest-scores. That
    quantity does not depend on the item at all, which is what makes this a
    genuine test rather than another threshold.

    Note that the rest-score is used deliberately: it excludes the item itself,
    so permuting the item cannot change what it is being correlated against.
    The test is exact.

    Returns the one-tailed p-value, or ``None`` when the item has no variance
    (everyone passed or everyone failed - there is nothing to test).

    The p-value is computed exactly by counting subsets when that is cheap, and
    by a normal approximation to the sampling distribution otherwise. See
    :func:`backwards_p_method` if you need to know which was used.
    """
    s = _validate(item_scores, "item_scores")
    totals = [float(t) for t in total_scores]
    if len(s) != len(totals):
        raise ValueError("item_scores and total_scores must be the same length")

    n = len(s)
    k = sum(s)
    if k == 0 or k == n:
        return None

    rest = [t - si for t, si in zip(totals, s)]
    observed = sum(r for r, si in zip(rest, s) if si == 1)

    mean_rest = sum(rest) / n
    var_rest = sum((r - mean_rest) ** 2 for r in rest) / n
    if var_rest == 0.0:
        # Every model scores the same on the rest of the suite: the item cannot
        # be backwards with respect to an ability ordering that does not exist.
        return None

    exact = _exact_lower_tail(rest, k, observed)
    if exact is not None:
        return exact

    mu = k * mean_rest
    sd = math.sqrt(k * (n - k) / (n - 1) * var_rest)
    z = (observed + 0.5 - mu) / sd
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def backwards_p_method(item_scores: Scores, total_scores: Sequence[float]) -> str:
    """``"exact"``, ``"normal"`` or ``"undefined"`` - how the p-value was reached."""
    s = _validate(item_scores, "item_scores")
    totals = [float(t) for t in total_scores]
    n, k = len(s), sum(s)
    if k == 0 or k == n:
        return "undefined"
    rest = [t - si for t, si in zip(totals, s)]
    mean_rest = sum(rest) / n
    if sum((r - mean_rest) ** 2 for r in rest) == 0.0:
        return "undefined"
    return "exact" if _exact_is_affordable(rest, k) else "normal"


# Budget on the dynamic-programming table below. Comfortably fast in pure
# Python for the cohort sizes item analysis is actually run on, and it degrades
# to a normal approximation rather than to a hang.
_EXACT_BUDGET = 5_000_000


def _exact_is_affordable(rest: Sequence[float], k: int) -> bool:
    ints = [r for r in rest if float(r).is_integer()]
    if len(ints) != len(rest):
        return False
    lo, hi = min(rest), max(rest)
    span = int(round(k * (hi - lo)))
    return len(rest) * k * (span + 1) <= _EXACT_BUDGET


def _exact_lower_tail(rest: Sequence[float], k: int, observed: float) -> float | None:
    """P(S <= observed) for S a sum of k of the rest-scores, drawn without
    replacement. Exact subset counting; ``None`` when that is too expensive."""
    if not _exact_is_affordable(rest, k):
        return None

    lo = min(rest)
    values = [int(round(r - lo)) for r in rest]
    target = int(math.floor(observed - k * lo + 1e-9))
    span = sum(sorted(values, reverse=True)[:k])
    if target < 0:
        return 0.0
    if target >= span:
        return 1.0

    # ways[j][s] = number of j-subsets summing to s
    ways = [[0] * (span + 1) for _ in range(k + 1)]
    ways[0][0] = 1
    for v in values:
        for j in range(k, 0, -1):
            row, prev = ways[j], ways[j - 1]
            for s_ in range(span, v - 1, -1):
                if prev[s_ - v]:
                    row[s_] += prev[s_ - v]

    total = sum(ways[k])
    if total == 0:
        return None
    return sum(ways[k][: target + 1]) / total


def benjamini_hochberg(p_values: Sequence[float | None], q: float = 0.05) -> list[bool]:
    """Which p-values survive a false-discovery-rate correction at level ``q``.

    A suite has hundreds of items. Testing every one of them at p < 0.05 and
    reporting the hits would manufacture roughly 5% of the suite as findings.
    Benjamini-Hochberg controls the expected share of *false* findings among
    those reported instead, which is the guarantee that matters when the output
    is a list of items for a human to go and investigate.

    ``None`` entries (items with no variance) are never selected.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be between 0 and 1, got {q}")
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    out = [False] * len(p_values)
    if not indexed:
        return out
    indexed.sort()
    m = len(indexed)
    cutoff = 0
    for rank, (p, _) in enumerate(indexed, 1):
        if p <= q * rank / m:
            cutoff = rank
    for rank, (_, i) in enumerate(indexed, 1):
        if rank <= cutoff:
            out[i] = True
    return out
