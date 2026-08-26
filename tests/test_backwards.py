"""Backwards-item detection tests, checked against values computed by hand.

The backwards test asks one question: if this item had nothing to do with model
ability, how often would the models that passed it score this badly on the rest
of the suite? Because the rest-score excludes the item itself, permuting the
item cannot move what it is correlated against, and the null distribution is
just "which k of the n models passed" - so the p-value can be counted exactly.

Reference case used below (4 models, item scores x, totals T):

    T    = [4, 3, 2, 1]
    x    = [0, 0, 1, 1]          the two weakest models pass
    rest = T - x = [4, 3, 1, 0]

    k = 2, so there are C(4,2) = 6 equally likely subsets, with sums

        {4,3}=7  {4,1}=5  {4,0}=4  {3,1}=4  {3,0}=3  {1,0}=1

    The observed subset is {1, 0}, sum 1 - the smallest of the six.

        p = P(S <= 1) = 1/6 = 0.1666...

    Which is the whole point: this is the most backwards an item can possibly
    look with four models, and it still is not significant at 0.05.
"""

import math

import pytest

from itemwise import RunResult, Suite, analyze
from itemwise.stats import (
    backwards_p_method,
    backwards_p_value,
    benjamini_hochberg,
)

TOTALS = [4.0, 3.0, 2.0, 1.0]


# ------------------------------------------------------------- the p-value

def test_most_backwards_possible_item_matches_hand_computation():
    assert backwards_p_value([0, 0, 1, 1], TOTALS) == pytest.approx(1 / 6)


def test_most_forward_possible_item_is_the_upper_tail():
    """x = [1,1,0,0] takes the largest of the six subsets, so P(S <= obs) = 1."""
    assert backwards_p_value([1, 1, 0, 0], TOTALS) == pytest.approx(1.0)


def test_middling_item_matches_hand_computation():
    """x = [1,0,0,1]: rest = [3,3,2,0], k=2, observed {3,0} = 3.

    Subsets: {3,3}=6 {3,2}=5 {3,0}=3 {3,2}=5 {3,0}=3 {2,0}=2
    Sums <= 3: {3,0}, {3,0}, {2,0} -> 3 of 6.
    """
    assert backwards_p_value([1, 0, 0, 1], TOTALS) == pytest.approx(3 / 6)


def test_p_value_is_undefined_for_items_with_no_variance():
    assert backwards_p_value([1, 1, 1, 1], TOTALS) is None
    assert backwards_p_value([0, 0, 0, 0], TOTALS) is None


def test_p_value_is_undefined_when_every_model_is_tied():
    """rest = T - x = [2,2,2,2]: no ability ordering exists once the item is
    removed, so 'backwards' has nothing to be backwards with respect to."""
    assert backwards_p_value([1, 0, 1, 0], [3.0, 2.0, 3.0, 2.0]) is None


def test_p_value_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        backwards_p_value([1, 0], TOTALS)


def test_small_cohorts_are_counted_exactly():
    assert backwards_p_method([0, 0, 1, 1], TOTALS) == "exact"
    assert backwards_p_method([1, 1, 1, 1], TOTALS) == "undefined"


def test_floor_on_the_p_value_is_one_over_n_choose_k():
    """The smallest reachable p-value is 1/C(n,k) - the observed split being the
    single most extreme of all of them. This is why tiny cohorts cannot convict."""
    for n in (4, 6, 8):
        for k in range(1, n):
            totals = [float(n - i) for i in range(n)]
            x = [0] * (n - k) + [1] * k  # the k weakest models pass
            p = backwards_p_value(x, totals)
            assert p >= 1 / math.comb(n, k) - 1e-12


# ---------------------------------------------- multiple-comparison control

def test_benjamini_hochberg_matches_hand_computation():
    """p = [0.001, 0.008, 0.039, 0.041, 0.9], m = 5, q = 0.05.

    Thresholds q*rank/m: 0.01, 0.02, 0.03, 0.04, 0.05
      0.001 <= 0.01  yes
      0.008 <= 0.02  yes
      0.039 <= 0.03  no
      0.041 <= 0.04  no
      0.900 <= 0.05  no
    Largest passing rank is 2, so the two smallest are selected.
    """
    got = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.9], q=0.05)
    assert got == [True, True, False, False, False]


def test_benjamini_hochberg_uses_the_largest_passing_rank():
    """0.02 fails its own threshold (0.02 > 0.0166) but rank 3 passes, so it is
    still selected - the step-up property, which a naive filter would get wrong."""
    got = benjamini_hochberg([0.001, 0.02, 0.03], q=0.05)
    assert got == [True, True, True]


def test_benjamini_hochberg_never_selects_none():
    assert benjamini_hochberg([None, None]) == [False, False]


def test_benjamini_hochberg_rejects_bad_q():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01], q=1.5)


# -------------------------------------------------------------- calibration

def _random_suite(n_models: int, n_items: int, seed: int) -> RunResult:
    """A suite with no broken items at all: every item's pass set is drawn
    independently of ability, so every backwards flag would be a false one."""
    import random

    rng = random.Random(seed)
    items = [{"id": f"i{j:03d}"} for j in range(n_items)]
    scores = {f"m{i:02d}": [] for i in range(n_models)}
    for _ in range(n_items):
        p = rng.uniform(0.2, 0.9)
        for m in scores:
            scores[m].append(1 if rng.random() < p else 0)
    return RunResult(Suite.from_dicts(items), scores)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_no_false_backwards_findings_on_data_with_no_signal(seed):
    """The failure this test exists to prevent: a fixed negative threshold
    reported dozens of 'broken graders' on data like this."""
    report = analyze(_random_suite(n_models=20, n_items=200, seed=seed))
    assert report.backwards_items() == []


def test_a_planted_broken_item_is_still_found_among_the_noise():
    """Calibration is worthless without power. Same noisy suite, one genuinely
    inverted item added - it must survive the correction."""
    run = _random_suite(n_models=20, n_items=200, seed=11)
    totals = {m: sum(v) for m, v in run.scores.items()}
    order = sorted(totals, key=lambda m: -totals[m])

    items = [{"id": i.id} for i in run.suite.items] + [{"id": "planted"}]
    scores = {m: list(v) for m, v in run.scores.items()}
    for rank, m in enumerate(order):
        scores[m].append(1 if rank >= len(order) // 2 else 0)

    report = analyze(RunResult(Suite.from_dicts(items), scores))
    assert "planted" in {s.item.id for s in report.backwards_items()}


def test_detectability_warning_tracks_cohort_size():
    tiny = analyze(_random_suite(n_models=5, n_items=40, seed=3))
    assert tiny.backwards_detectable() is False
    big = analyze(_random_suite(n_models=20, n_items=40, seed=3))
    assert big.backwards_detectable() is True
