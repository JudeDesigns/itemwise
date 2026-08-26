"""Statistics tests, checked against values computed by hand.

The reference matrix used throughout (4 models, 5 items):

              A  B  C  D
    item1     1  1  1  1     p = 1.00  (dead: everyone passes)
    item2     0  0  0  0     p = 0.00  (dead: nobody passes)
    item3     1  1  0  0     p = 0.50
    item4     1  0  0  0     p = 0.25
    item5     1  1  1  0     p = 0.75
    -------------------
    total     4  3  2  1
"""

import math

import pytest

from itemwise.stats import (
    cronbach_alpha,
    difficulty,
    discrimination_index,
    point_biserial,
    standard_error_of_measurement,
    variance,
)

MATRIX = [
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 0],
]
TOTALS = [4.0, 3.0, 2.0, 1.0]


# ---------------------------------------------------------------- difficulty

def test_difficulty_matches_hand_computation():
    assert difficulty(MATRIX[0]) == 1.00
    assert difficulty(MATRIX[1]) == 0.00
    assert difficulty(MATRIX[2]) == 0.50
    assert difficulty(MATRIX[3]) == 0.25
    assert difficulty(MATRIX[4]) == 0.75


def test_difficulty_rejects_empty_and_non_binary():
    with pytest.raises(ValueError, match="empty"):
        difficulty([])
    with pytest.raises(ValueError, match="only 0 or 1"):
        difficulty([0, 1, 2])


# ------------------------------------------------------------------ variance

def test_variance_is_population_not_sample():
    # By hand: mean 2.5, deviations 1.5/0.5/0.5/1.5, squares 2.25/0.25/0.25/2.25
    # Population (N=4): 5.0/4 = 1.25.  Sample (N-1=3) would be 1.667.
    assert variance([4, 3, 2, 1]) == pytest.approx(1.25)


def test_variance_of_constant_is_zero():
    assert variance([2, 2, 2]) == 0.0


# ------------------------------------------------------------ point-biserial

def test_point_biserial_item3_matches_hand_computation():
    # corrected totals = [3, 2, 2, 1]; mean 2.0; var 0.5; sd sqrt(0.5)
    # M1 (A,B) = 2.5, M0 (C,D) = 1.5, p = q = 0.5
    # r = (2.5 - 1.5)/sqrt(0.5) * sqrt(0.25) = 0.7071067811865475
    assert point_biserial(MATRIX[2], TOTALS) == pytest.approx(0.7071067811865475)


def test_point_biserial_item4_matches_hand_computation():
    # corrected totals = [3, 3, 2, 1]; mean 2.25; var 0.6875
    # M1 = 3.0, M0 = 2.0, p = 0.25
    # r = 1/sqrt(0.6875) * sqrt(0.1875) = 0.5222329678670935
    assert point_biserial(MATRIX[3], TOTALS) == pytest.approx(0.5222329678670935)


def test_point_biserial_item5_matches_hand_computation():
    # corrected totals = [3, 2, 1, 1]; mean 1.75; var 0.6875
    # M1 = 2.0, M0 = 1.0, p = 0.75
    assert point_biserial(MATRIX[4], TOTALS) == pytest.approx(0.5222329678670935)


def test_point_biserial_is_zero_for_items_with_no_variance():
    assert point_biserial(MATRIX[0], TOTALS) == 0.0   # everyone passes
    assert point_biserial(MATRIX[1], TOTALS) == 0.0   # nobody passes


def test_point_biserial_goes_negative_when_item_is_backwards():
    # Weak models pass, strong models fail. Uncorrected so the arithmetic
    # is directly checkable: M1 = 1.5, M0 = 3.5, sd = sqrt(1.25).
    r = point_biserial([0, 0, 1, 1], TOTALS, corrected=False)
    assert r == pytest.approx(-0.894427190999916)
    assert r < 0


def test_correction_lowers_the_correlation():
    """Uncorrected correlations are inflated by the item scoring itself."""
    raw = point_biserial(MATRIX[2], TOTALS, corrected=False)
    adj = point_biserial(MATRIX[2], TOTALS, corrected=True)
    assert raw > adj


def test_point_biserial_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        point_biserial([1, 0], [1.0, 2.0, 3.0])


# ------------------------------------------------------------ discrimination

def test_discrimination_index_matches_hand_computation():
    # n=4, group size = round(4 * 0.27) = 1, so top model vs bottom model.
    # item3: A (top) passed, D (bottom) failed -> D = 1 - 0 = 1.0
    assert discrimination_index(MATRIX[2], TOTALS) == pytest.approx(1.0)


def test_discrimination_index_is_zero_for_dead_items():
    assert discrimination_index(MATRIX[0], TOTALS) == 0.0
    assert discrimination_index(MATRIX[1], TOTALS) == 0.0


def test_discrimination_index_is_negative_when_backwards():
    assert discrimination_index([0, 0, 1, 1], TOTALS) == pytest.approx(-1.0)


def test_discrimination_index_rejects_bad_group_fraction():
    with pytest.raises(ValueError, match="group_fraction"):
        discrimination_index(MATRIX[2], TOTALS, group_fraction=0.9)


# ------------------------------------------------------------------- alpha

def test_cronbach_alpha_matches_hand_computation():
    # sum of item variances = 0 + 0 + 0.25 + 0.1875 + 0.1875 = 0.625
    # total variance = 1.25, k = 5
    # alpha = (5/4) * (1 - 0.625/1.25) = 1.25 * 0.5 = 0.625
    assert cronbach_alpha(MATRIX) == pytest.approx(0.625)


def test_cronbach_alpha_needs_at_least_two_items():
    with pytest.raises(ValueError, match="at least 2 items"):
        cronbach_alpha([[1, 0, 1]])


def test_cronbach_alpha_rejects_ragged_matrix():
    with pytest.raises(ValueError, match="every item"):
        cronbach_alpha([[1, 0, 1], [1, 0]])


def test_cronbach_alpha_is_zero_when_no_model_variance():
    # Every model scores the same total -> nothing to be reliable about.
    assert cronbach_alpha([[1, 0], [0, 1]]) == 0.0


# --------------------------------------------------------------------- SEM

def test_standard_error_of_measurement_matches_hand_computation():
    # sd_total = sqrt(1.25); alpha = 0.625
    # SEM = sqrt(1.25) * sqrt(0.375) = 0.6846531968814576
    expected = math.sqrt(1.25) * math.sqrt(0.375)
    assert standard_error_of_measurement(MATRIX) == pytest.approx(expected)
    assert standard_error_of_measurement(MATRIX) == pytest.approx(0.6846531968814576)
