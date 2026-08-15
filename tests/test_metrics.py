from __future__ import annotations

import math

import numpy as np

from energy_forecasting.metrics import mae, wape


def test_mae_matches_hand_calculation() -> None:
    assert mae([1, 2, 3], [2, 2, 5]) == 1.0


def test_wape_matches_hand_calculation() -> None:
    assert wape([10, 20], [12, 16]) == 0.2


def test_metrics_ignore_unpaired_missing_values() -> None:
    actual = [1.0, np.nan, 3.0]
    predicted = [2.0, 10.0, 3.0]
    assert mae(actual, predicted) == 0.5
    assert wape(actual, predicted) == 0.25


def test_wape_is_nan_when_actual_sum_is_zero() -> None:
    assert math.isnan(wape([0, 0], [1, 2]))


def test_metrics_are_nan_without_valid_pairs() -> None:
    assert math.isnan(mae([np.nan], [1]))
    assert math.isnan(wape([np.nan], [1]))


def test_metrics_reject_different_lengths() -> None:
    try:
        mae([1], [1, 2])
    except ValueError as exc:
        assert "length" in str(exc).lower()
    else:
        raise AssertionError("Expected mismatched inputs to fail")
