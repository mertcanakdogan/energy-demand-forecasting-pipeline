"""Forecast error metrics with explicit undefined cases."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _paired(
    actual: Sequence[float], predicted: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted values must have the same length")
    mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
    return actual_array[mask], predicted_array[mask]


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return mean absolute error, or NaN when no valid pairs exist."""

    actual_array, predicted_array = _paired(actual, predicted)
    if actual_array.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual_array - predicted_array)))


def wape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return weighted absolute percentage error as a decimal ratio.

    WAPE is undefined when the sum of absolute actual values is zero.
    """

    actual_array, predicted_array = _paired(actual, predicted)
    denominator = np.abs(actual_array).sum()
    if actual_array.size == 0 or denominator == 0:
        return float("nan")
    return float(np.abs(actual_array - predicted_array).sum() / denominator)
