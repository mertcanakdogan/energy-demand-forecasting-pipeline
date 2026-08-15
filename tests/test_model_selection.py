from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.model_selection import select_best_models


def test_selection_uses_lowest_average_valid_wape() -> None:
    results = pd.DataFrame(
        {
            "meter_id": ["M1", "M1", "M1", "M1"],
            "model": ["a", "a", "b", "b"],
            "wape": [0.2, 0.4, 0.1, 0.2],
            "mae": [2.0, 4.0, 1.0, 2.0],
            "status": ["ok"] * 4,
        }
    )
    selected = select_best_models(results, ["M1"])
    assert selected.loc[0, "selected_model"] == "b"
    assert selected.loc[0, "number_of_folds"] == 2


def test_selection_ignores_invalid_metrics_and_falls_back() -> None:
    results = pd.DataFrame(
        {
            "meter_id": ["M1"],
            "model": ["sarimax"],
            "wape": [np.nan],
            "mae": [np.nan],
            "status": ["failed"],
        }
    )
    selected = select_best_models(
        results, ["M1", "M2"], fallback_model="seasonal_naive"
    )
    assert set(selected["selected_model"]) == {"seasonal_naive"}
    assert (selected["selection_reason"] == "fallback").all()


def test_selection_requires_minimum_valid_folds() -> None:
    results = pd.DataFrame(
        {
            "meter_id": ["M1"],
            "model": ["a"],
            "wape": [0.1],
            "mae": [1.0],
            "status": ["ok"],
        }
    )
    selected = select_best_models(results, ["M1"], min_valid_folds=2)
    assert selected.loc[0, "selected_model"] == "seasonal_naive"
