from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.models import build_model


def test_seasonal_naive_repeats_previous_week() -> None:
    series = pd.Series(np.arange(200, dtype=float))
    forecast = build_model("seasonal_naive").forecast(series, 24)
    np.testing.assert_array_equal(forecast, np.arange(32, 56, dtype=float))


def test_seasonal_naive_short_history_uses_safe_level() -> None:
    forecast = build_model("seasonal_naive").forecast(pd.Series([2.0, 4.0]), 3)
    np.testing.assert_array_equal(forecast, [4.0, 4.0, 4.0])


@pytest.mark.parametrize(
    "name", ["seasonal_naive", "weighted_baseline", "holt_winters", "sarimax"]
)
def test_each_model_returns_finite_non_negative_horizon(
    name: str, hourly_frame
) -> None:
    series = hourly_frame.set_index("timestamp")["consumption_kwh"]
    forecast = build_model(name).forecast(series, 12)
    assert forecast.shape == (12,)
    assert np.isfinite(forecast).all()
    assert (forecast >= 0).all()


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("mystery")
