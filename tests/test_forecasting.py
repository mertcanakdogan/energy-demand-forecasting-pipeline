from __future__ import annotations

import pandas as pd

from energy_forecasting.forecasting import aggregate_forecasts, generate_forecasts


def test_generate_forecasts_has_24_future_hours_per_meter(two_meter_frame) -> None:
    selection = pd.DataFrame(
        {
            "meter_id": ["M0001", "M0002"],
            "selected_model": ["seasonal_naive", "weighted_baseline"],
        }
    )
    result = generate_forecasts(two_meter_frame, selection, horizon=24)
    assert result.groupby("meter_id").size().eq(24).all()
    assert result["timestamp"].min() == two_meter_frame[
        "timestamp"
    ].max() + pd.Timedelta("1h")
    assert (result["forecast_kwh"] >= 0).all()


def test_forecast_aggregations_reconcile(two_meter_frame) -> None:
    selection = pd.DataFrame(
        {
            "meter_id": ["M0001", "M0002"],
            "selected_model": ["seasonal_naive", "seasonal_naive"],
        }
    )
    meter = generate_forecasts(two_meter_frame, selection, horizon=24)
    group, portfolio = aggregate_forecasts(meter)
    by_hour = meter.groupby("timestamp")["forecast_kwh"].sum()
    pd.testing.assert_series_equal(
        portfolio.set_index("timestamp")["forecast_kwh"], by_hour, check_names=False
    )
    assert len(group) == 48
