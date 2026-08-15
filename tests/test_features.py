from __future__ import annotations

from energy_forecasting.features import add_calendar_features


def test_calendar_features_are_derived_without_mutating_input(hourly_frame) -> None:
    original_columns = hourly_frame.columns.tolist()
    featured = add_calendar_features(hourly_frame)
    assert hourly_frame.columns.tolist() == original_columns
    assert {"hour", "day_of_week", "month", "hour_sin", "hour_cos"} <= set(
        featured.columns
    )
    assert featured["hour"].between(0, 23).all()
