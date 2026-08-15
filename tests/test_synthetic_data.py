from __future__ import annotations

import pandas as pd

from energy_forecasting.synthetic_data import (
    SyntheticDataConfig,
    generate_synthetic_data,
)


def small_config(seed: int = 7) -> SyntheticDataConfig:
    return SyntheticDataConfig(
        n_meters=4,
        n_groups=2,
        start="2025-01-01",
        months=2,
        seed=seed,
        missing_rate=0.002,
        duplicate_rate=0.002,
        anomaly_rate=0.003,
    )


def test_generation_is_deterministic_for_same_seed() -> None:
    pd.testing.assert_frame_equal(
        generate_synthetic_data(small_config()),
        generate_synthetic_data(small_config()),
    )


def test_generation_changes_with_seed() -> None:
    left = generate_synthetic_data(small_config(7))
    right = generate_synthetic_data(small_config(8))
    assert not left["consumption_kwh"].equals(right["consumption_kwh"])


def test_generation_has_required_schema_and_cardinality() -> None:
    data = generate_synthetic_data(small_config())
    assert list(data.columns) == [
        "meter_id",
        "group_id",
        "timestamp",
        "consumption_kwh",
        "temperature_c",
        "is_weekend",
        "is_holiday",
    ]
    assert data["meter_id"].nunique() == 4
    assert data["group_id"].nunique() == 2


def test_generation_injects_quality_events() -> None:
    data = generate_synthetic_data(small_config())
    assert data.duplicated(["meter_id", "timestamp"]).any()
    full_hours = pd.date_range(
        data["timestamp"].min(), data["timestamp"].max(), freq="h"
    )
    first_meter = data.loc[data["meter_id"] == "M0001", "timestamp"]
    assert len(set(full_hours) - set(first_meter)) > 0
    median = data["consumption_kwh"].median()
    assert data["consumption_kwh"].max() > median * 3


def test_generation_rejects_more_groups_than_meters() -> None:
    config = SyntheticDataConfig(n_meters=2, n_groups=3, months=1)
    try:
        generate_synthetic_data(config)
    except ValueError as exc:
        assert "groups" in str(exc).lower()
    else:
        raise AssertionError("Expected invalid cardinality to fail")
