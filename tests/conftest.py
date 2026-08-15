from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=24 * 35, freq="h")
    hour = index.hour.to_numpy()
    values = 10 + 2 * np.sin(2 * np.pi * hour / 24)
    return pd.DataFrame(
        {
            "meter_id": "M0001",
            "group_id": "G01",
            "timestamp": index,
            "consumption_kwh": values,
            "temperature_c": 18 + 5 * np.sin(2 * np.pi * hour / 24),
            "is_weekend": index.dayofweek >= 5,
            "is_holiday": False,
        }
    )


@pytest.fixture
def two_meter_frame(hourly_frame: pd.DataFrame) -> pd.DataFrame:
    second = hourly_frame.copy()
    second["meter_id"] = "M0002"
    second["group_id"] = "G02"
    second["consumption_kwh"] *= 1.5
    return pd.concat([hourly_frame, second], ignore_index=True)
