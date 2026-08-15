from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.anomaly_detection import AnomalyConfig, detect_anomalies


def anomaly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "meter_id": ["M1"] * 5,
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
            "actual_kwh": [10.0, 20.0, 2.0, np.nan, 10.0],
            "expected_kwh": [10.0] * 5,
        }
    )


def test_detector_labels_spikes_drops_and_missing() -> None:
    result = detect_anomalies(anomaly_frame(), AnomalyConfig(0.5, -0.5, 1.5))
    assert set(result["anomaly_type"]) == {
        "positive_spike",
        "negative_drop",
        "missing_reading",
    }


def test_detector_excludes_ordinary_observations() -> None:
    result = detect_anomalies(anomaly_frame(), AnomalyConfig(0.5, -0.5, 1.5))
    assert 0 not in result.index
    assert 4 not in result.index


def test_detector_handles_zero_expected_values() -> None:
    frame = anomaly_frame().iloc[[0]].copy()
    frame["expected_kwh"] = 0.0
    result = detect_anomalies(frame)
    assert len(result) == 1
    assert np.isfinite(result.iloc[0]["deviation_pct"])


def test_detector_assigns_configurable_severity() -> None:
    result = detect_anomalies(anomaly_frame(), AnomalyConfig(0.5, -0.5, 0.8))
    spike = result.loc[result["anomaly_type"] == "positive_spike"].iloc[0]
    assert spike["severity"] == "high"
