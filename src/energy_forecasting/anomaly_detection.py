"""Explainable threshold-based consumption anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnomalyConfig:
    """Generic deviation thresholds expressed as decimal ratios."""

    positive_threshold: float = 0.60
    negative_threshold: float = -0.60
    severe_threshold: float = 1.20
    denominator_floor: float = 0.01

    def __post_init__(self) -> None:
        if (
            self.positive_threshold <= 0
            or self.negative_threshold >= 0
            or self.severe_threshold <= 0
        ):
            raise ValueError(
                "Anomaly thresholds must straddle zero and severe_threshold must be positive"
            )


def detect_anomalies(
    data: pd.DataFrame, config: AnomalyConfig | None = None
) -> pd.DataFrame:
    """Flag missing readings and large signed deviations from expected demand."""

    config = config or AnomalyConfig()
    required = {"meter_id", "timestamp", "actual_kwh", "expected_kwh"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing anomaly columns: {sorted(missing)}")

    result = data.copy()
    denominator = np.maximum(
        result["expected_kwh"].abs().to_numpy(dtype=float), config.denominator_floor
    )
    ratio = (
        result["actual_kwh"].to_numpy(dtype=float)
        - result["expected_kwh"].to_numpy(dtype=float)
    ) / denominator
    result["deviation_pct"] = ratio * 100
    missing_reading = result["actual_kwh"].isna()
    positive = ~missing_reading & (ratio >= config.positive_threshold)
    negative = ~missing_reading & (ratio <= config.negative_threshold)
    result["anomaly_type"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[positive, "anomaly_type"] = "positive_spike"
    result.loc[negative, "anomaly_type"] = "negative_drop"
    result.loc[missing_reading, "anomaly_type"] = "missing_reading"
    result["severity"] = "medium"
    result.loc[np.abs(ratio) >= config.severe_threshold, "severity"] = "high"
    result.loc[missing_reading, "severity"] = "high"
    columns = [
        "meter_id",
        "timestamp",
        "actual_kwh",
        "expected_kwh",
        "deviation_pct",
        "anomaly_type",
        "severity",
    ]
    return result.loc[result["anomaly_type"].notna(), columns]


def weekly_expected_frame(
    prepared_data: pd.DataFrame, seasonal_period: int = 168
) -> pd.DataFrame:
    """Create actual/expected pairs using the previous week's corresponding hour."""

    frames: list[pd.DataFrame] = []
    for meter_id, meter in prepared_data.groupby("meter_id", sort=True):
        meter = meter.sort_values("timestamp").copy()
        actual = meter["consumption_kwh"].mask(meter.get("was_missing", False))
        frames.append(
            pd.DataFrame(
                {
                    "meter_id": meter_id,
                    "timestamp": meter["timestamp"],
                    "actual_kwh": actual,
                    "expected_kwh": meter["consumption_kwh"].shift(seasonal_period),
                },
                index=meter.index,
            )
        )
    return pd.concat(frames, ignore_index=True)
