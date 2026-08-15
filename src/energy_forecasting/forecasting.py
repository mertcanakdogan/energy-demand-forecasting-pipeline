"""Next-horizon forecasting and portfolio aggregation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from .models import MODEL_FACTORIES, ForecastModel, SeasonalNaive

LOGGER = logging.getLogger(__name__)


def generate_forecasts(
    data: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    horizon: int = 24,
    model_factories: Mapping[str, Callable[[], ForecastModel]] | None = None,
    max_train_hours: int | None = 1344,
) -> pd.DataFrame:
    """Fit each selected model and produce aligned future meter forecasts."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    factories = model_factories or MODEL_FACTORIES
    common_end = pd.to_datetime(data["timestamp"]).max()
    future_index = pd.date_range(
        common_end + pd.Timedelta("1h"), periods=horizon, freq="h"
    )
    choice = selections.set_index("meter_id")["selected_model"].to_dict()
    records: list[pd.DataFrame] = []

    for meter_id, meter in data.groupby("meter_id", sort=True):
        meter = meter.sort_values("timestamp")
        series = meter.set_index("timestamp")["consumption_kwh"]
        if max_train_hours is not None:
            series = series.iloc[-max_train_hours:]
        selected_name = str(choice.get(meter_id, "seasonal_naive"))
        used_name = selected_name
        try:
            factory = factories[selected_name]
            prediction = np.asarray(factory().forecast(series, horizon), dtype=float)
        except Exception as exc:  # noqa: BLE001 - final model fits have a documented fallback
            LOGGER.warning(
                "Final fit failed for %s/%s (%s); using seasonal naive",
                meter_id,
                selected_name,
                exc,
            )
            prediction = SeasonalNaive().forecast(series, horizon)
            used_name = "seasonal_naive"
        records.append(
            pd.DataFrame(
                {
                    "meter_id": meter_id,
                    "group_id": str(meter["group_id"].iloc[-1]),
                    "timestamp": future_index,
                    "forecast_kwh": np.clip(prediction, 0, None),
                    "selected_model": used_name,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def aggregate_forecasts(
    meter_forecast: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return group-level and total portfolio forecasts."""

    group = (
        meter_forecast.groupby(["group_id", "timestamp"], as_index=False)[
            "forecast_kwh"
        ]
        .sum()
        .sort_values(["group_id", "timestamp"])
        .reset_index(drop=True)
    )
    portfolio = (
        meter_forecast.groupby("timestamp", as_index=False)["forecast_kwh"]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return group, portfolio
