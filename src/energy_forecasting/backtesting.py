"""Leakage-safe rolling-origin model evaluation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import mae, wape
from .models import ForecastModel

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    """Controls expanding-origin folds and bounded fit history."""

    horizon: int = 24
    folds: int = 2
    step_hours: int = 168
    min_train_hours: int = 672
    max_train_hours: int | None = 1344

    def __post_init__(self) -> None:
        if min(self.horizon, self.folds, self.step_hours, self.min_train_hours) < 1:
            raise ValueError("Backtest settings must be positive")
        if (
            self.max_train_hours is not None
            and self.max_train_hours < self.min_train_hours
        ):
            raise ValueError("max_train_hours cannot be shorter than min_train_hours")


def rolling_origin_backtest(
    series: pd.Series,
    model_factories: Mapping[str, Callable[[], ForecastModel]],
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Evaluate model factories on chronological rolling origins.

    Training slices always end before validation begins. Failures are returned
    as records so one model cannot abort an entire portfolio run.
    """

    config = config or BacktestConfig()
    series = pd.Series(series, dtype=float).sort_index()
    records: list[dict[str, object]] = []
    first_origin = len(series) - config.horizon - config.step_hours * (config.folds - 1)

    for fold_offset in range(config.folds):
        origin = first_origin + fold_offset * config.step_hours
        validation = series.iloc[max(origin, 0) : max(origin, 0) + config.horizon]
        train_start = 0
        if config.max_train_hours is not None:
            train_start = max(0, origin - config.max_train_hours)
        train = series.iloc[train_start : max(origin, 0)]

        for model_name, factory in model_factories.items():
            base: dict[str, object] = {
                "model": model_name,
                "fold": fold_offset + 1,
                "train_start": train.index.min() if not train.empty else pd.NaT,
                "train_end": train.index.max() if not train.empty else pd.NaT,
                "validation_start": validation.index.min()
                if not validation.empty
                else pd.NaT,
                "validation_end": validation.index.max()
                if not validation.empty
                else pd.NaT,
                "n_train": len(train),
                "n_valid": int(validation.notna().sum()),
                "wape": np.nan,
                "mae": np.nan,
                "status": "insufficient_history",
                "error": "",
            }
            if origin < config.min_train_hours or len(validation) < config.horizon:
                records.append(base)
                continue
            try:
                prediction = np.asarray(
                    factory().forecast(train, config.horizon), dtype=float
                )
                if prediction.shape != (config.horizon,):
                    raise ValueError(
                        f"expected {config.horizon} forecasts, received {prediction.shape}"
                    )
                fold_wape = wape(validation.to_numpy(), prediction)
                fold_mae = mae(validation.to_numpy(), prediction)
                base.update(
                    {
                        "wape": fold_wape,
                        "mae": fold_mae,
                        "status": "ok"
                        if np.isfinite(fold_wape) and np.isfinite(fold_mae)
                        else "invalid_metric",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - isolate diverse numerical model failures
                base.update(
                    {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                )
            records.append(base)

    return pd.DataFrame.from_records(records)


def backtest_portfolio(
    data: pd.DataFrame,
    model_factories: Mapping[str, Callable[[], ForecastModel]],
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run rolling-origin evaluation independently for every meter."""

    config = config or BacktestConfig()
    results: list[pd.DataFrame] = []
    grouped = data.sort_values("timestamp").groupby("meter_id", sort=True)
    total = data["meter_id"].nunique()
    for position, (meter_id, meter) in enumerate(grouped, start=1):
        series = meter.set_index("timestamp")["consumption_kwh"]
        result = rolling_origin_backtest(series, model_factories, config)
        result.insert(0, "group_id", str(meter["group_id"].iloc[0]))
        result.insert(0, "meter_id", meter_id)
        results.append(result)
        if position == 1 or position % 10 == 0 or position == total:
            LOGGER.info("Backtested %d/%d meters", position, total)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
