from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecasting.backtesting import BacktestConfig, rolling_origin_backtest


class RecordingModel:
    name = "recording"

    def __init__(self) -> None:
        self.max_train_values: list[float] = []

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        self.max_train_values.append(float(train.max()))
        return np.repeat(float(train.iloc[-1]), horizon)


def test_backtest_builds_requested_folds_without_leakage() -> None:
    series = pd.Series(
        np.arange(240, dtype=float) + 1,
        index=pd.date_range("2025-01-01", periods=240, freq="h"),
    )
    model = RecordingModel()
    result = rolling_origin_backtest(
        series,
        {"recording": lambda: model},
        BacktestConfig(
            horizon=24, folds=3, step_hours=24, min_train_hours=72, max_train_hours=120
        ),
    )
    assert len(result) == 3
    assert (result["train_end"] < result["validation_start"]).all()
    assert model.max_train_values == [168.0, 192.0, 216.0]


def test_backtest_marks_insufficient_history() -> None:
    series = pd.Series(
        [1.0] * 20, index=pd.date_range("2025-01-01", periods=20, freq="h")
    )
    result = rolling_origin_backtest(
        series,
        {"recording": RecordingModel},
        BacktestConfig(horizon=12, folds=2, min_train_hours=24),
    )
    assert set(result["status"]) == {"insufficient_history"}


def test_backtest_isolates_model_failure(hourly_frame) -> None:
    class BrokenModel:
        def forecast(self, train, horizon):
            raise RuntimeError("fit failed")

    series = hourly_frame.set_index("timestamp")["consumption_kwh"]
    result = rolling_origin_backtest(
        series,
        {"broken": BrokenModel},
        BacktestConfig(horizon=24, folds=1, min_train_hours=168),
    )
    assert result.loc[0, "status"] == "failed"
    assert "fit failed" in result.loc[0, "error"]
