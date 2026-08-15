"""Compact SARIMAX forecasting adapter."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsmodelsSARIMAX


class SARIMAX:
    """AR(1) SARIMAX with daily and weekly Fourier regressors.

    Fourier regressors keep the demonstration fast and interpretable while the
    SARIMAX error process captures short-term autocorrelation.
    """

    name = "sarimax"

    def __init__(self, max_history: int = 672, max_iterations: int = 15) -> None:
        self.max_history = max_history
        self.max_iterations = max_iterations

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        clean = (
            pd.Series(train, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .interpolate()
            .dropna()
        )
        if len(clean) < 72:
            from .seasonal_naive import SeasonalNaive

            return SeasonalNaive(24).forecast(clean, horizon)
        clean = clean.iloc[-self.max_history :]
        if isinstance(clean.index, pd.DatetimeIndex):
            train_hour = clean.index.hour.to_numpy()
            train_week_hour = (clean.index.dayofweek * 24 + clean.index.hour).to_numpy()
            future_index = pd.date_range(
                clean.index[-1] + pd.Timedelta("1h"), periods=horizon, freq="h"
            )
            future_hour = future_index.hour.to_numpy()
            future_week_hour = (
                future_index.dayofweek * 24 + future_index.hour
            ).to_numpy()
        else:
            positions = np.arange(len(clean))
            train_hour = positions % 24
            train_week_hour = positions % 168
            future_positions = np.arange(len(clean), len(clean) + horizon)
            future_hour = future_positions % 24
            future_week_hour = future_positions % 168

        def fourier(hour: np.ndarray, week_hour: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * week_hour / 168),
                    np.cos(2 * np.pi * week_hour / 168),
                ]
            )

        train_exog = fourier(train_hour, train_week_hour)
        future_exog = fourier(future_hour, future_week_hour)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = StatsmodelsSARIMAX(
                clean.to_numpy(),
                order=(1, 0, 0),
                seasonal_order=(0, 0, 0, 0),
                exog=train_exog,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=self.max_iterations)
        return np.clip(
            np.asarray(fitted.forecast(horizon, exog=future_exog), dtype=float), 0, None
        )
