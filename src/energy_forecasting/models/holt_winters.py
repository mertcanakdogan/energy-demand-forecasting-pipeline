"""Holt-Winters exponential-smoothing adapter."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWinters:
    """Additive daily-seasonal exponential smoothing."""

    name = "holt_winters"

    def __init__(self, seasonal_period: int = 24, max_history: int = 1344) -> None:
        self.seasonal_period = seasonal_period
        self.max_history = max_history

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        clean = (
            pd.Series(train, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .interpolate()
            .dropna()
        )
        if len(clean) < self.seasonal_period * 2:
            from .seasonal_naive import SeasonalNaive

            return SeasonalNaive(self.seasonal_period).forecast(clean, horizon)
        clean = clean.iloc[-self.max_history :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(
                clean.to_numpy(),
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=self.seasonal_period,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=True)
        return np.clip(np.asarray(fitted.forecast(horizon), dtype=float), 0, None)
