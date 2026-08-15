"""Weekly seasonal-naive forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    """Forecast each hour from the corresponding hour one week earlier."""

    name = "seasonal_naive"

    def __init__(self, seasonal_period: int = 168) -> None:
        self.seasonal_period = seasonal_period

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        values = np.asarray(train, dtype=float)
        if horizon < 1:
            raise ValueError("horizon must be positive")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("training data has no finite values")
        if len(values) < self.seasonal_period:
            return np.repeat(max(0.0, float(finite[-1])), horizon)
        seasonal = values[-self.seasonal_period :]
        forecast = np.resize(seasonal, horizon)
        return np.clip(forecast, 0, None)
