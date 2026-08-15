"""Calendar feature engineering for analysis and future model extensions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with readable and cyclical calendar features."""

    featured = data.copy()
    timestamp = pd.to_datetime(featured["timestamp"])
    featured["hour"] = timestamp.dt.hour
    featured["day_of_week"] = timestamp.dt.dayofweek
    featured["month"] = timestamp.dt.month
    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24)
    featured["week_sin"] = np.sin(2 * np.pi * featured["day_of_week"] / 7)
    featured["week_cos"] = np.cos(2 * np.pi * featured["day_of_week"] / 7)
    return featured
