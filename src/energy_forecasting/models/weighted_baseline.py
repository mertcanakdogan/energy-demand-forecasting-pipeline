"""Exponentially weighted recent-profile baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


class WeightedBaseline:
    """Repeat a recent daily profile scaled to an exponentially weighted level."""

    name = "weighted_baseline"

    def __init__(self, alpha: float = 0.15, profile_hours: int = 24) -> None:
        self.alpha = alpha
        self.profile_hours = profile_hours

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        clean = (
            pd.Series(train, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        )
        if clean.empty:
            raise ValueError("training data has no finite values")
        if len(clean) < self.profile_hours:
            return np.repeat(
                max(0.0, float(clean.ewm(alpha=self.alpha).mean().iloc[-1])), horizon
            )
        profile = clean.iloc[-self.profile_hours :].to_numpy()
        recent = clean.iloc[-min(len(clean), 168) :]
        target_level = float(recent.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1])
        profile_level = float(np.mean(profile))
        scale = target_level / profile_level if profile_level > 0 else 1.0
        return np.clip(np.resize(profile * scale, horizon), 0, None)
