"""Forecast model registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd

from .holt_winters import HoltWinters
from .sarimax import SARIMAX
from .seasonal_naive import SeasonalNaive
from .weighted_baseline import WeightedBaseline


class ForecastModel(Protocol):
    """Minimal interface consumed by backtesting and forecasting."""

    name: str

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray: ...


MODEL_FACTORIES: dict[str, Callable[[], ForecastModel]] = {
    "seasonal_naive": SeasonalNaive,
    "weighted_baseline": WeightedBaseline,
    "holt_winters": HoltWinters,
    "sarimax": SARIMAX,
}


def build_model(name: str) -> ForecastModel:
    """Construct a model by its stable public name."""

    try:
        return MODEL_FACTORIES[name]()
    except KeyError as exc:
        available = ", ".join(MODEL_FACTORIES)
        raise ValueError(
            f"Unknown model {name!r}. Available models: {available}"
        ) from exc


def selected_factories(
    names: tuple[str, ...] | list[str],
) -> dict[str, Callable[[], ForecastModel]]:
    """Return validated factories for an ordered collection of names."""

    unknown = set(names) - set(MODEL_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    return {name: MODEL_FACTORIES[name] for name in names}


__all__ = [
    "SARIMAX",
    "ForecastModel",
    "HoltWinters",
    "SeasonalNaive",
    "WeightedBaseline",
    "build_model",
    "selected_factories",
]
