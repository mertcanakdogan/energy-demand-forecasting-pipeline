"""Automatic per-meter forecast model selection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def select_best_models(
    backtest_results: pd.DataFrame,
    meter_ids: Iterable[str],
    *,
    fallback_model: str = "seasonal_naive",
    min_valid_folds: int = 1,
) -> pd.DataFrame:
    """Select lowest mean valid WAPE for each meter, with a safe fallback."""

    selected: list[dict[str, object]] = []
    for meter_id in sorted(set(meter_ids)):
        if backtest_results.empty:
            candidates = pd.DataFrame()
        else:
            meter = backtest_results.loc[
                (backtest_results["meter_id"] == meter_id)
                & (backtest_results["status"] == "ok")
                & np.isfinite(backtest_results["wape"])
            ]
            candidates = meter.groupby("model", as_index=False).agg(
                wape=("wape", "mean"),
                mae=("mae", "mean"),
                number_of_folds=("wape", "count"),
            )
            candidates = candidates.loc[
                candidates["number_of_folds"] >= min_valid_folds
            ]

        if candidates.empty:
            selected.append(
                {
                    "meter_id": meter_id,
                    "selected_model": fallback_model,
                    "wape": np.nan,
                    "mae": np.nan,
                    "number_of_folds": 0,
                    "selection_reason": "fallback",
                }
            )
            continue
        best = candidates.sort_values(["wape", "mae", "model"], kind="stable").iloc[0]
        selected.append(
            {
                "meter_id": meter_id,
                "selected_model": best["model"],
                "wape": float(best["wape"]),
                "mae": float(best["mae"]),
                "number_of_folds": int(best["number_of_folds"]),
                "selection_reason": "lowest_mean_wape",
            }
        )
    return pd.DataFrame(selected)
