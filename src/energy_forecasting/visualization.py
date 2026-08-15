"""Small, deterministic charts for the portfolio demonstration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CHART_STYLE = {
    "actual": "#334155",
    "forecast": "#0f766e",
    "accent": "#ea580c",
    "grid": "#cbd5e1",
}


def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def save_example_charts(
    prepared: pd.DataFrame,
    backtests: pd.DataFrame,
    portfolio_forecast: pd.DataFrame,
    anomaly_pairs: pd.DataFrame,
    anomalies: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write four recruiter-facing example charts and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    meter_id = str(prepared["meter_id"].min())
    meter = (
        prepared.loc[prepared["meter_id"] == meter_id].sort_values("timestamp").copy()
    )
    meter["weekly_forecast"] = meter["consumption_kwh"].shift(168)
    comparison = meter.tail(168)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        comparison["timestamp"],
        comparison["consumption_kwh"],
        label="Actual",
        color=CHART_STYLE["actual"],
        linewidth=1.4,
    )
    ax.plot(
        comparison["timestamp"],
        comparison["weekly_forecast"],
        label="Weekly seasonal forecast",
        color=CHART_STYLE["forecast"],
        linewidth=1.3,
    )
    ax.set(title=f"Actual vs forecast — {meter_id}", ylabel="Consumption (kWh)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.35, color=CHART_STYLE["grid"])
    paths["actual_vs_forecast"] = _finish(fig, output_dir / "actual_vs_forecast.png")

    valid = backtests.loc[backtests["status"] == "ok"]
    model_wape = valid.groupby("model")["wape"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    if model_wape.empty:
        ax.text(0.5, 0.5, "No valid backtest metrics", ha="center", va="center")
    else:
        ax.bar(model_wape.index, model_wape.values * 100, color=CHART_STYLE["forecast"])
    ax.set(title="Average validation WAPE by model", ylabel="WAPE (%)", xlabel="Model")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.35, color=CHART_STYLE["grid"])
    paths["model_wape_comparison"] = _finish(
        fig, output_dir / "model_wape_comparison.png"
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        portfolio_forecast["timestamp"],
        portfolio_forecast["forecast_kwh"],
        marker="o",
        markersize=3,
        color=CHART_STYLE["forecast"],
    )
    ax.fill_between(
        portfolio_forecast["timestamp"],
        portfolio_forecast["forecast_kwh"],
        alpha=0.12,
        color=CHART_STYLE["forecast"],
    )
    ax.set(
        title="Portfolio next-24-hour forecast",
        ylabel="Forecast (kWh)",
        xlabel="Forecast hour",
    )
    ax.grid(alpha=0.35, color=CHART_STYLE["grid"])
    paths["portfolio_24h_forecast"] = _finish(
        fig, output_dir / "portfolio_24h_forecast.png"
    )

    if anomalies.empty:
        example_meter = meter_id
        example_time = anomaly_pairs.loc[
            anomaly_pairs["meter_id"] == meter_id, "timestamp"
        ].iloc[-1]
    else:
        example = anomalies.loc[anomalies["anomaly_type"] != "missing_reading"]
        if example.empty:
            example = anomalies
        example_meter = str(example.iloc[0]["meter_id"])
        example_time = pd.Timestamp(example.iloc[0]["timestamp"])
    pairs = anomaly_pairs.loc[anomaly_pairs["meter_id"] == example_meter].copy()
    pairs["timestamp"] = pd.to_datetime(pairs["timestamp"])
    window = pairs.loc[
        pairs["timestamp"].between(
            example_time - pd.Timedelta("24h"), example_time + pd.Timedelta("24h")
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        window["timestamp"],
        window["actual_kwh"],
        label="Actual",
        color=CHART_STYLE["actual"],
        marker=".",
    )
    ax.plot(
        window["timestamp"],
        window["expected_kwh"],
        label="Expected",
        color=CHART_STYLE["forecast"],
        linestyle="--",
    )
    ax.axvline(
        example_time,
        color=CHART_STYLE["accent"],
        linestyle=":",
        label="Flagged observation",
    )
    ax.set(
        title=f"Explainable anomaly example — {example_meter}",
        ylabel="Consumption (kWh)",
    )
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.35, color=CHART_STYLE["grid"])
    paths["anomaly_example"] = _finish(fig, output_dir / "anomaly_example.png")
    return paths
