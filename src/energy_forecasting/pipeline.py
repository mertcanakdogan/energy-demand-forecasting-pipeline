"""End-to-end synthetic energy-demand forecasting demonstration."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .anomaly_detection import AnomalyConfig, detect_anomalies, weekly_expected_frame
from .backtesting import BacktestConfig, backtest_portfolio
from .features import add_calendar_features
from .forecasting import aggregate_forecasts, generate_forecasts
from .model_selection import select_best_models
from .models import selected_factories
from .synthetic_data import (
    SyntheticDataConfig,
    generate_synthetic_data,
    save_synthetic_data,
)
from .validation import prepare_data, validate_data
from .visualization import save_example_charts

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for a reproducible demonstration run."""

    seed: int = 42
    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    models: tuple[str, ...] = (
        "seasonal_naive",
        "weighted_baseline",
        "holt_winters",
        "sarimax",
    )
    forecast_horizon: int = 24
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    output_dir: Path = Path("outputs/examples")
    generated_data_path: Path = Path("data/generated/synthetic_meter_data.csv")

    def __post_init__(self) -> None:
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive")


def load_config(path: Path) -> PipelineConfig:
    """Load the public YAML configuration into typed settings."""

    with Path(path).open(encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}
    seed = int(raw.get("seed", 42))
    data_values = dict(raw.get("data", {}))
    data_values.setdefault("seed", seed)
    anomaly_values = dict(raw.get("anomaly", {}))
    return PipelineConfig(
        seed=seed,
        data=SyntheticDataConfig(**data_values),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        models=tuple(raw.get("models", PipelineConfig().models)),
        forecast_horizon=int(raw.get("forecast_horizon", 24)),
        anomaly=AnomalyConfig(**anomaly_values),
        output_dir=Path(raw.get("output_dir", "outputs/examples")),
        generated_data_path=Path(
            raw.get("generated_data_path", "data/generated/synthetic_meter_data.csv")
        ),
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return path


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Path]:
    """Run generation through anomaly reporting and write all artefacts."""

    config = config or PipelineConfig()
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Generating deterministic synthetic portfolio (seed=%d)", config.seed)
    raw = generate_synthetic_data(config.data)
    artefacts: dict[str, Path] = {
        "generated_data": save_synthetic_data(raw, config.generated_data_path)
    }

    validation = validate_data(raw)
    validation_frame = validation.to_frame()
    artefacts["validation_report"] = _write_csv(
        validation_frame, config.output_dir / "validation_report.csv"
    )
    LOGGER.info(
        "Raw validation found %d issue classes; preparing repairable gaps and duplicates",
        len(validation.issues),
    )
    prepared = prepare_data(raw)
    featured = add_calendar_features(prepared)
    LOGGER.info(
        "Prepared %s rows with %d engineered calendar columns",
        f"{len(prepared):,}",
        len(featured.columns) - len(prepared.columns),
    )

    factories = selected_factories(config.models)
    backtests = backtest_portfolio(prepared, factories, config.backtest)
    artefacts["backtest_results"] = _write_csv(
        backtests, config.output_dir / "backtest_results.csv"
    )
    selections = select_best_models(
        backtests,
        prepared["meter_id"].unique(),
        fallback_model="seasonal_naive",
        min_valid_folds=config.backtest.folds,
    )
    artefacts["model_selection"] = _write_csv(
        selections, config.output_dir / "model_selection.csv"
    )

    meter_forecast = generate_forecasts(
        prepared,
        selections,
        horizon=config.forecast_horizon,
        model_factories=factories,
        max_train_hours=config.backtest.max_train_hours,
    )
    group_forecast, portfolio_forecast = aggregate_forecasts(meter_forecast)
    artefacts["meter_forecast"] = _write_csv(
        meter_forecast, config.output_dir / "meter_forecast.csv"
    )
    artefacts["group_forecast"] = _write_csv(
        group_forecast, config.output_dir / "group_forecast.csv"
    )
    artefacts["portfolio_forecast"] = _write_csv(
        portfolio_forecast, config.output_dir / "portfolio_forecast.csv"
    )

    anomaly_pairs = weekly_expected_frame(prepared)
    anomalies = detect_anomalies(anomaly_pairs, config.anomaly)
    artefacts["anomalies"] = _write_csv(anomalies, config.output_dir / "anomalies.csv")
    chart_paths = save_example_charts(
        prepared,
        backtests,
        portfolio_forecast,
        anomaly_pairs,
        anomalies,
        config.output_dir,
    )
    artefacts.update(chart_paths)

    valid_selected = selections["wape"].dropna()
    metadata = {
        "seed": config.seed,
        "synthetic_only": True,
        "n_meters": int(prepared["meter_id"].nunique()),
        "n_groups": int(prepared["group_id"].nunique()),
        "raw_rows": len(raw),
        "prepared_rows": len(prepared),
        "history_start": pd.Timestamp(prepared["timestamp"].min()).isoformat(),
        "history_end": pd.Timestamp(prepared["timestamp"].max()).isoformat(),
        "forecast_horizon": config.forecast_horizon,
        "backtest_folds": config.backtest.folds,
        "models": list(config.models),
        "selected_model_counts": {
            str(key): int(value)
            for key, value in selections["selected_model"].value_counts().items()
        },
        "mean_selected_wape": round(float(valid_selected.mean()), 6)
        if not valid_selected.empty
        else None,
        "mean_selected_mae": round(float(selections["mae"].dropna().mean()), 6)
        if selections["mae"].notna().any()
        else None,
        "portfolio_forecast_kwh": round(
            float(portfolio_forecast["forecast_kwh"].sum()), 3
        ),
        "anomaly_count": len(anomalies),
        "validation_issue_counts": {
            issue.code: issue.count for issue in validation.issues
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    metadata_path = config.output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artefacts["run_metadata"] = metadata_path
    LOGGER.info(
        "Pipeline complete: %d meters, %.1f kWh next-day portfolio forecast, %d anomalies",
        metadata["n_meters"],
        metadata["portfolio_forecast_kwh"],
        metadata["anomaly_count"],
    )
    return artefacts


def main(argv: list[str] | None = None) -> None:
    """Run the configured demonstration from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_pipeline(load_config(args.config))


if __name__ == "__main__":
    main()
