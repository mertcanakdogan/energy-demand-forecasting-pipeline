from __future__ import annotations

import json

import pandas as pd

from energy_forecasting.backtesting import BacktestConfig
from energy_forecasting.pipeline import PipelineConfig, run_pipeline
from energy_forecasting.synthetic_data import SyntheticDataConfig


def test_small_pipeline_writes_reconciling_outputs(tmp_path) -> None:
    config = PipelineConfig(
        seed=11,
        data=SyntheticDataConfig(
            n_meters=3,
            n_groups=2,
            months=1,
            seed=11,
            missing_rate=0.001,
            duplicate_rate=0.001,
            anomaly_rate=0.002,
        ),
        backtest=BacktestConfig(
            horizon=12, folds=1, min_train_hours=168, max_train_hours=336
        ),
        models=("seasonal_naive", "weighted_baseline"),
        forecast_horizon=24,
        output_dir=tmp_path / "outputs",
        generated_data_path=tmp_path / "data" / "synthetic.csv",
    )
    artefacts = run_pipeline(config)
    required = {
        "model_selection.csv",
        "meter_forecast.csv",
        "group_forecast.csv",
        "portfolio_forecast.csv",
        "anomalies.csv",
        "validation_report.csv",
        "run_metadata.json",
        "actual_vs_forecast.png",
        "model_wape_comparison.png",
        "portfolio_24h_forecast.png",
        "anomaly_example.png",
    }
    assert required <= {path.name for path in artefacts.values()}
    meter = pd.read_csv(config.output_dir / "meter_forecast.csv")
    portfolio = pd.read_csv(config.output_dir / "portfolio_forecast.csv")
    assert len(meter) == 3 * 24
    assert len(portfolio) == 24
    metadata = json.loads(
        (config.output_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["seed"] == 11
    assert metadata["n_meters"] == 3


def test_small_pipeline_is_reproducible(tmp_path) -> None:
    base = {
        "seed": 5,
        "data": SyntheticDataConfig(n_meters=2, n_groups=1, months=1, seed=5),
        "backtest": BacktestConfig(
            horizon=12, folds=1, min_train_hours=168, max_train_hours=336
        ),
        "models": ("seasonal_naive",),
        "forecast_horizon": 24,
    }
    first = PipelineConfig(
        **base, output_dir=tmp_path / "one", generated_data_path=tmp_path / "one.csv"
    )
    second = PipelineConfig(
        **base, output_dir=tmp_path / "two", generated_data_path=tmp_path / "two.csv"
    )
    run_pipeline(first)
    run_pipeline(second)
    left = pd.read_csv(first.output_dir / "meter_forecast.csv")
    right = pd.read_csv(second.output_dir / "meter_forecast.csv")
    pd.testing.assert_frame_equal(left, right)
