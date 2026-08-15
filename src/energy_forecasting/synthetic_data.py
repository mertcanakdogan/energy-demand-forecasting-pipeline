"""Deterministic synthetic smart-meter data generation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SCHEMA = [
    "meter_id",
    "group_id",
    "timestamp",
    "consumption_kwh",
    "temperature_c",
    "is_weekend",
    "is_holiday",
]


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for a heterogeneous synthetic meter portfolio."""

    n_meters: int = 100
    n_groups: int = 12
    start: str = "2025-01-01 00:00:00"
    months: int = 13
    seed: int = 42
    missing_rate: float = 0.001
    duplicate_rate: float = 0.0002
    anomaly_rate: float = 0.0008


def _validate_config(config: SyntheticDataConfig) -> None:
    if config.n_meters < 1:
        raise ValueError("n_meters must be positive")
    if not 1 <= config.n_groups <= config.n_meters:
        raise ValueError("n_groups must be between 1 and the number of meters")
    if config.months < 1:
        raise ValueError("months must be positive")
    for name in ("missing_rate", "duplicate_rate", "anomaly_rate"):
        value = getattr(config, name)
        if not 0 <= value < 0.1:
            raise ValueError(f"{name} must be between 0 and 0.1")


def _holiday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Return a small fictional set of fixed-date public holidays."""

    month_day = index.strftime("%m-%d")
    return np.asarray(np.isin(month_day, ["01-01", "07-01", "12-25"]), dtype=bool)


def generate_synthetic_data(config: SyntheticDataConfig | None = None) -> pd.DataFrame:
    """Generate raw hourly readings with deterministic quality and anomaly events.

    Missing readings are represented by absent meter/timestamp rows. Duplicate rows
    are intentionally retained so the validation stage can demonstrate detection.
    """

    config = config or SyntheticDataConfig()
    _validate_config(config)
    rng = np.random.default_rng(config.seed)

    start = pd.Timestamp(config.start)
    end = start + pd.DateOffset(months=config.months)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    hour = index.hour.to_numpy()
    day_of_week = index.dayofweek.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    is_weekend = day_of_week >= 5
    is_holiday = _holiday_mask(index)

    annual_temperature = 16 + 10 * np.sin(2 * np.pi * (day_of_year - 30) / 365.25)
    daily_temperature = 2.5 * np.sin(2 * np.pi * (hour - 8) / 24)
    shared_temperature = (
        annual_temperature + daily_temperature + rng.normal(0, 1.2, len(index))
    )

    frames: list[pd.DataFrame] = []
    for meter_number in range(1, config.n_meters + 1):
        meter_id = f"M{meter_number:04d}"
        group_id = f"G{((meter_number - 1) % config.n_groups) + 1:02d}"
        base = rng.uniform(1.5, 18.0)
        amplitude = rng.uniform(0.15, 0.55)
        peak_hour = rng.choice([7.0, 8.0, 17.0, 18.0, 20.0]) + rng.normal(0, 0.7)
        hourly_shape = 1 + amplitude * np.cos(2 * np.pi * (hour - peak_hour) / 24)
        weekend_factor = np.where(is_weekend, rng.uniform(0.75, 1.18), 1.0)
        holiday_factor = np.where(is_holiday, rng.uniform(0.65, 1.10), 1.0)
        annual_load = 1 + rng.uniform(0.04, 0.18) * np.cos(
            2 * np.pi * (day_of_year - rng.uniform(0, 365)) / 365.25
        )
        temperature = shared_temperature + rng.normal(0, 0.5, len(index))
        sensitivity = rng.choice([0.0, rng.uniform(0.01, 0.06)], p=[0.35, 0.65])
        temperature_effect = 1 + sensitivity * np.maximum(
            np.abs(temperature - 19) - 4, 0
        )
        signal = (
            base
            * hourly_shape
            * weekend_factor
            * holiday_factor
            * annual_load
            * temperature_effect
        )
        noise = rng.normal(0, np.maximum(signal * rng.uniform(0.03, 0.12), 0.05))
        consumption = np.clip(signal + noise, 0.02, None)

        anomaly_count = (
            max(2, int(len(index) * config.anomaly_rate)) if config.anomaly_rate else 0
        )
        if anomaly_count:
            anomaly_positions = rng.choice(
                len(index), size=anomaly_count, replace=False
            )
            split = max(1, anomaly_count // 2)
            consumption[anomaly_positions[:split]] *= rng.uniform(4.0, 6.0, split)
            consumption[anomaly_positions[split:]] *= rng.uniform(
                0.02, 0.18, anomaly_count - split
            )

        frame = pd.DataFrame(
            {
                "meter_id": meter_id,
                "group_id": group_id,
                "timestamp": index,
                "consumption_kwh": consumption,
                "temperature_c": temperature,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
            }
        )
        missing_count = (
            max(1, int(len(frame) * config.missing_rate)) if config.missing_rate else 0
        )
        if missing_count:
            missing_positions = rng.choice(
                len(frame), size=missing_count, replace=False
            )
            frame = frame.drop(frame.index[missing_positions])
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    duplicate_count = (
        max(1, int(len(data) * config.duplicate_rate)) if config.duplicate_rate else 0
    )
    if duplicate_count:
        duplicate_positions = rng.choice(len(data), size=duplicate_count, replace=False)
        data = pd.concat(
            [data, data.iloc[duplicate_positions].copy()], ignore_index=True
        )

    return (
        data.loc[:, SCHEMA]
        .sort_values(["meter_id", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def save_synthetic_data(data: pd.DataFrame, path: Path) -> Path:
    """Save generated data, creating only the requested parent directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return path


def main(argv: list[str] | None = None) -> None:
    """Generate the default full synthetic dataset from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/generated/synthetic_meter_data.csv")
    )
    parser.add_argument("--meters", type=int, default=100)
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--months", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    config = SyntheticDataConfig(
        n_meters=args.meters,
        n_groups=args.groups,
        months=args.months,
        seed=args.seed,
    )
    data = generate_synthetic_data(config)
    path = save_synthetic_data(data, args.output)
    LOGGER.info("Wrote %s synthetic rows to %s", f"{len(data):,}", path)


if __name__ == "__main__":
    main()
