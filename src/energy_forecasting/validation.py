"""Data-quality validation and explicit preparation for modelling."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "meter_id",
    "group_id",
    "timestamp",
    "consumption_kwh",
    "temperature_c",
    "is_weekend",
    "is_holiday",
)
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ValidationIssue:
    """One class of data-quality problem."""

    code: str
    message: str
    count: int
    severity: str = "error"


@dataclass(frozen=True)
class ValidationReport:
    """Collection of validation findings."""

    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_frame(self) -> pd.DataFrame:
        columns = ["code", "message", "count", "severity"]
        return pd.DataFrame([issue.__dict__ for issue in self.issues], columns=columns)


class DataValidationError(ValueError):
    """Raised when input data cannot safely be prepared."""


def _append(issues: list[ValidationIssue], code: str, message: str, count: int) -> None:
    if count:
        issues.append(ValidationIssue(code, message, int(count)))


def validate_data(
    data: pd.DataFrame, *, raise_on_error: bool = False
) -> ValidationReport:
    """Validate schema, keys, continuity, identifiers, and plausible values."""

    issues: list[ValidationIssue] = []
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    unexpected = sorted(set(data.columns) - set(REQUIRED_COLUMNS))
    _append(
        issues, "missing_columns", f"Missing required columns: {missing}", len(missing)
    )
    _append(
        issues,
        "unexpected_columns",
        f"Unexpected columns: {unexpected}",
        len(unexpected),
    )

    if not missing:
        timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
        _append(
            issues,
            "invalid_timestamp",
            "Timestamps must be parseable",
            timestamps.isna().sum(),
        )
        duplicates = data.duplicated(["meter_id", "timestamp"], keep=False).sum()
        _append(
            issues,
            "duplicate_reading",
            "Duplicate meter_id/timestamp combinations found",
            duplicates,
        )
        if timestamps.notna().any():
            temp = data.assign(timestamp=timestamps).dropna(subset=["timestamp"])
            missing_hours = 0
            for _, meter in temp.groupby("meter_id", sort=False):
                unique_hours = meter["timestamp"].nunique()
                span = meter["timestamp"].max() - meter["timestamp"].min()
                expected_hours = int(span.total_seconds() // 3600) + 1
                missing_hours += max(0, expected_hours - unique_hours)
            _append(
                issues,
                "missing_timestamp",
                "Hourly gaps found within meter histories",
                missing_hours,
            )

        consumption = pd.to_numeric(data["consumption_kwh"], errors="coerce")
        _append(
            issues,
            "invalid_consumption",
            "Consumption must be numeric",
            consumption.isna().sum(),
        )
        _append(
            issues,
            "negative_consumption",
            "Consumption cannot be negative",
            (consumption < 0).sum(),
        )
        temperature = pd.to_numeric(data["temperature_c"], errors="coerce")
        invalid_temp = temperature.isna() | ~temperature.between(-60, 60)
        _append(
            issues,
            "invalid_temperature",
            "Temperature must be numeric and between -60 and 60 C",
            invalid_temp.sum(),
        )
        for column, code in (
            ("meter_id", "invalid_meter_id"),
            ("group_id", "invalid_group_id"),
        ):
            invalid_id = ~data[column].astype("string").fillna("").str.match(ID_PATTERN)
            _append(
                issues, code, f"{column} contains invalid identifiers", invalid_id.sum()
            )

    report = ValidationReport(tuple(issues))
    if raise_on_error and not report.is_valid:
        codes = ", ".join(issue.code for issue in report.issues)
        raise DataValidationError(f"Data validation failed: {codes}")
    return report


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Resolve duplicates and gaps after rejecting unsafe quality problems.

    Duplicate numeric readings are averaged. Missing hours are linearly
    interpolated and tagged with ``was_missing`` for anomaly reporting.
    """

    report = validate_data(data)
    repairable = {"duplicate_reading", "missing_timestamp"}
    fatal = [issue for issue in report.issues if issue.code not in repairable]
    if fatal:
        codes = ", ".join(issue.code for issue in fatal)
        raise DataValidationError(f"Data preparation rejected input: {codes}")

    working = data.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"])
    aggregation = {
        "group_id": "first",
        "consumption_kwh": "mean",
        "temperature_c": "mean",
        "is_weekend": "max",
        "is_holiday": "max",
    }
    working = (
        working.groupby(["meter_id", "timestamp"], as_index=False, sort=True)
        .agg(aggregation)
        .sort_values(["meter_id", "timestamp"])
    )

    prepared: list[pd.DataFrame] = []
    for meter_id, meter in working.groupby("meter_id", sort=True):
        meter = meter.set_index("timestamp").sort_index()
        full_index = pd.date_range(meter.index.min(), meter.index.max(), freq="h")
        original_index = meter.index
        meter = meter.reindex(full_index)
        meter.index.name = "timestamp"
        meter["was_missing"] = (
            ~meter.index.isin(original_index) | meter["consumption_kwh"].isna()
        )
        meter["meter_id"] = meter_id
        meter["group_id"] = meter["group_id"].ffill().bfill()
        meter["temperature_c"] = meter["temperature_c"].interpolate(
            limit_direction="both"
        )
        meter["consumption_kwh"] = meter["consumption_kwh"].interpolate(
            limit_direction="both"
        )
        meter["is_weekend"] = meter.index.dayofweek >= 5
        holiday = np.isin(meter.index.strftime("%m-%d"), ["01-01", "07-01", "12-25"])
        meter["is_holiday"] = holiday
        prepared.append(meter.reset_index())

    columns = list(REQUIRED_COLUMNS) + ["was_missing"]
    return pd.concat(prepared, ignore_index=True).loc[:, columns]
