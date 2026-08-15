from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting.validation import (
    DataValidationError,
    prepare_data,
    validate_data,
)


def issue_codes(frame: pd.DataFrame) -> set[str]:
    return {issue.code for issue in validate_data(frame).issues}


def test_valid_frame_has_no_errors(hourly_frame: pd.DataFrame) -> None:
    assert validate_data(hourly_frame).is_valid


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda df: df.drop(columns="temperature_c"), "missing_columns"),
        (lambda df: df.assign(extra="unexpected"), "unexpected_columns"),
        (lambda df: pd.concat([df, df.iloc[[0]]]), "duplicate_reading"),
        (lambda df: df.drop(index=10), "missing_timestamp"),
        (
            lambda df: df.assign(
                consumption_kwh=lambda x: x["consumption_kwh"].mask(x.index == 0, -1)
            ),
            "negative_consumption",
        ),
        (lambda df: df.assign(meter_id="bad id"), "invalid_meter_id"),
        (lambda df: df.assign(group_id="!"), "invalid_group_id"),
        (lambda df: df.assign(temperature_c=200), "invalid_temperature"),
    ],
)
def test_validation_reports_quality_problem(hourly_frame, mutation, code) -> None:
    assert code in issue_codes(mutation(hourly_frame.copy()))


def test_strict_validation_raises_clear_error(hourly_frame: pd.DataFrame) -> None:
    invalid = pd.concat([hourly_frame, hourly_frame.iloc[[0]]])
    with pytest.raises(DataValidationError, match="duplicate_reading"):
        validate_data(invalid, raise_on_error=True)


def test_prepare_data_resolves_duplicates_and_gaps(hourly_frame: pd.DataFrame) -> None:
    damaged = pd.concat(
        [hourly_frame.drop(index=10), hourly_frame.iloc[[0]]], ignore_index=True
    )
    prepared = prepare_data(damaged)
    assert not prepared.duplicated(["meter_id", "timestamp"]).any()
    assert len(prepared) == len(hourly_frame)
    missing_row = prepared.loc[
        prepared["timestamp"] == hourly_frame.loc[10, "timestamp"]
    ].iloc[0]
    assert bool(missing_row["was_missing"])
    assert pd.notna(missing_row["consumption_kwh"])


def test_prepare_data_rejects_negative_consumption(hourly_frame: pd.DataFrame) -> None:
    hourly_frame.loc[0, "consumption_kwh"] = -0.1
    with pytest.raises(DataValidationError, match="negative_consumption"):
        prepare_data(hourly_frame)
