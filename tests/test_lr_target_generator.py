"""Tests for linear_regression.target_generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from linear_regression.exceptions import UnsupportedTargetError
from linear_regression.target_generator import (
    REGRESSION_TARGET_REGISTRY,
    TARGET_TO_PREDICTION_FIELD,
    compute_target,
    compute_targets,
)
from ml_pipeline.label_generator import REGRESSION_REGISTRY as ML_PIPELINE_TARGETS


def _df() -> pd.DataFrame:
    high = [1.1010, 1.1030, 1.1005, 1.1050, 1.0990, 1.1060, 1.1000]
    low = [1.0990, 1.1000, 1.0980, 1.1010, 1.0950, 1.1020, 1.0970]
    close = [1.1000, 1.1010, 1.0990, 1.1030, 1.0970, 1.1040, 1.0985]
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=len(close), freq="5min"),
        "open": close, "high": high, "low": low, "close": close, "volume": [100.0] * len(close),
    })


def test_reuses_ml_pipeline_targets_without_duplication() -> None:
    for name, fn in ML_PIPELINE_TARGETS.items():
        assert REGRESSION_TARGET_REGISTRY[name] is fn  # literally the same function object


def test_new_targets_present() -> None:
    for name in ("maximum_favorable_excursion", "maximum_adverse_excursion",
                 "average_future_price", "future_range", "future_midpoint"):
        assert name in REGRESSION_TARGET_REGISTRY


def test_mfe_mae_are_nonnegative_and_bracket_correctly() -> None:
    df = _df()
    mfe = compute_target(df, 0, 3, "maximum_favorable_excursion")
    mae = compute_target(df, 0, 3, "maximum_adverse_excursion")
    entry = df["close"].iloc[0]
    window = df.iloc[1:4]
    assert mfe == pytest.approx(window["high"].max() - entry)
    assert mae == pytest.approx(entry - window["low"].min())
    assert mfe >= 0
    assert mae >= 0


def test_average_future_price() -> None:
    df = _df()
    avg = compute_target(df, 0, 3, "average_future_price")
    assert avg == pytest.approx(df["close"].iloc[1:4].mean())


def test_future_range_and_midpoint_consistency() -> None:
    df = _df()
    rng = compute_target(df, 0, 3, "future_range")
    mid = compute_target(df, 0, 3, "future_midpoint")
    window = df.iloc[1:4]
    assert rng == pytest.approx(window["high"].max() - window["low"].min())
    assert mid == pytest.approx((window["high"].max() + window["low"].min()) / 2.0)


def test_targets_degrade_gracefully_at_series_end() -> None:
    df = _df()
    last = len(df) - 1
    # No future bars available -- must not crash, returns a sane default.
    assert compute_target(df, last, 3, "future_range") == 0.0
    assert compute_target(df, last, 3, "maximum_favorable_excursion") == 0.0


def test_unsupported_target_raises() -> None:
    with pytest.raises(UnsupportedTargetError):
        compute_target(_df(), 0, 1, "not_a_real_target")


def test_compute_targets_batch() -> None:
    df = _df()
    out = compute_targets(df, 0, 2, ["next_close", "future_range"])
    assert set(out) == {"next_close", "future_range"}


def test_target_to_prediction_field_covers_named_output_fields() -> None:
    expected_fields = {
        "expected_close", "expected_high", "expected_low", "expected_return",
        "expected_pip_move", "expected_volatility", "expected_MFE", "expected_MAE",
    }
    assert set(TARGET_TO_PREDICTION_FIELD.values()) == expected_fields
