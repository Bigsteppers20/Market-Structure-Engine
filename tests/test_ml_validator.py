"""Tests for ml_pipeline.validator -- especially the leakage assertion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from ml_pipeline.dataset_builder import Dataset
from ml_pipeline.validator import (
    DataValidationError,
    LeakageError,
    assert_no_future_leakage,
    validate_dataset,
    validate_input_data,
)


def test_validate_input_data_clean() -> None:
    df = make_ohlcv(100)
    assert validate_input_data(df) == []


def test_validate_input_data_missing_columns_raises() -> None:
    df = make_ohlcv(10).drop(columns=["close"])
    with pytest.raises(DataValidationError):
        validate_input_data(df)


def test_validate_input_data_empty_raises() -> None:
    with pytest.raises(DataValidationError):
        validate_input_data(pd.DataFrame())


def test_validate_input_data_flags_duplicates_and_gaps() -> None:
    df = make_ohlcv(20)
    df.loc[19, "timestamp"] = df.loc[18, "timestamp"]  # duplicate
    dropped = df.drop(index=10).reset_index(drop=True)  # gap
    issues = validate_input_data(dropped)
    assert any("gap" in i.lower() for i in issues)

    dup_issues = validate_input_data(df)
    assert any("duplicate" in i.lower() for i in dup_issues)


def test_assert_no_future_leakage_ok() -> None:
    # window [10..20], decision at 20, label at 21 -- valid.
    assert_no_future_leakage(decision_index=20, window_start=10, window_end=20, label_index=21)


def test_assert_no_future_leakage_window_end_after_decision() -> None:
    with pytest.raises(LeakageError):
        assert_no_future_leakage(decision_index=20, window_start=10, window_end=21, label_index=22)


def test_assert_no_future_leakage_inverted_window() -> None:
    with pytest.raises(LeakageError):
        assert_no_future_leakage(decision_index=20, window_start=21, window_end=15, label_index=25)


def test_assert_no_future_leakage_label_not_in_future() -> None:
    with pytest.raises(LeakageError):
        assert_no_future_leakage(decision_index=20, window_start=10, window_end=20, label_index=20)
    with pytest.raises(LeakageError):
        assert_no_future_leakage(decision_index=20, window_start=10, window_end=20, label_index=19)


def _tiny_dataset(n: int = 5, n_features: int = 3) -> Dataset:
    X = np.arange(n * n_features, dtype=np.float64).reshape(n, n_features)
    names = [f"f{i}" for i in range(n_features)]
    y_reg = {"next_close": np.linspace(1.0, 2.0, n)}
    y_cls = np.array([0, 1, 2, 1, 0])
    meta = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min"),
        "symbol": "EUR_USD", "timeframe": "M5",
        "window_start": pd.date_range("2025-01-01", periods=n, freq="5min"),
        "window_end": pd.date_range("2025-01-01", periods=n, freq="5min"),
    })
    return Dataset(X=X, feature_names=names, y_reg=y_reg, y_cls=y_cls,
                    class_names=["SELL", "NO_TRADE", "BUY"], metadata=meta)


def test_validate_dataset_happy_path() -> None:
    ds = _tiny_dataset()
    report = validate_dataset(ds, expected_n_features=3)
    assert report.is_valid
    assert report.n_samples == 5
    assert report.feature_count_ok
    assert report.class_balance == {"SELL": 2, "NO_TRADE": 2, "BUY": 1}
    assert "next_close" in report.regression_stats


def test_validate_dataset_flags_feature_count_mismatch() -> None:
    ds = _tiny_dataset()
    report = validate_dataset(ds, expected_n_features=99)
    assert not report.is_valid
    assert not report.feature_count_ok


def test_validate_dataset_flags_nan() -> None:
    ds = _tiny_dataset()
    ds.X[0, 0] = np.nan
    report = validate_dataset(ds, expected_n_features=3)
    assert not report.is_valid
    assert report.nan_count == 1


def test_validate_dataset_flags_misaligned_targets() -> None:
    ds = _tiny_dataset()
    ds.y_reg["next_close"] = ds.y_reg["next_close"][:-1]
    report = validate_dataset(ds, expected_n_features=3)
    assert not report.is_valid
    assert not report.target_alignment_ok
