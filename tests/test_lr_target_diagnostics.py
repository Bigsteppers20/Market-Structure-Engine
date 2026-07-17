"""Tests for linear_regression.target_diagnostics -- Phase 1 target audit."""
from __future__ import annotations

import numpy as np
import pytest

from linear_regression.target_diagnostics import (
    analyze_all_targets,
    analyze_target,
    augmented_dickey_fuller,
)


def test_adf_detects_stationary_white_noise() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(0, 1, 500)
    result = augmented_dickey_fuller(series)
    assert result.is_stationary_5pct is True
    assert result.adf_statistic < result.critical_values["5%"]


def test_adf_detects_non_stationary_random_walk() -> None:
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 500))
    result = augmented_dickey_fuller(series)
    assert result.is_stationary_5pct is False
    assert result.adf_statistic > result.critical_values["5%"]


def test_adf_handles_short_series_gracefully() -> None:
    result = augmented_dickey_fuller(np.array([1.0, 2.0, 3.0]))
    assert result.is_stationary_5pct is False
    assert not np.isfinite(result.adf_statistic)


def test_analyze_target_basic_statistics() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(5.0, 2.0, 300)
    X = rng.normal(0, 1, (300, 4))
    names = ["f0", "f1", "f2", "f3"]
    diag = analyze_target("my_target", y, X, names)
    assert diag.n_samples == 300
    assert diag.mean == pytest.approx(y.mean())
    assert diag.std == pytest.approx(y.std())
    assert diag.missing_percentage == 0.0
    assert len(diag.top_correlated_features) <= 15


def test_analyze_target_detects_missing_values() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 100)
    y[:10] = np.nan
    X = rng.normal(0, 1, (100, 3))
    diag = analyze_target("t", y, X, ["a", "b", "c"])
    assert diag.missing_percentage == pytest.approx(10.0)
    assert diag.n_samples == 90


def test_analyze_target_detects_outliers() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 500)
    y[0:5] = 100.0  # extreme outliers
    X = rng.normal(0, 1, (500, 2))
    diag = analyze_target("t", y, X, ["a", "b"])
    assert diag.outlier_percentage > 0.0


def test_analyze_target_ranks_correlated_feature_first() -> None:
    rng = np.random.default_rng(4)
    n = 500
    strong_feature = rng.normal(0, 1, n)
    y = strong_feature * 2.0 + rng.normal(0, 0.01, n)  # near-perfect linear relationship
    weak_feature = rng.normal(0, 1, n)
    X = np.column_stack([weak_feature, strong_feature])
    diag = analyze_target("t", y, X, ["weak", "strong"])
    assert diag.top_correlated_features[0][0] == "strong"
    assert abs(diag.top_correlated_features[0][1]) > 0.9


def test_analyze_target_all_missing_raises() -> None:
    y = np.full(50, np.nan)
    X = np.zeros((50, 2))
    with pytest.raises(ValueError):
        analyze_target("t", y, X, ["a", "b"])


def test_analyze_all_targets_returns_one_result_per_target() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (200, 3))
    y_reg = {"target_a": rng.normal(0, 1, 200), "target_b": rng.normal(5, 2, 200)}
    results = analyze_all_targets(y_reg, X, ["f0", "f1", "f2"])
    assert set(results) == {"target_a", "target_b"}


def test_target_diagnostics_to_dict_serializable() -> None:
    rng = np.random.default_rng(6)
    y = rng.normal(0, 1, 200)
    X = rng.normal(0, 1, (200, 3))
    diag = analyze_target("t", y, X, ["a", "b", "c"])
    d = diag.to_dict()
    assert d["name"] == "t"
    assert "stationarity" in d and isinstance(d["stationarity"], dict)
    assert "top_correlated_features" in d
