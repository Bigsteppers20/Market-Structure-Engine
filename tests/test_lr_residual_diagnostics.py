"""Tests for linear_regression.residual_diagnostics -- Phase 3 residual analysis."""
from __future__ import annotations

import numpy as np
import pytest

from linear_regression.residual_diagnostics import (
    analyze_residuals,
    autocorrelation,
    breusch_pagan_test,
    durbin_watson,
    qq_analysis,
    residual_regime_breakdown,
    test_normality as check_normality,
)


def test_durbin_watson_near_2_for_uncorrelated_residuals() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 1, 1000)
    dw = durbin_watson(residuals)
    assert 1.8 < dw < 2.2


def test_durbin_watson_low_for_positively_autocorrelated_residuals() -> None:
    rng = np.random.default_rng(1)
    n = 1000
    residuals = np.zeros(n)
    residuals[0] = rng.normal()
    for t in range(1, n):
        residuals[t] = 0.9 * residuals[t - 1] + rng.normal(0, 0.3)
    dw = durbin_watson(residuals)
    assert dw < 1.0


def test_durbin_watson_handles_all_zero_residuals() -> None:
    assert np.isnan(durbin_watson(np.zeros(50)))


def test_autocorrelation_lag1_matches_expected_sign() -> None:
    rng = np.random.default_rng(2)
    n = 1000
    residuals = np.zeros(n)
    for t in range(1, n):
        residuals[t] = 0.8 * residuals[t - 1] + rng.normal(0, 0.3)
    acf = autocorrelation(residuals, max_lag=3)
    assert acf[1] > 0.5


def test_qq_analysis_high_correlation_for_normal_data() -> None:
    rng = np.random.default_rng(3)
    residuals = rng.normal(0, 1, 1000)
    result = qq_analysis(residuals)
    assert result.qq_correlation > 0.95
    assert result.n_points == 1000


def test_qq_analysis_lower_correlation_for_heavy_tailed_data() -> None:
    rng = np.random.default_rng(4)
    normal_residuals = rng.normal(0, 1, 2000)
    heavy_tailed = rng.standard_t(df=2, size=2000)
    normal_result = qq_analysis(normal_residuals)
    heavy_result = qq_analysis(heavy_tailed)
    assert normal_result.qq_correlation > heavy_result.qq_correlation


def test_normality_accepts_normal_data() -> None:
    rng = np.random.default_rng(5)
    residuals = rng.normal(0, 1, 1000)
    result = check_normality(residuals)
    assert result.is_normal_5pct is True


def test_normality_rejects_heavy_tailed_data() -> None:
    rng = np.random.default_rng(6)
    residuals = rng.standard_t(df=2, size=1000)
    result = check_normality(residuals)
    assert result.is_normal_5pct is False


def test_breusch_pagan_detects_heteroscedasticity() -> None:
    rng = np.random.default_rng(7)
    n = 1000
    fitted = rng.uniform(1, 10, n)
    # Residual variance grows with fitted value -- classic heteroscedasticity.
    residuals = rng.normal(0, 1, n) * fitted
    result = breusch_pagan_test(residuals, fitted)
    assert result.is_heteroscedastic_5pct is True


def test_breusch_pagan_accepts_homoscedastic_residuals() -> None:
    rng = np.random.default_rng(8)
    n = 1000
    fitted = rng.uniform(1, 10, n)
    residuals = rng.normal(0, 1, n)  # constant variance, independent of fitted value
    result = breusch_pagan_test(residuals, fitted)
    assert result.is_heteroscedastic_5pct is False


def test_residual_regime_breakdown_partitions_by_mask() -> None:
    rng = np.random.default_rng(9)
    n = 200
    residuals = rng.normal(0, 1, n)
    X = np.zeros((n, 2))
    X[:100, 0] = 0.8  # trending
    X[:100, 1] = 1.0
    breakdown = residual_regime_breakdown(residuals, X, ["trend_strength", "trend_direction"])
    assert breakdown["trending"]["n_samples"] == 100
    assert breakdown["ranging"]["n_samples"] == 100


def test_residual_regime_breakdown_flags_insufficient_samples() -> None:
    rng = np.random.default_rng(10)
    n = 20
    residuals = rng.normal(0, 1, n)
    X = np.zeros((n, 1))
    breakdown = residual_regime_breakdown(residuals, X, ["vol_expansion"], min_samples=10)
    assert breakdown["high_volatility"]["note"] == "insufficient samples"


def test_analyze_residuals_full_pipeline() -> None:
    rng = np.random.default_rng(11)
    n = 300
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + rng.normal(0, 0.2, n)
    X = rng.normal(0, 1, (n, 3))
    names = ["trend_strength", "vol_expansion", "session_is_london"]
    diag = analyze_residuals("t", y_true, y_pred, X, names)
    assert diag.target == "t"
    assert "mean" in diag.residual_statistics
    assert diag.qq.n_points == n
    d = diag.to_dict()
    assert "regime_breakdown" in d and "durbin_watson" in d
