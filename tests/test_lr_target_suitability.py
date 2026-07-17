"""Tests for linear_regression.target_suitability -- Phase 6 target
suitability. Pure aggregation over Phase 1/2/3 diagnostics -- built here via
the real analyze_target / audit_features_for_target / analyze_residuals
functions (never hand-constructed dataclasses), same convention as the
other lr_ test files."""
from __future__ import annotations

import numpy as np

from linear_regression.feature_diagnostics import audit_features_for_target
from linear_regression.residual_diagnostics import analyze_residuals
from linear_regression.target_diagnostics import analyze_target
from linear_regression.target_suitability import assess_target_suitability


def test_poor_target_flagged_with_low_signal_reasons() -> None:
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(0, 1, (n, 5))
    y = rng.normal(0, 1, n)  # independent noise -- no relationship to X
    names = ["a", "b", "c", "d", "e"]

    target_diag = analyze_target("noisy_target", y, X, names)
    feature_audit = audit_features_for_target(X, y, names, "noisy_target")
    y_pred = np.full(n, y.mean())
    residual_diag = analyze_residuals("noisy_target", y, y_pred, X, names)

    result = assess_target_suitability(
        "noisy_target", r2_before=-0.01, cv_mean_r2=-0.02, cv_std_r2=0.05,
        target_diag=target_diag, feature_audit=feature_audit, residual_diag=residual_diag,
    )
    assert result.verdict == "poor"
    assert any("weak" in r or "signal" in r or "below" in r for r in result.reasons)
    assert result.suggested_reformulation is not None


def test_suitable_target_has_no_poor_reasons_and_no_reformulation() -> None:
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(0, 1, (n, 4))
    y = X[:, 0] * 5.0 + rng.normal(0, 0.05, n)
    names = ["strong", "b", "c", "d"]

    target_diag = analyze_target("clean_target", y, X, names)
    feature_audit = audit_features_for_target(X, y, names, "clean_target")
    y_pred = X[:, 0] * 5.0
    residual_diag = analyze_residuals("clean_target", y, y_pred, X, names)

    result = assess_target_suitability(
        "clean_target", r2_before=0.9, cv_mean_r2=0.85, cv_std_r2=0.03,
        target_diag=target_diag, feature_audit=feature_audit, residual_diag=residual_diag,
    )
    assert result.verdict == "suitable"
    assert result.suggested_reformulation is None


def test_price_level_target_with_trivial_correlation_flagged_despite_high_r2() -> None:
    rng = np.random.default_rng(2)
    n = 400
    y = rng.normal(1.1000, 0.01, n)
    close_feature = y + rng.normal(0, 1e-7, n)  # near-identical to the target
    other = rng.normal(0, 1, n)
    X = np.column_stack([close_feature, other])
    names = ["close", "other"]

    target_diag = analyze_target("next_close", y, X, names)
    feature_audit = audit_features_for_target(X, y, names, "next_close")
    y_pred = y + rng.normal(0, 0.0001, n)
    residual_diag = analyze_residuals("next_close", y, y_pred, X, names)

    result = assess_target_suitability(
        "next_close", r2_before=0.999, cv_mean_r2=0.998, cv_std_r2=0.0005,
        target_diag=target_diag, feature_audit=feature_audit, residual_diag=residual_diag,
    )
    assert result.verdict == "suitable"
    assert result.suggested_reformulation is not None
    assert "autocorrelation" in result.suggested_reformulation
    assert any("autocorrelation" in r for r in result.reasons)


def test_to_dict_serializable() -> None:
    rng = np.random.default_rng(3)
    n = 200
    X = rng.normal(0, 1, (n, 3))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.5, n)
    names = ["a", "b", "c"]

    target_diag = analyze_target("t", y, X, names)
    feature_audit = audit_features_for_target(X, y, names, "t")
    residual_diag = analyze_residuals("t", y, X[:, 0] * 2.0, X, names)

    result = assess_target_suitability(
        "t", r2_before=0.5, cv_mean_r2=None, cv_std_r2=None,
        target_diag=target_diag, feature_audit=feature_audit, residual_diag=residual_diag,
    )
    d = result.to_dict()
    assert d["target"] == "t"
    assert d["cv_mean_r2"] is None
    assert "reasons" in d and isinstance(d["reasons"], list)
