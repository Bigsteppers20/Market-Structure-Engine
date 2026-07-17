"""Tests for linear_regression.feature_diagnostics -- Phase 2 feature audit.
Recommendations only -- nothing is ever auto-removed."""
from __future__ import annotations

import numpy as np
import pytest

from linear_regression.feature_diagnostics import (
    audit_features_for_target,
    feature_importance_vs_target,
    highly_correlated_pairs,
    mutual_information_vs_target,
    near_constant_features,
    pearson_vs_target,
    spearman_vs_target,
    variance_inflation_factors,
)


def test_pearson_vs_target_detects_strong_relationship() -> None:
    rng = np.random.default_rng(0)
    n = 500
    strong = rng.normal(0, 1, n)
    y = strong * 3.0 + rng.normal(0, 0.01, n)
    weak = rng.normal(0, 1, n)
    X = np.column_stack([weak, strong])
    result = pearson_vs_target(X, y, ["weak", "strong"])
    assert abs(result["strong"]) > 0.9
    assert abs(result["weak"]) < 0.3


def test_spearman_vs_target_detects_monotonic_nonlinear_relationship() -> None:
    rng = np.random.default_rng(1)
    n = 500
    x = rng.uniform(0.1, 10, n)
    y = np.log(x) + rng.normal(0, 0.01, n)  # monotonic but non-linear
    result = spearman_vs_target(x.reshape(-1, 1), y, ["x"])
    assert abs(result["x"]) > 0.95


def test_mutual_information_vs_target_nonnegative() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (300, 4))
    y = rng.normal(0, 1, 300)
    mi = mutual_information_vs_target(X, y, ["a", "b", "c", "d"])
    assert all(v >= 0.0 for v in mi.values())


def test_near_constant_features_flags_zero_variance_column() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (200, 3))
    X[:, 1] = 5.0  # constant column
    flagged = near_constant_features(X, ["a", "constant", "c"])
    assert "constant" in flagged
    assert "a" not in flagged


def test_near_constant_features_flags_imbalanced_binary() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (1000, 2))
    binary = np.zeros(1000)
    binary[:5] = 1.0  # 99.5% zeros
    X = np.column_stack([X, binary])
    flagged = near_constant_features(X, ["a", "b", "imbalanced_binary"])
    assert "imbalanced_binary" in flagged


def test_variance_inflation_factors_high_for_duplicated_feature() -> None:
    rng = np.random.default_rng(5)
    n = 500
    base = rng.normal(0, 1, n)
    duplicate = base + rng.normal(0, 0.001, n)  # near-exact duplicate
    independent = rng.normal(0, 1, n)
    X = np.column_stack([base, duplicate, independent])
    vif = variance_inflation_factors(X, ["base", "duplicate", "independent"])
    assert vif["base"] > 10.0
    assert vif["duplicate"] > 10.0
    assert vif["independent"] < 5.0


def test_highly_correlated_pairs_detects_duplicate() -> None:
    rng = np.random.default_rng(6)
    n = 500
    base = rng.normal(0, 1, n)
    duplicate = base * 2.0  # perfectly correlated (different scale)
    independent = rng.normal(0, 1, n)
    X = np.column_stack([base, duplicate, independent])
    pairs = highly_correlated_pairs(X, ["base", "duplicate", "independent"])
    assert len(pairs) == 1
    assert set(pairs[0][:2]) == {"base", "duplicate"}
    assert abs(pairs[0][2]) > 0.99


def test_audit_features_for_target_never_removes_anything() -> None:
    """Phase 2's explicit requirement: recommendations only."""
    rng = np.random.default_rng(7)
    n = 400
    X = rng.normal(0, 1, (n, 5))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.5, n)
    names = ["a", "b", "c", "d", "e"]
    report = audit_features_for_target(X, y, names, "my_target")
    # Every original feature name must still be present in every diagnostic dict.
    assert set(report.pearson) == set(names)
    assert set(report.mutual_information) == set(names)
    assert set(report.vif) == set(names)
    assert isinstance(report.recommendations, list) and report.recommendations


def test_audit_recommendations_mention_high_correlation_pair() -> None:
    rng = np.random.default_rng(8)
    n = 500
    base = rng.normal(0, 1, n)
    duplicate = base + rng.normal(0, 0.0001, n)
    y = base + rng.normal(0, 0.5, n)
    X = np.column_stack([base, duplicate])
    report = audit_features_for_target(X, y, ["base", "duplicate"], "t")
    assert any("highly correlated" in r for r in report.recommendations)


def test_audit_report_to_dict_serializable() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(0, 1, (200, 3))
    y = rng.normal(0, 1, 200)
    report = audit_features_for_target(X, y, ["a", "b", "c"], "t")
    d = report.to_dict()
    assert d["target"] == "t"
    assert "recommendations" in d


def test_feature_importance_vs_target_keyed_by_every_feature_name() -> None:
    rng = np.random.default_rng(10)
    n = 400
    X = rng.normal(0, 1, (n, 4))
    y = X[:, 2] * 5.0 + rng.normal(0, 0.1, n)
    names = ["a", "b", "strong", "d"]
    importance = feature_importance_vs_target(X, y, names)
    assert set(importance) == set(names)
    assert all(v >= 0.0 for v in importance.values())
    assert importance["strong"] > importance["a"]


def test_audit_features_for_target_populates_feature_importance() -> None:
    rng = np.random.default_rng(11)
    n = 400
    X = rng.normal(0, 1, (n, 5))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.5, n)
    names = ["a", "b", "c", "d", "e"]
    report = audit_features_for_target(X, y, names, "my_target")
    assert set(report.feature_importance) == set(names)
    d = report.to_dict()
    assert "feature_importance" in d and set(d["feature_importance"]) == set(names)
