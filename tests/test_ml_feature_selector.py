"""Tests for ml_pipeline.feature_selector."""
from __future__ import annotations

import numpy as np
import pytest

from ml_pipeline.feature_selector import FeatureSelector


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    n = 300
    useful = rng.normal(size=(n, 2))
    noise = rng.normal(scale=0.001, size=(n, 1))  # near-zero variance
    duplicate = useful[:, [0]] + rng.normal(scale=1e-6, size=(n, 1))  # near-perfectly correlated
    X = np.hstack([useful, noise, duplicate])
    names = ["useful_0", "useful_1", "near_constant", "duplicate_of_0"]
    y = useful[:, 0] * 2.0 + useful[:, 1] * -1.0 + rng.normal(scale=0.01, size=n)
    return X, y, names


def test_variance_selector_drops_near_constant_column(regression_data) -> None:
    X, y, names = regression_data
    sel = FeatureSelector("variance", variance_threshold=1e-4)
    X_out, kept = sel.fit_transform(X, y, names)
    assert "near_constant" not in kept
    assert X_out.shape[1] == len(kept)


def test_correlation_selector_drops_duplicate(regression_data) -> None:
    X, y, names = regression_data
    sel = FeatureSelector("correlation", correlation_threshold=0.99)
    X_out, kept = sel.fit_transform(X, None, names)
    assert "useful_0" in kept
    assert "duplicate_of_0" not in kept  # dropped as the later-encountered near-duplicate


def test_kbest_selector_keeps_requested_k(regression_data) -> None:
    X, y, names = regression_data
    sel = FeatureSelector("kbest", target_type="regression", k=2)
    X_out, kept = sel.fit_transform(X, y, names)
    assert len(kept) == 2
    assert X_out.shape[1] == 2
    # f_regression is univariate (doesn't penalize redundancy), so it may
    # keep duplicate_of_0 alongside useful_0 -- but it must always exclude
    # the near-constant noise column, which carries no signal at all.
    assert "near_constant" not in kept


def test_mutual_info_selector_runs_regression_and_classification(regression_data) -> None:
    X, y, names = regression_data
    sel = FeatureSelector("mutual_info", target_type="regression", k=2, random_state=0)
    _, kept = sel.fit_transform(X, y, names)
    assert len(kept) == 2

    y_cls = (y > np.median(y)).astype(int)
    sel_cls = FeatureSelector("mutual_info", target_type="classification", k=2, random_state=0)
    _, kept_cls = sel_cls.fit_transform(X, y_cls, names)
    assert len(kept_cls) == 2


def test_rfe_selector_keeps_requested_k(regression_data) -> None:
    X, y, names = regression_data
    sel = FeatureSelector("rfe", target_type="regression", k=2)
    _, kept = sel.fit_transform(X, y, names)
    assert len(kept) == 2


def test_transform_reorders_defensively(regression_data) -> None:
    X, y, names = regression_data
    sel = FeatureSelector("variance", variance_threshold=1e-4).fit(X, y, names)
    shuffled = list(reversed(names))
    X_shuffled = X[:, ::-1]
    X_out, kept = sel.transform(X_shuffled, shuffled)
    X_out_direct, kept_direct = sel.transform(X, names)
    assert kept == kept_direct
    np.testing.assert_allclose(X_out, X_out_direct)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        FeatureSelector("bogus")


def test_transform_before_fit_raises(regression_data) -> None:
    X, _, names = regression_data
    sel = FeatureSelector("variance")
    with pytest.raises(RuntimeError):
        sel.transform(X, names)
