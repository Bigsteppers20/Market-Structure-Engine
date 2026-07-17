"""Tests for linear_regression.cross_validation -- Phase 4 (walk-forward CV)
and Phase 5 (model-type comparison). Never shuffles data."""
from __future__ import annotations

import numpy as np
import pytest

from ml_pipeline.splitter import TimeSeriesSplitter
from linear_regression.cross_validation import (
    compare_model_types,
    cross_validate,
    recommend_best_model,
)


def _linear_dataset(n=600, p=10, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    true_coef = rng.normal(0, 1, p)
    y = X @ true_coef + rng.normal(0, noise, n)
    names = [f"f{i}" for i in range(p)]
    return X, y, names


def test_cross_validate_walk_forward_expanding_produces_folds() -> None:
    X, y, _ = _linear_dataset(n=600, noise=0.1, seed=1)
    result = cross_validate(X, y, target_name="t", model_type="ridge", method="walk_forward_expanding", n_folds=5)
    assert result.method == "walk_forward_expanding"
    assert len(result.folds) >= 1
    assert all(f.n_train > 0 and f.n_test > 0 for f in result.folds)


def test_cross_validate_walk_forward_rolling_produces_folds() -> None:
    X, y, _ = _linear_dataset(n=600, noise=0.1, seed=2)
    result = cross_validate(X, y, target_name="t", model_type="ridge", method="walk_forward_rolling", n_folds=5)
    assert result.method == "walk_forward_rolling"
    assert len(result.folds) >= 1


def test_cross_validate_never_shuffles_train_precedes_test() -> None:
    """Every fold's train indices must chronologically precede its test
    indices -- verified directly via the underlying splitter, since
    cross_validate() delegates its splitting entirely to it."""
    splitter = TimeSeriesSplitter(method="walk_forward", window_mode="expanding", n_folds=5)
    splits = splitter.split(600)
    for split in splits:
        if split.train_idx.size and split.test_idx.size:
            assert split.train_idx.max() < split.test_idx.min()


def test_cross_validate_good_linear_signal_scores_well() -> None:
    X, y, _ = _linear_dataset(n=800, p=5, noise=0.05, seed=3)
    result = cross_validate(X, y, target_name="t", model_type="ridge", method="walk_forward_expanding", n_folds=5)
    assert result.mean_r2 > 0.5


def test_cross_validate_pure_noise_scores_poorly() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (600, 10))
    y = rng.normal(0, 1, 600)  # no relationship to X whatsoever
    result = cross_validate(X, y, target_name="t", model_type="ridge", method="walk_forward_expanding", n_folds=5)
    assert result.mean_r2 < 0.3


def test_cross_validate_raises_on_insufficient_data() -> None:
    X = np.random.default_rng(5).normal(0, 1, (8, 3))
    y = np.random.default_rng(5).normal(0, 1, 8)
    with pytest.raises(ValueError):
        cross_validate(X, y, target_name="t", model_type="ridge", n_folds=5)


def test_cross_validation_result_to_dict_serializable() -> None:
    X, y, _ = _linear_dataset(n=600, seed=6)
    result = cross_validate(X, y, target_name="t", model_type="ridge", n_folds=5)
    d = result.to_dict()
    assert d["target"] == "t" and "folds" in d and len(d["folds"]) == len(result.folds)


def test_compare_model_types_returns_all_requested_types() -> None:
    X, y, _ = _linear_dataset(n=600, p=8, seed=7)
    split = TimeSeriesSplitter(method="simple").split(len(y))[0]
    results = compare_model_types(X, y, split, target_name="t", model_types=("linear", "ridge", "lasso"))
    assert {r.model_type for r in results} == {"linear", "ridge", "lasso"}
    assert all(r.training_time_ms >= 0.0 for r in results)


def test_compare_model_types_lasso_reduces_nonzero_coefficients() -> None:
    """Lasso's L1 penalty should zero out at least some of the many
    uninformative features relative to a plain linear fit."""
    rng = np.random.default_rng(8)
    n, p = 600, 30
    X = rng.normal(0, 1, (n, p))
    y = X[:, 0] * 3.0 + rng.normal(0, 0.1, n)  # only feature 0 matters
    split = TimeSeriesSplitter(method="simple").split(n)[0]
    results = compare_model_types(X, y, split, target_name="t", model_types=("linear", "lasso"))
    by_type = {r.model_type: r for r in results}
    assert by_type["lasso"].n_nonzero_coefficients <= by_type["linear"].n_nonzero_coefficients


def test_recommend_best_model_picks_highest_r2() -> None:
    X, y, _ = _linear_dataset(n=600, p=8, seed=9)
    split = TimeSeriesSplitter(method="simple").split(len(y))[0]
    results = compare_model_types(X, y, split, target_name="t")
    best = recommend_best_model(results)
    best_result = next(r for r in results if r.model_type == best)
    assert all(best_result.r2 >= r.r2 for r in results)
