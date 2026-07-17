"""Tests for linear_regression.regression_model."""
from __future__ import annotations

import numpy as np
import pytest

from linear_regression.exceptions import ModelNotTrainedError, UnsupportedModelTypeError
from linear_regression.regression_model import RegressionModel


@pytest.fixture()
def linear_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    true_coef = np.array([2.0, -1.0, 0.5, 0.0])
    y = X @ true_coef + rng.normal(scale=0.01, size=200)
    return X, y


def test_fit_predict_recovers_linear_relationship(linear_data) -> None:
    X, y = linear_data
    model = RegressionModel("linear", n_bootstrap=0)
    model.fit(X, y, ["y"])
    preds = model.predict(X).ravel()
    assert np.corrcoef(preds, y)[0, 1] > 0.99


def test_unsupported_model_type_raises() -> None:
    with pytest.raises(UnsupportedModelTypeError):
        RegressionModel("not_a_real_model")


def test_predict_before_fit_raises(linear_data) -> None:
    X, _ = linear_data
    model = RegressionModel("linear")
    with pytest.raises(ModelNotTrainedError):
        model.predict(X)


def test_multi_output_regression(linear_data) -> None:
    X, y = linear_data
    y2 = np.column_stack([y, -y])
    model = RegressionModel("ridge", n_bootstrap=0, alpha=0.5)
    model.fit(X, y2, ["a", "b"])
    preds = model.predict(X)
    assert preds.shape == (200, 2)
    assert model.n_outputs_ == 2


def test_bootstrap_ensemble_produces_uncertainty(linear_data) -> None:
    X, y = linear_data
    model = RegressionModel("linear", n_bootstrap=15, random_state=1)
    model.fit(X, y, ["y"])
    point, std = model.predict_with_uncertainty(X[:5])
    assert point.shape == (5, 1)
    assert std is not None
    assert std.shape == (5, 1)
    assert (std >= 0).all()


def test_no_bootstrap_means_no_uncertainty(linear_data) -> None:
    X, y = linear_data
    model = RegressionModel("linear", n_bootstrap=0)
    model.fit(X, y, ["y"])
    _, std = model.predict_with_uncertainty(X[:5])
    assert std is None


def test_feature_importance_reflects_true_coefficients(linear_data) -> None:
    X, y = linear_data
    model = RegressionModel("linear", n_bootstrap=0)
    model.fit(X, y, ["y"])
    importance = model.feature_importance(["f0", "f1", "f2", "f3"])
    assert importance is not None
    # f0 has the largest true coefficient magnitude (2.0) -> largest importance
    assert importance["f0"] == max(importance.values())
    # f3 has true coefficient 0 -> smallest importance
    assert importance["f3"] == min(importance.values())


def test_ridge_lasso_elasticnet_all_fit(linear_data) -> None:
    X, y = linear_data
    for model_type in ("ridge", "lasso", "elasticnet"):
        model = RegressionModel(model_type, n_bootstrap=0, alpha=0.1)
        model.fit(X, y, ["y"])
        preds = model.predict(X)
        assert np.isfinite(preds).all()
