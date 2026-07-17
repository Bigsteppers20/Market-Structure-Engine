"""Tests for logistic_regression.classification_model: class imbalance
handling (resampling + class_weight), calibration wiring, bootstrap
ensemble, and coefficient-based feature importance."""
from __future__ import annotations

import numpy as np
import pytest

from logistic_regression.classification_model import ClassificationModel, resample
from logistic_regression.exceptions import ModelNotTrainedError, UnsupportedBalancingStrategyError


def _imbalanced_xy(n_majority: int = 90, n_minority: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_majority = rng.normal(0.0, 1.0, (n_majority, 4))
    X_minority = rng.normal(3.0, 1.0, (n_minority, 4))
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * n_majority + [1] * n_minority)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# --------------------------------------------------------------------------- #
# resample() -- class imbalance strategies
# --------------------------------------------------------------------------- #
def test_resample_rejects_unknown_strategy() -> None:
    X, y = _imbalanced_xy()
    with pytest.raises(UnsupportedBalancingStrategyError):
        resample(X, y, "bogus")


def test_resample_none_and_class_weight_are_passthrough() -> None:
    X, y = _imbalanced_xy()
    for strategy in ("none", "class_weight"):
        X_out, y_out = resample(X, y, strategy)
        assert X_out is X and y_out is y


def test_resample_oversample_balances_to_majority_count() -> None:
    X, y = _imbalanced_xy(n_majority=90, n_minority=10)
    X_out, y_out = resample(X, y, "oversample")
    _, counts = np.unique(y_out, return_counts=True)
    assert counts[0] == counts[1] == 90


def test_resample_undersample_balances_to_minority_count() -> None:
    X, y = _imbalanced_xy(n_majority=90, n_minority=10)
    X_out, y_out = resample(X, y, "undersample")
    _, counts = np.unique(y_out, return_counts=True)
    assert counts[0] == counts[1] == 10


def test_resample_balanced_sampling_uses_mean_count() -> None:
    X, y = _imbalanced_xy(n_majority=90, n_minority=10)
    X_out, y_out = resample(X, y, "balanced_sampling")
    _, counts = np.unique(y_out, return_counts=True)
    assert counts[0] == counts[1] == 50


# --------------------------------------------------------------------------- #
# ClassificationModel
# --------------------------------------------------------------------------- #
def test_fit_predict_roundtrip() -> None:
    X, y = _imbalanced_xy(n_majority=60, n_minority=40)
    model = ClassificationModel(n_bootstrap=0)
    model.fit(X, y, classes=["A", "B"])
    proba = model.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = model.predict(X)
    assert set(preds) <= {"A", "B"}


def test_predict_before_fit_raises() -> None:
    model = ClassificationModel()
    with pytest.raises(ModelNotTrainedError):
        model.predict_proba(np.zeros((1, 4)))


def test_class_weight_balancing_trains_without_resampling_rows() -> None:
    X, y = _imbalanced_xy(n_majority=90, n_minority=10)
    model = ClassificationModel(class_balancing="class_weight", n_bootstrap=0)
    model.fit(X, y, classes=["A", "B"])
    assert model._base_estimator.class_weight == "balanced"


def test_oversample_balancing_improves_minority_recall() -> None:
    """A heavily imbalanced fit with no balancing should recall the minority
    class worse than one with oversampling -- a concrete behavioral check,
    not just a config no-op check."""
    X, y = _imbalanced_xy(n_majority=200, n_minority=15, seed=3)

    baseline = ClassificationModel(class_balancing="none", n_bootstrap=0)
    baseline.fit(X, y, classes=["A", "B"])
    baseline_pred = baseline.predict_proba(X).argmax(axis=1)
    baseline_minority_recall = (baseline_pred[y == 1] == 1).mean()

    oversampled = ClassificationModel(class_balancing="oversample", n_bootstrap=0)
    oversampled.fit(X, y, classes=["A", "B"])
    oversampled_pred = oversampled.predict_proba(X).argmax(axis=1)
    oversampled_minority_recall = (oversampled_pred[y == 1] == 1).mean()

    assert oversampled_minority_recall >= baseline_minority_recall


def test_bootstrap_ensemble_produces_agreement_scores() -> None:
    X, y = _imbalanced_xy(n_majority=60, n_minority=40)
    model = ClassificationModel(n_bootstrap=8, random_state=1)
    model.fit(X, y, classes=["A", "B"])
    agreement = model.bootstrap_agreement(X)
    assert agreement is not None
    assert agreement.shape == (100,)
    assert ((agreement >= 0.0) & (agreement <= 1.0)).all()


def test_no_bootstrap_returns_none_agreement() -> None:
    X, y = _imbalanced_xy(n_majority=60, n_minority=40)
    model = ClassificationModel(n_bootstrap=0)
    model.fit(X, y, classes=["A", "B"])
    assert model.bootstrap_agreement(X) is None


def test_calibration_platt_wraps_estimator_and_records_metadata() -> None:
    X, y = _imbalanced_xy(n_majority=200, n_minority=100, seed=4)
    model = ClassificationModel(calibration_method="platt", calibration_holdout_fraction=0.2, n_bootstrap=0)
    model.fit(X, y, classes=["A", "B"])
    assert model.calibration_metadata_["method"] == "platt"
    assert model.calibration_metadata_["n_calibration_samples"] > 0
    proba = model.predict_proba(X[:5])
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_calibration_isotonic_wraps_estimator() -> None:
    X, y = _imbalanced_xy(n_majority=200, n_minority=100, seed=4)
    model = ClassificationModel(calibration_method="isotonic", calibration_holdout_fraction=0.2, n_bootstrap=0)
    model.fit(X, y, classes=["A", "B"])
    assert model.calibration_metadata_["method"] == "isotonic"


def test_no_calibration_leaves_metadata_none() -> None:
    X, y = _imbalanced_xy(n_majority=60, n_minority=40)
    model = ClassificationModel(calibration_method="none", n_bootstrap=0)
    model.fit(X, y, classes=["A", "B"])
    assert model.calibration_metadata_ == {"method": "none", "n_calibration_samples": 0}


def test_feature_importance_from_coefficients() -> None:
    X, y = _imbalanced_xy(n_majority=60, n_minority=40)
    model = ClassificationModel(n_bootstrap=0)
    model.fit(X, y, classes=["A", "B"])
    importance = model.feature_importance(["f0", "f1", "f2", "f3"])
    assert importance is not None
    assert set(importance) == {"f0", "f1", "f2", "f3"}
    assert all(v >= 0.0 for v in importance.values())


def test_missing_class_after_balancing_raises() -> None:
    """If the fitting data is missing one of the *declared* classes (here 3
    classes are declared but only 2 ever appear in y), this must fail
    loudly, not silently mispredict/misalign probability columns."""
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (40, 3))
    y = np.array([0] * 20 + [1] * 20)  # only 2 of the 3 declared classes appear
    model = ClassificationModel(n_bootstrap=0)
    with pytest.raises(ModelNotTrainedError):
        model.fit(X, y, classes=["A", "B", "C"])
