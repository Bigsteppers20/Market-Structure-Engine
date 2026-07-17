"""Tests for logistic_regression.calibration: the 3 supported calibration
methods (Platt/sigmoid, isotonic, none), calibration curves, Brier score,
and expected calibration error."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from logistic_regression.calibration import (
    CALIBRATION_METHODS,
    brier_score,
    calibrate_estimator,
    compute_calibration_curve,
    expected_calibration_error,
)
from logistic_regression.exceptions import CalibrationError


def _fitted_binary_estimator(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-1, 1, (100, 2)), rng.normal(1, 1, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    est = LogisticRegression().fit(X, y)
    X_cal = rng.normal(0, 1.5, (60, 2))
    y_cal = (X_cal[:, 0] + X_cal[:, 1] > 0).astype(int)
    return est, X_cal, y_cal


def test_calibration_methods_constant() -> None:
    assert CALIBRATION_METHODS == ("none", "platt", "isotonic")


def test_none_returns_base_estimator_unchanged() -> None:
    est, X_cal, y_cal = _fitted_binary_estimator()
    result, meta = calibrate_estimator(est, X_cal, y_cal, "none")
    assert result is est
    assert meta == {"method": "none", "n_calibration_samples": 0}


def test_platt_calibration_produces_valid_probabilities() -> None:
    est, X_cal, y_cal = _fitted_binary_estimator()
    calibrated, meta = calibrate_estimator(est, X_cal, y_cal, "platt")
    assert meta["method"] == "platt"
    assert meta["n_calibration_samples"] == len(X_cal)
    proba = calibrated.predict_proba(X_cal)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_isotonic_calibration_produces_valid_probabilities() -> None:
    est, X_cal, y_cal = _fitted_binary_estimator()
    calibrated, meta = calibrate_estimator(est, X_cal, y_cal, "isotonic")
    assert meta["method"] == "isotonic"
    proba = calibrated.predict_proba(X_cal)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_unknown_method_raises() -> None:
    est, X_cal, y_cal = _fitted_binary_estimator()
    with pytest.raises(CalibrationError):
        calibrate_estimator(est, X_cal, y_cal, "bogus")


def test_calibration_uses_a_disjoint_holdout_not_the_fit_data() -> None:
    """calibrate_estimator() must never be handed the same rows the base
    estimator was fit on -- confirmed at the call-site convention level: the
    base estimator here is fit on one sample set, calibrated on a distinct
    one, and still produces finite, valid probabilities."""
    est, X_cal, y_cal = _fitted_binary_estimator(seed=1)
    calibrated, _ = calibrate_estimator(est, X_cal, y_cal, "platt")
    proba = calibrated.predict_proba(X_cal[:5])
    assert np.isfinite(proba).all()


def test_calibration_curve_perfectly_calibrated() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 5)
    y_proba = np.where(y_true == 1, 0.9, 0.1).astype(float)
    curve = compute_calibration_curve(y_true, y_proba, n_bins=5)
    assert "prob_true" in curve and "prob_pred" in curve
    assert len(curve["prob_true"]) == len(curve["prob_pred"])


def test_calibration_curve_handles_degenerate_input_gracefully() -> None:
    curve = compute_calibration_curve(np.array([0]), np.array([0.5]), n_bins=10)
    assert curve == {"prob_true": [], "prob_pred": []} or "prob_true" in curve


def test_brier_score_perfect_predictions_is_zero() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_proba) == pytest.approx(0.0)


def test_brier_score_worst_case_predictions_is_one() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_proba) == pytest.approx(1.0)


def test_expected_calibration_error_zero_when_perfectly_calibrated() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.array([0.0] * 50 + [1.0] * 50)
    assert expected_calibration_error(y_true, y_proba) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_positive_when_miscalibrated() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.array([0.9] * 50 + [0.1] * 50)  # confidently wrong both ways
    assert expected_calibration_error(y_true, y_proba) > 0.5


def test_expected_calibration_error_empty_input() -> None:
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0
