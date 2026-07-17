"""Tests for linear_regression.confidence -- must be independent of the
predicted value, by construction (the function doesn't even accept it)."""
from __future__ import annotations

import inspect

import pytest

from linear_regression.confidence import compute_confidence


def test_compute_confidence_signature_has_no_predicted_value_parameter() -> None:
    params = set(inspect.signature(compute_confidence).parameters)
    assert "prediction" not in params
    assert "predicted_value" not in params
    assert "value" not in params
    assert "point_estimate" not in params


def test_confidence_bounded_0_100() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.8,
        feature_completeness_fraction=0.9, mean_abs_z_score=0.5, ensemble_std=0.005,
    )
    assert 0.0 <= breakdown.overall <= 100.0


def test_good_model_scores_higher_than_bad_model() -> None:
    good = compute_confidence(
        residual_std=0.001, target_std=0.02, test_r2=0.9,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.1, ensemble_std=0.001,
    )
    bad = compute_confidence(
        residual_std=0.019, target_std=0.02, test_r2=-1.0,
        feature_completeness_fraction=0.3, mean_abs_z_score=5.0, ensemble_std=0.018,
    )
    assert good.overall > bad.overall


def test_negative_r2_clips_to_zero_historical_accuracy() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=-3.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.0, ensemble_std=0.0,
    )
    assert breakdown.historical_accuracy == 0.0


def test_missing_ensemble_std_is_neutral_not_penalized() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.0, ensemble_std=None,
    )
    assert breakdown.prediction_stability == 50.0


def test_far_from_training_distribution_lowers_score() -> None:
    near = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.1, ensemble_std=0.005,
    )
    far = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=8.0, ensemble_std=0.005,
    )
    assert near.distribution_distance > far.distribution_distance
    assert near.overall > far.overall


def test_breakdown_to_dict_has_all_factors() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.5, ensemble_std=0.005,
    )
    d = breakdown.to_dict()
    assert set(d) == {
        "residual_quality", "historical_accuracy", "feature_completeness",
        "distribution_distance", "prediction_stability", "cv_stability", "interval_width", "overall",
    }


def test_cv_stability_neutral_when_cv_not_run() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.0, ensemble_std=0.0,
    )
    assert breakdown.cv_stability == 60.0


def test_cv_stability_high_for_high_stable_mean_r2() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.0, ensemble_std=0.0,
        cv_mean_r2=0.9, cv_std_r2=0.01,
    )
    assert breakdown.cv_stability > 80.0


def test_cv_stability_low_for_high_variance_folds() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.0, ensemble_std=0.0,
        cv_mean_r2=0.5, cv_std_r2=0.5,
    )
    assert breakdown.cv_stability < 50.0


def test_interval_width_neutral_when_not_supplied() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.0, ensemble_std=0.0,
    )
    assert breakdown.interval_width == 60.0


def test_interval_width_high_for_narrow_relative_interval() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.0, ensemble_std=0.0,
        interval_width_fraction=0.01,
    )
    assert breakdown.interval_width > 90.0


def test_interval_width_low_for_wide_relative_interval() -> None:
    breakdown = compute_confidence(
        residual_std=0.01, target_std=0.02, test_r2=0.5,
        feature_completeness_fraction=1.0, mean_abs_z_score=0.0, ensemble_std=0.0,
        interval_width_fraction=2.0,
    )
    assert breakdown.interval_width == 0.0


def test_compute_confidence_never_accepts_a_predicted_value_under_any_name() -> None:
    """Regression guard for the Phase 7 addition specifically: the new
    interval-width factor must take a pre-normalized ratio, never the raw
    predicted value itself, under any parameter name."""
    params = set(inspect.signature(compute_confidence).parameters)
    for forbidden in ("prediction", "predicted_value", "value", "point_estimate", "prediction_interval_width"):
        assert forbidden not in params
