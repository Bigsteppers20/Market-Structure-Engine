"""Tests for model_monitor.health_score -- the pure 10-factor blending
function (MODEL HEALTH section)."""
from __future__ import annotations

import pytest

from model_monitor.health_score import HEALTH_SCORE_FACTORS, compute_health_score, severity_to_score


def test_severity_to_score_inverse_relationship() -> None:
    assert severity_to_score(0.0) == 100.0
    assert severity_to_score(1.0) == 0.0
    assert severity_to_score(0.5) == 50.0


def test_severity_to_score_clips_out_of_range() -> None:
    assert severity_to_score(-1.0) == 100.0
    assert severity_to_score(2.0) == 0.0


def _all_100():
    return compute_health_score(
        prediction_accuracy=100.0, prediction_stability=100.0, confidence_calibration=100.0,
        feature_drift=100.0, rolling_error=100.0, prediction_coverage=100.0,
        target_drift=100.0, residual_drift=100.0, market_regime_change=100.0, training_age=100.0,
    )


def test_all_factors_perfect_gives_overall_100() -> None:
    assert _all_100().overall == pytest.approx(100.0)


def test_all_factors_zero_gives_overall_0() -> None:
    breakdown = compute_health_score(
        prediction_accuracy=0.0, prediction_stability=0.0, confidence_calibration=0.0,
        feature_drift=0.0, rolling_error=0.0, prediction_coverage=0.0,
        target_drift=0.0, residual_drift=0.0, market_regime_change=0.0, training_age=0.0,
    )
    assert breakdown.overall == pytest.approx(0.0)


def test_optional_factors_default_to_neutral_70() -> None:
    breakdown = compute_health_score(
        prediction_accuracy=100.0, prediction_stability=100.0, confidence_calibration=100.0,
        feature_drift=100.0, rolling_error=100.0, prediction_coverage=100.0,
    )
    assert breakdown.target_drift == 70.0
    assert breakdown.residual_drift == 70.0
    assert breakdown.market_regime_change == 70.0
    assert breakdown.training_age == 70.0
    assert breakdown.overall < 100.0  # neutral factors still pull overall down from a perfect 100


def test_values_clipped_to_0_100_range() -> None:
    breakdown = compute_health_score(
        prediction_accuracy=150.0, prediction_stability=-20.0, confidence_calibration=100.0,
        feature_drift=100.0, rolling_error=100.0, prediction_coverage=100.0,
    )
    assert breakdown.prediction_accuracy == 100.0
    assert breakdown.prediction_stability == 0.0


def test_custom_weights_change_overall() -> None:
    weights_favor_accuracy = {k: 0.0 for k in HEALTH_SCORE_FACTORS}
    weights_favor_accuracy["prediction_accuracy"] = 1.0
    breakdown = compute_health_score(
        prediction_accuracy=90.0, prediction_stability=0.0, confidence_calibration=0.0,
        feature_drift=0.0, rolling_error=0.0, prediction_coverage=0.0,
        weights=weights_favor_accuracy,
    )
    assert breakdown.overall == pytest.approx(90.0)


def test_to_dict_contains_every_factor_and_overall() -> None:
    d = _all_100().to_dict()
    assert set(d) == set(HEALTH_SCORE_FACTORS) | {"overall"}


def test_health_score_factors_matches_default_weights_keys() -> None:
    from model_monitor.health_score import DEFAULT_WEIGHTS
    assert set(DEFAULT_WEIGHTS) == set(HEALTH_SCORE_FACTORS)
