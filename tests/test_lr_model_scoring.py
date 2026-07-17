"""Tests for linear_regression.model_scoring -- Phase 8 training-time
0-100 health scores, purely descriptive (never gate training/serving)."""
from __future__ import annotations

import numpy as np
import pytest

from linear_regression.model_scoring import (
    compute_model_health_scores,
    cross_validation_score,
    feature_quality_score,
    generalization_score,
    model_health_score,
    target_reliability,
)


def test_generalization_score_perfect_when_no_overfitting_gap() -> None:
    assert generalization_score(train_r2=0.8, holdout_r2=0.8) == pytest.approx(100.0)
    assert generalization_score(train_r2=0.7, holdout_r2=0.9) == pytest.approx(100.0)


def test_generalization_score_degrades_with_overfitting_gap() -> None:
    small_gap = generalization_score(train_r2=0.6, holdout_r2=0.5)
    large_gap = generalization_score(train_r2=0.95, holdout_r2=0.1)
    assert small_gap > large_gap
    assert large_gap < 50.0


def test_generalization_score_neutral_when_both_nonpositive() -> None:
    assert generalization_score(train_r2=-0.5, holdout_r2=-0.2) == 60.0


def test_cross_validation_score_neutral_when_not_run() -> None:
    assert cross_validation_score(None, None) == 60.0


def test_cross_validation_score_high_for_high_stable_mean() -> None:
    score = cross_validation_score(cv_mean_r2=0.9, cv_std_r2=0.02)
    assert score > 85.0


def test_cross_validation_score_low_for_negative_mean() -> None:
    score = cross_validation_score(cv_mean_r2=-0.5, cv_std_r2=0.1)
    assert score < 40.0


def test_cross_validation_score_penalizes_high_variance() -> None:
    stable = cross_validation_score(cv_mean_r2=0.6, cv_std_r2=0.01)
    unstable = cross_validation_score(cv_mean_r2=0.6, cv_std_r2=0.5)
    assert stable > unstable


def test_feature_quality_score_higher_for_informative_features() -> None:
    rng = np.random.default_rng(0)
    n = 500
    informative = rng.normal(0, 1, n)
    y = informative * 2.0 + rng.normal(0, 0.1, n)
    uninformative = rng.normal(0, 1, (n, 5))
    X_good = np.column_stack([informative, uninformative])
    X_bad = rng.normal(0, 1, (n, 6))  # no relationship to y at all
    good = feature_quality_score(X_good, y)
    bad = feature_quality_score(X_bad, y)
    assert good > bad


def test_feature_quality_score_penalizes_constant_features() -> None:
    rng = np.random.default_rng(1)
    n = 300
    y = rng.normal(0, 1, n)
    X_varied = rng.normal(0, 1, (n, 4))
    X_constant = np.ones((n, 4)) * 5.0
    varied = feature_quality_score(X_varied, y)
    constant = feature_quality_score(X_constant, y)
    assert varied > constant


def test_feature_quality_score_handles_zero_variance_target() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (100, 3))
    y = np.zeros(100)
    score = feature_quality_score(X, y)
    assert 0.0 <= score <= 100.0


def test_target_reliability_prefers_cv_mean_when_available() -> None:
    assert target_reliability(holdout_r2=0.2, cv_mean_r2=0.9) == pytest.approx(90.0)
    assert target_reliability(holdout_r2=0.5, cv_mean_r2=None) == pytest.approx(50.0)


def test_target_reliability_clips_negative_r2_to_zero() -> None:
    assert target_reliability(holdout_r2=-2.0, cv_mean_r2=None) == 0.0


def test_model_health_score_is_mean_of_four_components() -> None:
    score = model_health_score(generalization=80.0, cross_validation=60.0, feature_quality=100.0, target_reliability_score=40.0)
    assert score == pytest.approx(70.0)


def test_compute_model_health_scores_full_pipeline() -> None:
    rng = np.random.default_rng(3)
    n = 400
    X_train = rng.normal(0, 1, (n, 5))
    y_train = X_train[:, 0] * 2.0 + rng.normal(0, 0.2, n)
    scores = compute_model_health_scores(
        train_r2=0.85, holdout_r2=0.7, X_train=X_train, y_train=y_train,
        cv_mean_r2=0.65, cv_std_r2=0.05,
    )
    assert 0.0 <= scores.model_health_score <= 100.0
    d = scores.to_dict()
    assert set(d) == {
        "generalization_score", "cross_validation_score", "feature_quality_score",
        "target_reliability", "model_health_score",
    }


def test_compute_model_health_scores_without_cv() -> None:
    rng = np.random.default_rng(4)
    n = 300
    X_train = rng.normal(0, 1, (n, 4))
    y_train = rng.normal(0, 1, n)
    scores = compute_model_health_scores(train_r2=0.5, holdout_r2=0.4, X_train=X_train, y_train=y_train)
    assert scores.cross_validation_score == 60.0
