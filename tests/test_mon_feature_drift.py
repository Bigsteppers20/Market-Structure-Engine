"""Tests for model_monitor.feature_drift.FeatureDriftDetector -- distribution
drift, mean/variance shift, correlation drift, missing feature drift, and
outlier frequency (FEATURE DRIFT section)."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.exceptions import InsufficientDataError
from model_monitor.feature_drift import FeatureDriftDetector


NAMES = ["f0", "f1", "f2", "f3", "f4"]


def _detector(seed: int = 0) -> FeatureDriftDetector:
    rng = np.random.default_rng(seed)
    X_train = rng.normal(0, 1, (500, 5))
    return FeatureDriftDetector().fit_baseline(X_train, NAMES), rng


def test_detect_before_fit_raises() -> None:
    detector = FeatureDriftDetector()
    with pytest.raises(InsufficientDataError):
        detector.detect(np.zeros((10, 5)), NAMES)


def test_fit_baseline_needs_at_least_two_rows() -> None:
    with pytest.raises(InsufficientDataError):
        FeatureDriftDetector().fit_baseline(np.zeros((1, 5)), NAMES)


def test_identical_distribution_is_low_severity() -> None:
    detector, rng = _detector()
    X_live = rng.normal(0, 1, (200, 5))
    report = detector.detect(X_live, NAMES)
    assert report.severity_label == "low"
    assert report.overall_severity < 0.15


def test_mean_shift_detected() -> None:
    detector, rng = _detector()
    X_live = rng.normal(5, 1, (200, 5))
    report = detector.detect(X_live, NAMES)
    assert report.overall_severity > 0.3
    for m in report.per_feature.values():
        assert m.mean_shift > 1.0


def test_variance_shift_detected() -> None:
    detector, rng = _detector()
    X_live = rng.normal(0, 5, (200, 5))
    report = detector.detect(X_live, NAMES)
    for m in report.per_feature.values():
        assert m.variance_shift > 1.0


def test_most_drifted_ranks_the_shifted_feature_first() -> None:
    rng = np.random.default_rng(1)
    X_train = rng.normal(0, 1, (500, 3))
    detector = FeatureDriftDetector(top_n=3).fit_baseline(X_train, ["a", "b", "c"])
    X_live = X_train[-100:].copy()
    X_live[:, 1] += 8.0  # only feature "b" drifts
    report = detector.detect(X_live, ["a", "b", "c"])
    assert report.most_drifted[0] == "b"


def test_missing_feature_flagged_and_maximally_severe() -> None:
    detector, rng = _detector()
    X_live = rng.normal(0, 1, (100, 4))
    report = detector.detect(X_live, NAMES[:4])
    assert report.missing_features == ["f4"]
    assert report.per_feature["f4"].drift_score == 1.0
    assert report.per_feature["f4"].missing_frequency == 1.0


def test_unexpected_feature_flagged() -> None:
    detector, rng = _detector()
    X_live = rng.normal(0, 1, (100, 6))
    report = detector.detect(X_live, NAMES + ["f5"])
    assert report.unexpected_features == ["f5"]
    assert "f5" not in report.per_feature


def test_outlier_frequency_high_when_live_data_is_extreme() -> None:
    detector, rng = _detector()
    X_live = np.full((50, 5), 10.0)  # 10 std-devs out for every feature
    report = detector.detect(X_live, NAMES)
    for m in report.per_feature.values():
        assert m.outlier_frequency == 1.0


def test_valid_mask_drives_missing_frequency() -> None:
    detector, rng = _detector()
    X_live = rng.normal(0, 1, (50, 5))
    valid_mask = np.ones((50, 5), dtype=bool)
    valid_mask[:, 2] = False  # feature f2 invalid for every live row
    report = detector.detect(X_live, NAMES, valid_mask=valid_mask)
    assert report.per_feature["f2"].missing_frequency == 1.0
    assert report.per_feature["f0"].missing_frequency == 0.0


def test_correlation_drift_detected_when_relationships_change() -> None:
    rng = np.random.default_rng(2)
    n = 1000
    x0 = rng.normal(0, 1, n)
    x1 = x0 + rng.normal(0, 0.01, n)  # near-perfectly correlated in training
    X_train = np.column_stack([x0, x1])
    detector = FeatureDriftDetector().fit_baseline(X_train, ["a", "b"])

    x0_live = rng.normal(0, 1, 200)
    x1_live = rng.normal(0, 1, 200)  # uncorrelated live
    X_live = np.column_stack([x0_live, x1_live])
    report = detector.detect(X_live, ["a", "b"])
    assert report.correlation_drift > 0.2


def test_report_to_dict_serializable() -> None:
    detector, rng = _detector()
    report = detector.detect(rng.normal(0, 1, (50, 5)), NAMES)
    d = report.to_dict()
    assert set(d) == {
        "per_feature", "most_drifted", "missing_features", "unexpected_features",
        "correlation_drift", "overall_severity", "severity_label", "n_live_samples",
    }
    assert d["n_live_samples"] == 50
