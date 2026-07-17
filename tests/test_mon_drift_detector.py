"""Tests for model_monitor.drift_detector: market regime classification,
regime drift, target/residual distribution drift, and the DriftDetector
orchestrator (FEATURE DRIFT + MARKET REGIME DETECTION sections)."""
from __future__ import annotations

import numpy as np
import pytest

from model_monitor.drift_detector import (
    DriftDetector,
    RegimeSnapshot,
    classify_regime,
    detect_distribution_shift,
    detect_regime_drift,
)
from model_monitor.exceptions import InsufficientDataError

NAMES = ["f0", "f1", "f2"]


# --------------------------------------------------------------------------- #
# classify_regime
# --------------------------------------------------------------------------- #
def test_classify_regime_trending_up() -> None:
    state = {"trend_direction": 1.0, "trend_strength": 0.8}
    assert classify_regime(state).trend_state == "trending_up"


def test_classify_regime_trending_down() -> None:
    state = {"trend_direction": -1.0, "trend_strength": 0.9}
    assert classify_regime(state).trend_state == "trending_down"


def test_classify_regime_ranging_when_strength_low() -> None:
    state = {"trend_direction": 1.0, "trend_strength": 0.1}
    assert classify_regime(state).trend_state == "ranging"


def test_classify_regime_volatility_states() -> None:
    assert classify_regime({"vol_expansion": 1.0}).volatility_state == "high_volatility"
    assert classify_regime({"vol_compression": 1.0}).volatility_state == "low_volatility"
    assert classify_regime({}).volatility_state == "normal"


def test_classify_regime_session() -> None:
    assert classify_regime({"session_is_london": 1.0}).session == "london"
    assert classify_regime({}).session == "none"


def test_classify_regime_liquidity() -> None:
    assert classify_regime({"spread_spike": 1.0}).liquidity_state == "widened_spread"
    assert classify_regime({"spread_percentile": 0.95}).liquidity_state == "widened_spread"
    assert classify_regime({"spread_percentile": 0.1}).liquidity_state == "normal"


def test_classify_regime_missing_keys_default_neutral() -> None:
    snapshot = classify_regime({})
    assert snapshot.trend_state == "ranging"
    assert snapshot.volatility_state == "normal"
    assert snapshot.session == "none"
    assert snapshot.liquidity_state == "normal"


# --------------------------------------------------------------------------- #
# detect_regime_drift
# --------------------------------------------------------------------------- #
def test_regime_drift_needs_nonempty_inputs() -> None:
    with pytest.raises(InsufficientDataError):
        detect_regime_drift([], [])


def test_regime_drift_identical_distributions_no_shift() -> None:
    snaps = [RegimeSnapshot("ranging", "normal", "london", "normal")] * 50
    report = detect_regime_drift(snaps, snaps)
    assert report.overall_shift == pytest.approx(0.0)
    assert not report.shift_detected


def test_regime_drift_detects_full_shift() -> None:
    baseline = [RegimeSnapshot("ranging", "normal", "london", "normal")] * 50
    current = [RegimeSnapshot("trending_up", "high_volatility", "newyork", "widened_spread")] * 50
    report = detect_regime_drift(baseline, current)
    assert report.overall_shift == pytest.approx(1.0)
    assert report.shift_detected
    assert set(report.differing_dimensions) == {"trend_state", "volatility_state", "session", "liquidity_state"}


def test_regime_drift_dominant_current_regime() -> None:
    baseline = [RegimeSnapshot("ranging", "normal", "london", "normal")]
    current = [RegimeSnapshot("trending_up", "high_volatility", "newyork", "normal")] * 3 + \
              [RegimeSnapshot("ranging", "normal", "london", "normal")]
    report = detect_regime_drift(baseline, current)
    assert report.dominant_current_regime["trend_state"] == "trending_up"


# --------------------------------------------------------------------------- #
# detect_distribution_shift
# --------------------------------------------------------------------------- #
def test_distribution_shift_needs_two_values() -> None:
    with pytest.raises(InsufficientDataError):
        detect_distribution_shift(np.array([1.0]), np.array([1.0, 2.0]))


def test_distribution_shift_identical_series_low_severity() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 500)
    current = rng.normal(0, 1, 200)
    shift = detect_distribution_shift(baseline, current)
    assert shift.severity < 0.2


def test_distribution_shift_detects_mean_shift() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 500)
    current = rng.normal(5, 1, 200)
    shift = detect_distribution_shift(baseline, current)
    assert shift.mean_shift > 3.0
    assert shift.severity > 0.5


# --------------------------------------------------------------------------- #
# DriftDetector orchestrator
# --------------------------------------------------------------------------- #
def test_orchestrator_detect_before_fit_raises() -> None:
    with pytest.raises(InsufficientDataError):
        DriftDetector().detect(np.zeros((10, 3)), NAMES)


def test_orchestrator_feature_only_when_no_optional_baselines() -> None:
    rng = np.random.default_rng(3)
    X_train = rng.normal(0, 1, (300, 3))
    detector = DriftDetector().fit_baseline(X_train, NAMES)
    report = detector.detect(rng.normal(0, 1, (50, 3)), NAMES)
    assert report.regime_drift is None
    assert report.target_drift is None
    assert report.residual_drift is None
    assert report.overall_severity == report.feature_drift.overall_severity


def test_orchestrator_combines_every_component_when_supplied() -> None:
    rng = np.random.default_rng(4)
    X_train = rng.normal(0, 1, (300, 3))
    train_regimes = [RegimeSnapshot("ranging", "normal", "london", "normal")] * 50
    train_targets = rng.normal(0, 1, 300)
    train_residuals = rng.normal(0, 0.1, 300)
    detector = DriftDetector().fit_baseline(
        X_train, NAMES, regime_snapshots=train_regimes, target_values=train_targets, residual_values=train_residuals,
    )

    live_regimes = [RegimeSnapshot("trending_up", "high_volatility", "newyork", "widened_spread")] * 50
    live_targets = rng.normal(5, 1, 50)
    live_residuals = rng.normal(2, 0.1, 50)
    report = detector.detect(
        rng.normal(3, 1, (50, 3)), NAMES, current_regimes=live_regimes,
        current_targets=live_targets, current_residuals=live_residuals,
    )
    assert report.regime_drift is not None and report.regime_drift.shift_detected
    assert report.target_drift is not None and report.target_drift.severity > 0.3
    assert report.residual_drift is not None and report.residual_drift.severity > 0.3
    assert report.overall_severity > 0.3


def test_drift_report_to_dict_serializable() -> None:
    rng = np.random.default_rng(5)
    X_train = rng.normal(0, 1, (300, 3))
    detector = DriftDetector().fit_baseline(X_train, NAMES)
    report = detector.detect(rng.normal(0, 1, (50, 3)), NAMES)
    d = report.to_dict()
    assert set(d) == {"feature_drift", "regime_drift", "target_drift", "residual_drift", "overall_severity", "severity_label"}
    assert d["regime_drift"] is None
