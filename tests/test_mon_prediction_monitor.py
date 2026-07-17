"""Tests for model_monitor.prediction_monitor: PredictionSnapshot,
ResolvedPrediction, adapters, and PredictionLog (outcome resolution +
PREDICTION COVERAGE)."""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.label_generator import REGRESSION_REGISTRY, ThresholdLabelGenerator
from model_monitor.prediction_monitor import (
    PredictionLog,
    PredictionSnapshot,
    from_classification_prediction,
    from_regression_prediction,
)


def _snapshot(**overrides) -> PredictionSnapshot:
    base = dict(
        task_type="regression", model_name="m", model_version="1", feature_version="1",
        training_version="1", symbol="EUR_USD", timeframe="M5", prediction_horizon=5,
        timestamp="2026-01-01T00:00:00+00:00", decision_index=10, feature_vector=[1.0, 2.0],
        feature_names=["a", "b"], confidence=70.0, predicted_value=1.1234,
        raw_predictions={"next_close": 1.1234}, primary_target="next_close",
    )
    base.update(overrides)
    return PredictionSnapshot(**base)


def _ohlcv_df(n: int = 60) -> pd.DataFrame:
    close = np.linspace(1.10, 1.20, n)
    return pd.DataFrame({"open": close, "high": close + 0.001, "low": close - 0.001, "close": close})


# --------------------------------------------------------------------------- #
# Adapters (duck-typed, no import of linear_regression/logistic_regression)
# --------------------------------------------------------------------------- #
class _FakeRegressionPrediction:
    model_version = "1.0.0"
    feature_version = "1.0.0"
    training_version = "1.0.0"
    symbol = "EUR_USD"
    timeframe = "M5"
    prediction_horizon = 5
    timestamp = "t"
    prediction_confidence = 62.0
    raw_predictions = {"next_close": 1.1234, "next_high": 1.13}


class _FakeClassificationPrediction:
    model_version = "1.0.0"
    feature_version = "1.0.0"
    training_version = "1.0.0"
    symbol = "EUR_USD"
    timeframe = "M5"
    prediction_horizon = 5
    timestamp = "t"
    prediction_confidence = 81.0
    predicted_class = "BUY"
    class_probabilities = {"SELL": 0.1, "NO_TRADE": 0.2, "BUY": 0.7}


def test_from_regression_prediction_adapter() -> None:
    snap = from_regression_prediction(
        _FakeRegressionPrediction(), model_name="lr_m", primary_target="next_close",
        decision_index=5, feature_vector=[1.0, 2.0], feature_names=["a", "b"],
    )
    assert snap.task_type == "regression"
    assert snap.predicted_value == 1.1234
    assert snap.raw_predictions == {"next_close": 1.1234, "next_high": 1.13}


def test_from_classification_prediction_adapter() -> None:
    snap = from_classification_prediction(
        _FakeClassificationPrediction(), model_name="lgr_m", decision_index=5,
        feature_vector=[1.0, 2.0], feature_names=["a", "b"],
    )
    assert snap.task_type == "classification"
    assert snap.predicted_class == "BUY"
    assert snap.class_probabilities == {"SELL": 0.1, "NO_TRADE": 0.2, "BUY": 0.7}


# --------------------------------------------------------------------------- #
# ResolvedPrediction
# --------------------------------------------------------------------------- #
def test_regression_error_and_directional_correctness() -> None:
    from model_monitor.prediction_monitor import ResolvedPrediction

    snap = _snapshot(predicted_value=0.001)
    resolved = ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=0.002)
    assert resolved.error == pytest.approx(0.001)
    assert resolved.is_correct() is True  # same sign

    resolved_wrong_sign = ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=-0.002)
    assert resolved_wrong_sign.is_correct() is False


def test_regression_tolerance_based_correctness() -> None:
    from model_monitor.prediction_monitor import ResolvedPrediction

    snap = _snapshot(predicted_value=1.0)
    resolved = ResolvedPrediction(snapshot=snap, resolved_at="t", actual_value=1.05)
    assert resolved.is_correct(regression_tolerance=0.1) is True
    assert resolved.is_correct(regression_tolerance=0.01) is False


def test_classification_correctness_and_residual() -> None:
    from model_monitor.prediction_monitor import ResolvedPrediction

    snap = _snapshot(
        task_type="classification", predicted_value=None, predicted_class="BUY",
        class_probabilities={"SELL": 0.1, "NO_TRADE": 0.2, "BUY": 0.7},
    )
    correct = ResolvedPrediction(snapshot=snap, resolved_at="t", actual_class="BUY")
    wrong = ResolvedPrediction(snapshot=snap, resolved_at="t", actual_class="SELL")
    assert correct.is_correct() is True
    assert wrong.is_correct() is False
    assert correct.classification_residual == pytest.approx(0.3)
    assert wrong.classification_residual == pytest.approx(0.9)


def test_correctness_none_when_unresolved() -> None:
    from model_monitor.prediction_monitor import ResolvedPrediction

    snap = _snapshot()
    unresolved = ResolvedPrediction(snapshot=snap, resolved_at="t")
    assert unresolved.is_correct() is None
    assert unresolved.error is None


# --------------------------------------------------------------------------- #
# PredictionLog
# --------------------------------------------------------------------------- #
def test_log_and_coverage() -> None:
    log = PredictionLog()
    assert log.coverage() == 1.0  # vacuously "complete" when empty
    log.log(_snapshot(decision_index=10))
    assert log.coverage() == 0.0
    assert len(log) == 1


def test_resolve_regression_with_ml_pipeline_registry_function() -> None:
    df = _ohlcv_df()
    log = PredictionLog()
    log.log(_snapshot(decision_index=10, prediction_horizon=5))
    resolver = functools.partial(REGRESSION_REGISTRY["next_close"], pip_size=0.0001)
    resolved = log.resolve(df, resolver, now_iso="2026-01-02T00:00:00+00:00")
    assert len(resolved) == 1
    assert resolved[0].actual_value == pytest.approx(df["close"].iloc[15])
    assert log.coverage() == 1.0


def test_resolve_classification_with_label_generator_bound_method() -> None:
    df = _ohlcv_df()
    gen = ThresholdLabelGenerator()
    log = PredictionLog()
    log.log(_snapshot(
        task_type="classification", predicted_value=None, predicted_class="BUY",
        class_probabilities={"SELL": 0.1, "NO_TRADE": 0.2, "BUY": 0.7},
        decision_index=10, prediction_horizon=5,
    ))
    resolved = log.resolve(df, gen.label, now_iso="2026-01-02T00:00:00+00:00")
    assert len(resolved) == 1
    assert resolved[0].actual_class in ("SELL", "NO_TRADE", "BUY")


def test_resolve_leaves_unelapsed_predictions_pending() -> None:
    df = _ohlcv_df(n=12)
    log = PredictionLog()
    log.log(_snapshot(decision_index=10, prediction_horizon=5))  # end index 15 >= len(df)=12
    resolver = functools.partial(REGRESSION_REGISTRY["next_close"], pip_size=0.0001)
    resolved = log.resolve(df, resolver, now_iso="t")
    assert resolved == []
    assert len(log.pending) == 1
    assert log.coverage() == 0.0


def test_rolling_returns_most_recent_n() -> None:
    df = _ohlcv_df(n=100)
    log = PredictionLog()
    for i in range(10, 50, 5):
        log.log(_snapshot(decision_index=i, prediction_horizon=5))
    resolver = functools.partial(REGRESSION_REGISTRY["next_close"], pip_size=0.0001)
    log.resolve(df, resolver, now_iso="t")
    rolling3 = log.rolling(3)
    assert len(rolling3) == 3
    assert rolling3 == log.resolved[-3:]


def test_recent_snapshots_tracks_log_order_regardless_of_resolution() -> None:
    df = _ohlcv_df(n=100)
    log = PredictionLog()
    for i in range(10, 30, 5):
        log.log(_snapshot(decision_index=i, prediction_horizon=5))
    resolver = functools.partial(REGRESSION_REGISTRY["next_close"], pip_size=0.0001)
    log.resolve(df, resolver, now_iso="t")
    log.log(_snapshot(decision_index=90, prediction_horizon=50))  # stays pending
    recent = log.recent_snapshots(2)
    assert len(recent) == 2
    assert recent[-1].decision_index == 90
