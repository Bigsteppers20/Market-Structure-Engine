"""Tests for logistic_regression.threshold_manager.ThresholdManager --
per-class decision thresholds as an alternative to plain argmax."""
from __future__ import annotations

import numpy as np
import pytest

from logistic_regression.threshold_manager import ThresholdManager


def test_default_strategy_is_argmax() -> None:
    tm = ThresholdManager(classes=["SELL", "NO_TRADE", "BUY"])
    assert tm.strategy == "argmax"


def test_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        ThresholdManager(classes=["SELL", "NO_TRADE", "BUY"], strategy="bogus")


def test_argmax_apply_picks_highest_probability() -> None:
    tm = ThresholdManager(classes=["SELL", "NO_TRADE", "BUY"], strategy="argmax")
    probs = {"SELL": 0.2, "NO_TRADE": 0.3, "BUY": 0.5}
    assert tm.apply(probs) == "BUY"


def test_optimize_finds_reasonable_per_class_thresholds() -> None:
    rng = np.random.default_rng(0)
    n = 300
    y_true_encoded = rng.integers(0, 3, n)
    # Probabilities strongly aligned with the true label -- should let
    # optimize() find thresholds that are not simply 0.05 (degenerate).
    probabilities = np.zeros((n, 3))
    for i, cls in enumerate(y_true_encoded):
        probabilities[i, cls] = 0.7 + rng.uniform(0, 0.25)
        remaining = 1.0 - probabilities[i, cls]
        others = [c for c in range(3) if c != cls]
        probabilities[i, others[0]] = remaining * 0.5
        probabilities[i, others[1]] = remaining * 0.5

    tm = ThresholdManager(classes=["SELL", "NO_TRADE", "BUY"], strategy="optimized")
    thresholds = tm.optimize(y_true_encoded, probabilities)
    assert set(thresholds) == {"SELL", "NO_TRADE", "BUY"}
    assert all(0.0 <= t <= 1.0 for t in thresholds.values())
    assert tm.thresholds == thresholds


def test_optimized_apply_uses_fitted_thresholds() -> None:
    tm = ThresholdManager(classes=["SELL", "NO_TRADE", "BUY"], strategy="optimized")
    tm.thresholds = {"SELL": 0.8, "NO_TRADE": 0.2, "BUY": 0.8}
    # Only NO_TRADE clears its (low) threshold.
    probs = {"SELL": 0.3, "NO_TRADE": 0.4, "BUY": 0.3}
    assert tm.apply(probs) == "NO_TRADE"


def test_optimized_apply_falls_back_to_argmax_when_nothing_clears() -> None:
    tm = ThresholdManager(classes=["SELL", "NO_TRADE", "BUY"], strategy="optimized")
    tm.thresholds = {"SELL": 0.9, "NO_TRADE": 0.9, "BUY": 0.9}
    probs = {"SELL": 0.3, "NO_TRADE": 0.4, "BUY": 0.3}
    assert tm.apply(probs) == "NO_TRADE"  # argmax fallback


def test_custom_strategy_with_no_thresholds_set_falls_back_to_argmax() -> None:
    tm = ThresholdManager(classes=["SELL", "NO_TRADE", "BUY"], strategy="custom")
    probs = {"SELL": 0.6, "NO_TRADE": 0.1, "BUY": 0.3}
    assert tm.apply(probs) == "SELL"


def test_custom_thresholds_applied_directly() -> None:
    tm = ThresholdManager(
        classes=["SELL", "NO_TRADE", "BUY"], strategy="custom",
        thresholds={"SELL": 0.7, "NO_TRADE": 0.1, "BUY": 0.7},
    )
    probs = {"SELL": 0.3, "NO_TRADE": 0.4, "BUY": 0.3}
    assert tm.apply(probs) == "NO_TRADE"
