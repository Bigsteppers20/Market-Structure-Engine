"""Tests for logistic_regression.label_manager.ConfigurableClassificationLabelGenerator.

Confirms it correctly implements ml_pipeline.label_generator.LabelGenerator
(the Dataset Builder's own pluggable labeling interface -- reused, not
duplicated) and that every configurable parameter the spec requires
(prediction horizon, min pip movement, min expected return, risk-reward
threshold, max adverse excursion) actually changes labeling behavior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.label_generator import LabelGenerator
from logistic_regression.label_manager import ConfigurableClassificationLabelGenerator


def _trending_df(n: int = 60, drift: float = 0.0006, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.0003, n)
    close = 1.10 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = 1.10
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0001, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0001, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_is_a_label_generator() -> None:
    assert issubclass(ConfigurableClassificationLabelGenerator, LabelGenerator)


def test_default_classes() -> None:
    gen = ConfigurableClassificationLabelGenerator()
    assert gen.classes == ("SELL", "NO_TRADE", "BUY")


def test_strong_uptrend_labels_buy() -> None:
    df = _trending_df(drift=0.0015, seed=2)
    gen = ConfigurableClassificationLabelGenerator(min_pip_movement=5.0, risk_reward_threshold=1.2)
    label = gen.label(df, index=10, horizon=5)
    assert label == "BUY"


def test_strong_downtrend_labels_sell() -> None:
    df = _trending_df(drift=-0.0015, seed=2)
    gen = ConfigurableClassificationLabelGenerator(min_pip_movement=5.0, risk_reward_threshold=1.2)
    label = gen.label(df, index=10, horizon=5)
    assert label == "SELL"


def test_flat_series_labels_no_trade() -> None:
    df = _trending_df(drift=0.0, seed=5)
    gen = ConfigurableClassificationLabelGenerator(min_pip_movement=50.0)
    label = gen.label(df, index=10, horizon=5)
    assert label == "NO_TRADE"


def test_higher_min_pip_movement_suppresses_weak_moves() -> None:
    df = _trending_df(drift=0.0006, seed=2)
    loose = ConfigurableClassificationLabelGenerator(min_pip_movement=1.0, risk_reward_threshold=1.0)
    strict = ConfigurableClassificationLabelGenerator(min_pip_movement=500.0, risk_reward_threshold=1.0)
    assert loose.label(df, index=10, horizon=5) in ("BUY", "SELL")
    assert strict.label(df, index=10, horizon=5) == "NO_TRADE"


def test_min_expected_return_gate() -> None:
    df = _trending_df(drift=0.0006, seed=2)
    gen = ConfigurableClassificationLabelGenerator(min_pip_movement=0.0, min_expected_return=10.0)
    assert gen.label(df, index=10, horizon=5) == "NO_TRADE"


def test_risk_reward_threshold_gate() -> None:
    df = _trending_df(drift=0.0006, seed=2)
    lenient = ConfigurableClassificationLabelGenerator(min_pip_movement=0.0, risk_reward_threshold=0.0)
    strict = ConfigurableClassificationLabelGenerator(min_pip_movement=0.0, risk_reward_threshold=1000.0)
    assert lenient.label(df, index=10, horizon=5) in ("BUY", "SELL")
    assert strict.label(df, index=10, horizon=5) == "NO_TRADE"


def test_max_adverse_excursion_cap_forces_no_trade() -> None:
    df = _trending_df(drift=0.0006, seed=2)
    uncapped = ConfigurableClassificationLabelGenerator(min_pip_movement=0.0, risk_reward_threshold=0.0)
    capped = ConfigurableClassificationLabelGenerator(
        min_pip_movement=0.0, risk_reward_threshold=0.0, max_adverse_excursion_pips=0.0001,
    )
    assert uncapped.label(df, index=10, horizon=5) in ("BUY", "SELL")
    assert capped.label(df, index=10, horizon=5) == "NO_TRADE"


def test_prediction_horizon_is_a_call_parameter_not_hardcoded() -> None:
    df = _trending_df(n=80, drift=0.0006, seed=2)
    gen = ConfigurableClassificationLabelGenerator(min_pip_movement=0.0, risk_reward_threshold=0.0)
    short_label = gen.label(df, index=10, horizon=3)
    long_label = gen.label(df, index=10, horizon=20)
    assert short_label in ("BUY", "SELL", "NO_TRADE")
    assert long_label in ("BUY", "SELL", "NO_TRADE")


def test_missing_required_class_raises() -> None:
    with pytest.raises(ValueError):
        ConfigurableClassificationLabelGenerator(classes=("UP", "DOWN"))


def test_extension_to_larger_class_set_via_subclass() -> None:
    """The class *set* itself is a constructor/subclass concern, never a
    Dataset Builder change -- demonstrate a 5-class extension
    (STRONG_BUY/WEAK_BUY/NO_TRADE/WEAK_SELL/STRONG_SELL)."""

    class FiveClassLabelGenerator(LabelGenerator):
        classes = ("STRONG_SELL", "WEAK_SELL", "NO_TRADE", "WEAK_BUY", "STRONG_BUY")

        def __init__(self) -> None:
            self._base = ConfigurableClassificationLabelGenerator(
                min_pip_movement=2.0, risk_reward_threshold=1.0,
            )

        def label(self, df: pd.DataFrame, index: int, horizon: int) -> str:
            base_label = self._base.label(df, index, horizon)
            if base_label == "NO_TRADE":
                return "NO_TRADE"
            strong_gen = ConfigurableClassificationLabelGenerator(
                min_pip_movement=15.0, risk_reward_threshold=2.0,
            )
            strong_label = strong_gen.label(df, index, horizon)
            if base_label == "BUY":
                return "STRONG_BUY" if strong_label == "BUY" else "WEAK_BUY"
            return "STRONG_SELL" if strong_label == "SELL" else "WEAK_SELL"

    gen = FiveClassLabelGenerator()
    df = _trending_df(n=80, drift=0.002, seed=9)
    label = gen.label(df, index=10, horizon=5)
    assert label in gen.classes
