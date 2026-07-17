"""Tests for ml_pipeline.label_generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.label_generator import (
    REGRESSION_REGISTRY,
    LabelGenerator,
    ThresholdLabelGenerator,
    build_classification_label_generator,
    compute_regression_targets,
)


def _df() -> pd.DataFrame:
    close = [1.1000, 1.1010, 1.1005, 1.1030, 1.0990, 1.1050]
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=len(close), freq="5min"),
        "open": close, "high": [c + 0.0005 for c in close], "low": [c - 0.0005 for c in close],
        "close": close, "volume": [100.0] * len(close),
    })


def test_next_close_open_high_low() -> None:
    df = _df()
    assert REGRESSION_REGISTRY["next_close"](df, 0, 1, 0.0001) == pytest.approx(1.1010)
    assert REGRESSION_REGISTRY["next_open"](df, 0, 1, 0.0001) == pytest.approx(1.1010)
    assert REGRESSION_REGISTRY["next_high"](df, 0, 1, 0.0001) == pytest.approx(1.1015)
    assert REGRESSION_REGISTRY["next_low"](df, 0, 1, 0.0001) == pytest.approx(1.1005)


def test_next_return_and_log_return() -> None:
    df = _df()
    ret = REGRESSION_REGISTRY["next_return"](df, 0, 1, 0.0001)
    assert ret == pytest.approx((1.1010 - 1.1000) / 1.1000)
    log_ret = REGRESSION_REGISTRY["next_log_return"](df, 0, 1, 0.0001)
    assert log_ret == pytest.approx(np.log(1.1010 / 1.1000))


def test_expected_pip_movement_and_percentage_change() -> None:
    df = _df()
    pips = REGRESSION_REGISTRY["expected_pip_movement"](df, 0, 1, 0.0001)
    assert pips == pytest.approx((1.1010 - 1.1000) / 0.0001)
    pct = REGRESSION_REGISTRY["expected_percentage_change"](df, 0, 1, 0.0001)
    assert pct == pytest.approx((1.1010 - 1.1000) / 1.1000 * 100.0)


def test_future_atr_and_volatility_are_forward_looking_only() -> None:
    df = _df()
    atr = REGRESSION_REGISTRY["future_atr"](df, 1, 2, 0.0001)
    assert atr >= 0.0
    vol = REGRESSION_REGISTRY["future_volatility"](df, 1, 2, 0.0001)
    assert vol >= 0.0
    # horizon=0-length tail (index near end) should degrade gracefully, not crash.
    assert REGRESSION_REGISTRY["future_atr"](df, len(df) - 1, 1, 0.0001) == 0.0


def test_compute_regression_targets_subset() -> None:
    df = _df()
    out = compute_regression_targets(df, 0, 1, ["next_close", "next_return"])
    assert set(out) == {"next_close", "next_return"}


def test_threshold_label_generator_boundaries() -> None:
    df = _df()
    gen = ThresholdLabelGenerator(buy_threshold=0.0005, sell_threshold=-0.0005)
    # bar0->bar1: (1.1010-1.1000)/1.1000 ~ +0.00091 > 0.0005 => BUY
    assert gen.label(df, 0, 1) == "BUY"
    # bar1->bar2: (1.1005-1.1010)/1.1010 ~ -0.00045, within band => NO_TRADE
    assert gen.label(df, 1, 1) == "NO_TRADE"
    # bar3->bar4: (1.0990-1.1030)/1.1030 ~ -0.00363 < -0.0005 => SELL
    assert gen.label(df, 3, 1) == "SELL"
    assert gen.classes == ("SELL", "NO_TRADE", "BUY")


def test_threshold_label_generator_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        ThresholdLabelGenerator(buy_threshold=-0.001)
    with pytest.raises(ValueError):
        ThresholdLabelGenerator(sell_threshold=0.001)


def test_build_classification_label_generator_registry() -> None:
    gen = build_classification_label_generator("threshold", buy_threshold=0.001, sell_threshold=-0.001)
    assert isinstance(gen, ThresholdLabelGenerator)
    with pytest.raises(ValueError):
        build_classification_label_generator("does_not_exist")


def test_custom_label_generator_plugs_in() -> None:
    class UpDownLabelGenerator(LabelGenerator):
        classes = ("DOWN", "UP")

        def label(self, df: pd.DataFrame, index: int, horizon: int) -> str:
            return "UP" if df["close"].iloc[index + horizon] >= df["close"].iloc[index] else "DOWN"

    gen = UpDownLabelGenerator()
    df = _df()
    assert gen.label(df, 0, 1) == "UP"
    assert gen.classes == ("DOWN", "UP")
