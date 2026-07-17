"""Tests for ml_pipeline.dataset_builder -- the rolling-window orchestrator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from market_structure.candle import dataframe_to_candles
from ml_pipeline.config import DatasetConfig
from ml_pipeline.dataset_builder import DatasetBuilder
from ml_pipeline.label_generator import ThresholdLabelGenerator


@pytest.fixture()
def small_df() -> pd.DataFrame:
    return make_ohlcv(300, seed=11)


def test_build_produces_expected_shapes(small_df: pd.DataFrame) -> None:
    cfg = DatasetConfig(window_size=50, horizon=2, stride=3, symbol="TEST", timeframe="M5")
    builder = DatasetBuilder(cfg)
    ds = builder.build(small_df)

    expected_n = len(list(range(cfg.window_size - 1, len(small_df) - cfg.horizon, cfg.stride)))
    assert ds.X.shape == (expected_n, 185)
    assert len(ds.feature_names) == 185
    for name in cfg.regression_targets:
        assert ds.y_reg[name].shape == (expected_n,)
    assert ds.y_cls.shape == (expected_n,)
    assert len(ds.metadata) == expected_n
    assert list(ds.metadata.columns) == ["timestamp", "symbol", "timeframe", "window_start", "window_end"]
    assert (ds.metadata["symbol"] == "TEST").all()
    assert (ds.metadata["timeframe"] == "M5").all()


def test_window_never_reaches_decision_or_future_bar(small_df: pd.DataFrame) -> None:
    """Structural leakage check: each sample's window_end timestamp must equal
    the decision timestamp, and must be strictly before the next candle used
    for labeling."""
    cfg = DatasetConfig(window_size=40, horizon=1, stride=10)
    builder = DatasetBuilder(cfg)
    ds = builder.build(small_df)
    ts = small_df["timestamp"]
    for i, row in ds.metadata.iterrows():
        assert row["window_end"] == row["timestamp"]
        # window_start must be exactly window_size candles before window_end
        window_start_idx = ts[ts == row["window_start"]].index[0]
        window_end_idx = ts[ts == row["window_end"]].index[0]
        assert window_end_idx - window_start_idx + 1 == cfg.window_size


def test_metadata_timestamps_strictly_increasing(small_df: pd.DataFrame) -> None:
    cfg = DatasetConfig(window_size=30, horizon=1, stride=1)
    ds = DatasetBuilder(cfg).build(small_df)
    ts = ds.metadata["timestamp"]
    assert ts.is_monotonic_increasing
    assert not ts.duplicated().any()


def test_insufficient_data_raises(small_df: pd.DataFrame) -> None:
    cfg = DatasetConfig(window_size=1000, horizon=1)
    with pytest.raises(ValueError):
        DatasetBuilder(cfg).build(small_df)


def test_last_report_is_populated_and_valid(small_df: pd.DataFrame) -> None:
    cfg = DatasetConfig(window_size=40, horizon=2, stride=5)
    builder = DatasetBuilder(cfg)
    builder.build(small_df)
    report = builder.last_report
    assert report is not None
    assert report.is_valid
    assert report.leakage_ok
    assert report.feature_count_ok


def test_custom_label_generator_overrides_config(small_df: pd.DataFrame) -> None:
    cfg = DatasetConfig(window_size=40, horizon=1, stride=5, classification_label="threshold")
    strict_gen = ThresholdLabelGenerator(buy_threshold=0.05, sell_threshold=-0.05)  # near-impossible band
    ds = DatasetBuilder(cfg, label_generator=strict_gen).build(small_df)
    assert ds.class_names == ["SELL", "NO_TRADE", "BUY"]
    # With such an extreme threshold, everything should fall into NO_TRADE.
    assert (ds.y_cls == ds.class_names.index("NO_TRADE")).all()


def test_accepts_list_of_candle_input(small_df: pd.DataFrame) -> None:
    candles = dataframe_to_candles(small_df)
    cfg = DatasetConfig(window_size=30, horizon=1, stride=10)
    ds = DatasetBuilder(cfg).build(candles)
    assert ds.X.shape[1] == 185
    assert len(ds) > 0


def test_rejects_unsupported_input_type() -> None:
    cfg = DatasetConfig(window_size=30, horizon=1)
    with pytest.raises(TypeError):
        DatasetBuilder(cfg).build(12345)


def test_csv_input_round_trip(tmp_path, small_df: pd.DataFrame) -> None:
    path = tmp_path / "candles.csv"
    small_df.to_csv(path, index=False)
    cfg = DatasetConfig(window_size=30, horizon=1, stride=10)
    ds = DatasetBuilder(cfg).build(path)
    assert ds.X.shape[1] == 185


def test_regression_targets_are_configurable(small_df: pd.DataFrame) -> None:
    cfg = DatasetConfig(
        window_size=30, horizon=2, stride=10,
        regression_targets=["next_high", "future_volatility"],
    )
    ds = DatasetBuilder(cfg).build(small_df)
    assert set(ds.y_reg.keys()) == {"next_high", "future_volatility"}
