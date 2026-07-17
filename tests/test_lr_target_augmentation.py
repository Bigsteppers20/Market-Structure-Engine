"""Tests for linear_regression.target_generator.augment_dataset_targets and
RegressionEngine.train()'s raw_df augmentation path -- the fix for
ml_pipeline.DatasetBuilder not knowing about this engine's 5 new targets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from linear_regression.config import RegressionConfig
from linear_regression.exceptions import UnsupportedTargetError
from linear_regression.regression_engine import RegressionEngine
from linear_regression.target_generator import augment_dataset_targets, compute_target
from ml_pipeline import DatasetBuilder
from ml_pipeline.config import DatasetConfig as MLDatasetConfig
from ml_pipeline.splitter import TimeSeriesSplitter
from training.config import TrainingConfig


@pytest.fixture()
def native_dataset_and_raw_df():
    df = make_ohlcv(400, seed=23)
    cfg = MLDatasetConfig(window_size=80, horizon=4, stride=5, symbol="TEST", timeframe="M5",
                           regression_targets=["next_close"])
    dataset = DatasetBuilder(cfg).build(df)
    return dataset, df, cfg


def test_augment_matches_direct_computation(native_dataset_and_raw_df) -> None:
    dataset, df, cfg = native_dataset_and_raw_df
    extra = augment_dataset_targets(dataset, df, ["future_range"], cfg.horizon)
    assert "future_range" in extra
    assert len(extra["future_range"]) == len(dataset)

    # Spot-check the first sample against a direct computation.
    ts = dataset.metadata["timestamp"].iloc[0]
    idx = df.index[df["timestamp"] == ts][0]
    expected = compute_target(df, idx, cfg.horizon, "future_range")
    assert extra["future_range"][0] == pytest.approx(expected)


def test_augment_rejects_mismatched_raw_df(native_dataset_and_raw_df) -> None:
    dataset, df, cfg = native_dataset_and_raw_df
    # make_ohlcv's timestamps don't depend on seed (only prices do), so use a
    # different start date to get genuinely non-overlapping timestamps.
    wrong_df = make_ohlcv(400, seed=23, start_price=1.3000)
    wrong_df["timestamp"] = wrong_df["timestamp"] + pd.Timedelta(days=365)
    with pytest.raises(ValueError):
        augment_dataset_targets(dataset, wrong_df, ["future_range"], cfg.horizon)


def test_regression_engine_train_requires_raw_df_for_new_targets(tmp_path, native_dataset_and_raw_df) -> None:
    dataset, df, cfg = native_dataset_and_raw_df
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    tc = TrainingConfig(experiment_name="augment_test", output_root=str(tmp_path))
    reg_cfg = RegressionConfig(targets=["maximum_favorable_excursion"], prediction_horizon=cfg.horizon,
                                n_bootstrap=0, training_config=tc)
    engine = RegressionEngine(reg_cfg)
    with pytest.raises(UnsupportedTargetError):
        engine.train(dataset, split)  # no raw_df -- must raise, not silently fail


def test_regression_engine_train_succeeds_with_raw_df(tmp_path, native_dataset_and_raw_df) -> None:
    dataset, df, cfg = native_dataset_and_raw_df
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    tc = TrainingConfig(experiment_name="augment_test2", output_root=str(tmp_path))
    reg_cfg = RegressionConfig(targets=["maximum_favorable_excursion"], prediction_horizon=cfg.horizon,
                                n_bootstrap=0, training_config=tc)
    engine = RegressionEngine(reg_cfg)
    records = engine.train(dataset, split, raw_df=df)
    assert "maximum_favorable_excursion" in records
    assert np.isfinite(records["maximum_favorable_excursion"].training_metrics["mae"])


def test_unknown_target_raises_before_needing_raw_df(tmp_path, native_dataset_and_raw_df) -> None:
    dataset, df, cfg = native_dataset_and_raw_df
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    tc = TrainingConfig(experiment_name="augment_test3", output_root=str(tmp_path))
    reg_cfg = RegressionConfig(targets=["totally_bogus_target"], prediction_horizon=cfg.horizon,
                                n_bootstrap=0, training_config=tc)
    engine = RegressionEngine(reg_cfg)
    with pytest.raises(UnsupportedTargetError):
        engine.train(dataset, split)
