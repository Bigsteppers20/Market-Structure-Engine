"""Shared fixtures for the Market Structure Engine test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n: int = 500,
    seed: int = 7,
    start_price: float = 1.1000,
    drift: float = 0.0,
    vol: float = 0.0008,
    freq: str = "5min",
) -> pd.DataFrame:
    """Generate a random-walk OHLCV DataFrame with valid candle geometry."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, vol * start_price, (2, n)))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    volume = rng.integers(50, 500, n).astype(float)
    ts = pd.date_range("2025-01-06", periods=n, freq=freq)
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low,
         "close": close, "volume": volume}
    )


@pytest.fixture()
def random_df() -> pd.DataFrame:
    """500 candles of seeded random-walk data."""
    return make_ohlcv()


@pytest.fixture()
def trending_up_df() -> pd.DataFrame:
    """Uptrending series with enough noise to form HH/HL swing structure."""
    return make_ohlcv(n=400, seed=3, drift=0.0004, vol=0.0012)


@pytest.fixture()
def trending_down_df() -> pd.DataFrame:
    """Downtrending series with pullbacks (LH/LL structure)."""
    return make_ohlcv(n=400, seed=4, drift=-0.0004, vol=0.0012)


def _build_market_state(df: pd.DataFrame, swing_window: int = 5):
    from market_structure import EngineConfig, MarketStructureEngine

    engine = MarketStructureEngine(EngineConfig(swing_window=swing_window))
    engine.load(df)
    engine.analyze()
    return engine.market_state()


@pytest.fixture()
def market_state(random_df: pd.DataFrame):
    """A real MarketState (not a mock) from the unmodified Market Structure
    Engine, for strategy/ tests that must consume the real public API."""
    return _build_market_state(random_df)


@pytest.fixture()
def bullish_market_state(trending_up_df: pd.DataFrame):
    return _build_market_state(trending_up_df, swing_window=3)


@pytest.fixture()
def bearish_market_state(trending_down_df: pd.DataFrame):
    return _build_market_state(trending_down_df, swing_window=3)


@pytest.fixture(scope="session")
def lr_dataset_and_split():
    """A real ml_pipeline.Dataset + chronological split, built once per test
    session (dataset building is not free) -- shared by every
    linear_regression/ integration test that needs real X/y data."""
    from ml_pipeline import DatasetBuilder
    from ml_pipeline.config import DatasetConfig as MLDatasetConfig
    from ml_pipeline.splitter import TimeSeriesSplitter

    df = make_ohlcv(700, seed=17)
    cfg = MLDatasetConfig(
        window_size=120, horizon=5, stride=3, symbol="TEST", timeframe="M5",
        regression_targets=["next_close", "next_high", "next_return", "expected_pip_movement"],
    )
    dataset = DatasetBuilder(cfg).build(df)
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    return dataset, split, cfg


@pytest.fixture(scope="session")
def lgr_dataset_and_split():
    """A real ml_pipeline.Dataset (with classification labels from
    logistic_regression's own ConfigurableClassificationLabelGenerator) +
    chronological split, built once per test session -- shared by every
    logistic_regression/ integration test that needs real X/y_cls data.

    Same underlying series/window/horizon/stride as ``lr_dataset_and_split``
    (confirmed to yield all 3 default classes -- SELL/NO_TRADE/BUY -- in
    every one of train/val/test) so the two engines' tests are directly
    comparable.
    """
    from ml_pipeline import DatasetBuilder
    from ml_pipeline.config import DatasetConfig as MLDatasetConfig
    from ml_pipeline.splitter import TimeSeriesSplitter

    from logistic_regression.label_manager import ConfigurableClassificationLabelGenerator

    df = make_ohlcv(700, seed=17)
    cfg = MLDatasetConfig(window_size=120, horizon=5, stride=3, symbol="TEST", timeframe="M5")
    label_generator = ConfigurableClassificationLabelGenerator(
        min_pip_movement=5.0, risk_reward_threshold=1.5,
    )
    dataset = DatasetBuilder(cfg, label_generator=label_generator).build(df)
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    return dataset, split, cfg


@pytest.fixture(scope="session")
def mon_regression_engine(lr_dataset_and_split, tmp_path_factory):
    """A trained, loaded ``linear_regression.RegressionEngine`` (single
    target, ``next_close``) -- shared by every ``model_monitor/`` test that
    needs a real regression predictor rather than a hand-built prediction."""
    from linear_regression import RegressionConfig, RegressionEngine
    from training.config import TrainingConfig

    dataset, split, ml_cfg = lr_dataset_and_split
    output_root = tmp_path_factory.mktemp("mon_lr_output")
    tc = TrainingConfig(experiment_name="mon_lr_test", output_root=str(output_root), random_seed=1)
    cfg = RegressionConfig(
        targets=["next_close"], prediction_horizon=5, model_type="linear", n_bootstrap=5, training_config=tc,
    )
    engine = RegressionEngine(cfg)
    records = engine.train(dataset, split)
    engine.load_from_records(records)
    return engine, dataset, split, ml_cfg, records


@pytest.fixture(scope="session")
def mon_classification_engine(lgr_dataset_and_split, tmp_path_factory):
    """A trained, loaded ``logistic_regression.LogisticRegressionEngine`` --
    shared by every ``model_monitor/`` test that needs a real classification
    predictor rather than a hand-built prediction."""
    from logistic_regression import ClassificationConfig, LogisticRegressionEngine
    from training.config import TrainingConfig

    dataset, split, ml_cfg = lgr_dataset_and_split
    output_root = tmp_path_factory.mktemp("mon_lgr_output")
    tc = TrainingConfig(experiment_name="mon_lgr_test", output_root=str(output_root), random_seed=1)
    cfg = ClassificationConfig(prediction_horizon=5, n_bootstrap=5, training_config=tc)
    engine = LogisticRegressionEngine(cfg)
    record = engine.train(dataset, split)
    engine.load_from_record(record)
    return engine, dataset, split, ml_cfg, record
