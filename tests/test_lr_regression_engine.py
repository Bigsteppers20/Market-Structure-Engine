"""End-to-end tests for linear_regression.RegressionEngine, including a real
MarketState produced by the unmodified Market Structure Engine (not a mock)."""
from __future__ import annotations

import pytest

from conftest import make_ohlcv
from linear_regression import RegressionConfig, RegressionEngine
from linear_regression.exceptions import ModelNotTrainedError
from market_structure import EngineConfig, MarketStructureEngine
from training.config import TrainingConfig


def _engine_config(tmp_path, **overrides):
    tc = TrainingConfig(experiment_name="lr_engine_test", output_root=str(tmp_path), random_seed=9)
    base = dict(
        targets=["next_close", "next_high", "expected_pip_movement"],
        prediction_horizon=5, model_type="ridge", model_hyperparameters={"alpha": 1.0},
        n_bootstrap=6, symbol="EUR_USD", timeframe="M5", training_config=tc,
    )
    base.update(overrides)
    return RegressionConfig(**base)


def test_train_then_predict_live_market_state(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, ml_cfg = lr_dataset_and_split
    engine = RegressionEngine(_engine_config(tmp_path))
    records = engine.train(dataset, split)
    assert set(records) == {"next_close", "next_high", "expected_pip_movement"}

    engine.load_from_records(records)

    # A REAL MarketState from the unmodified Market Structure Engine, using
    # the same trailing window size the model trained on (see the smoke-test
    # note in LINEAR_REGRESSION_ENGINE_REPORT.md about why this matters for
    # the confidence engine's distribution-distance factor).
    df = make_ohlcv(700, seed=17)
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(df.iloc[-ml_cfg.window_size:])
    mse_engine.analyze()
    market_state = mse_engine.market_state()

    prediction = engine.predict(market_state, symbol="EUR_USD", timeframe="M5")
    assert prediction.expected_close is not None
    assert prediction.expected_high is not None
    assert prediction.expected_pip_move is not None
    assert prediction.expected_low is None
    assert 0.0 <= prediction.prediction_confidence <= 100.0
    assert prediction.symbol == "EUR_USD" and prediction.timeframe == "M5"
    assert any("Confidence:" in line for line in prediction.explanation)


def test_predict_before_load_raises(tmp_path) -> None:
    engine = RegressionEngine(_engine_config(tmp_path))  # construction alone must not raise
    with pytest.raises(ModelNotTrainedError):
        engine.predict(object())  # no pipelines loaded yet -- raises before touching the arg


def test_registry_populated_after_training(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, _ = lr_dataset_and_split
    engine = RegressionEngine(_engine_config(tmp_path, targets=["next_close"]))
    engine.train(dataset, split)
    models = engine.registry.list_models()
    assert "lr_engine_test__next_close" in models
    meta = engine.registry.get("lr_engine_test__next_close")
    assert meta.regression_target == "next_close"
    assert meta.supported_symbols == ["EUR_USD"]


def test_single_target_config_leaves_other_fields_none(tmp_path, lr_dataset_and_split) -> None:
    dataset, split, ml_cfg = lr_dataset_and_split
    engine = RegressionEngine(_engine_config(tmp_path, targets=["next_close"]))
    records = engine.train(dataset, split)
    engine.load_from_records(records)

    df = make_ohlcv(700, seed=17)
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(df.iloc[-ml_cfg.window_size:])
    mse_engine.analyze()
    prediction = engine.predict(mse_engine.market_state())
    assert prediction.expected_close is not None
    assert prediction.expected_high is None
    assert prediction.expected_pip_move is None
