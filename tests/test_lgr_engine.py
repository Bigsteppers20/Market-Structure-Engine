"""End-to-end tests for logistic_regression.LogisticRegressionEngine,
including a real MarketState produced by the unmodified Market Structure
Engine (not a mock) -- the full Live Broker Data -> MarketState ->
LogisticRegressionEngine -> ClassificationPrediction flow."""
from __future__ import annotations

import pytest

from conftest import make_ohlcv
from logistic_regression import ClassificationConfig, LogisticRegressionEngine
from logistic_regression.exceptions import ModelNotTrainedError
from market_structure import EngineConfig, MarketStructureEngine
from training.config import TrainingConfig


def _engine_config(tmp_path, **overrides):
    tc = TrainingConfig(experiment_name="lgr_engine_test", output_root=str(tmp_path), random_seed=9)
    base = dict(prediction_horizon=5, n_bootstrap=6, symbol="EUR_USD", timeframe="M5", training_config=tc)
    base.update(overrides)
    return ClassificationConfig(**base)


def test_train_then_predict_live_market_state(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, ml_cfg = lgr_dataset_and_split
    engine = LogisticRegressionEngine(_engine_config(tmp_path))
    record = engine.train(dataset, split)
    assert record.model_family == "logistic_regression"

    engine.load_from_record(record)

    # A REAL MarketState from the unmodified Market Structure Engine, using
    # the same trailing window size the model trained on.
    df = make_ohlcv(700, seed=17)
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(df.iloc[-ml_cfg.window_size:])
    mse_engine.analyze()
    market_state = mse_engine.market_state()

    prediction = engine.predict(market_state, symbol="EUR_USD", timeframe="M5")

    assert prediction.predicted_class in ("SELL", "NO_TRADE", "BUY")
    total = prediction.buy_probability + prediction.sell_probability + prediction.no_trade_probability
    assert total == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= prediction.prediction_confidence <= 100.0
    assert prediction.symbol == "EUR_USD" and prediction.timeframe == "M5"
    assert prediction.prediction_horizon == 5
    assert any("Predicted class:" in line for line in prediction.explanation)


def test_predict_before_load_raises(tmp_path) -> None:
    engine = LogisticRegressionEngine(_engine_config(tmp_path))  # construction alone must not raise
    with pytest.raises(ModelNotTrainedError):
        engine.predict(object())  # no pipeline loaded yet -- raises before touching the arg


def test_registry_populated_after_training(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, _ = lgr_dataset_and_split
    engine = LogisticRegressionEngine(_engine_config(tmp_path))
    engine.train(dataset, split)
    models = engine.registry.list_models()
    assert "lgr_engine_test" in models
    meta = engine.registry.get("lgr_engine_test")
    assert meta.classification_labels == ("SELL", "NO_TRADE", "BUY")
    assert meta.prediction_horizon == 5
    assert meta.supported_symbols == ["EUR_USD"]
    assert meta.calibration_method == "none"


def test_model_persists_and_reloads_from_disk(tmp_path, lgr_dataset_and_split) -> None:
    """Model persistence: train once, then load a FRESH engine instance
    purely from the artifact directory on disk (simulating a new process)."""
    dataset, split, ml_cfg = lgr_dataset_and_split
    cfg = _engine_config(tmp_path)
    engine = LogisticRegressionEngine(cfg)
    record = engine.train(dataset, split)
    artifact_dir = record.artifact_dir

    fresh_engine = LogisticRegressionEngine(_engine_config(tmp_path))
    fresh_engine.load(artifact_dir)

    df = make_ohlcv(700, seed=17)
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(df.iloc[-ml_cfg.window_size:])
    mse_engine.analyze()
    market_state = mse_engine.market_state()

    prediction = fresh_engine.predict(market_state)
    assert prediction.predicted_class in ("SELL", "NO_TRADE", "BUY")


def test_calibrated_model_end_to_end(tmp_path, lgr_dataset_and_split) -> None:
    dataset, split, ml_cfg = lgr_dataset_and_split
    engine = LogisticRegressionEngine(_engine_config(tmp_path, calibration_method="isotonic"))
    engine.train(dataset, split)
    engine.load(engine.registry.get("lgr_engine_test").artifact_dir)

    df = make_ohlcv(700, seed=17)
    mse_engine = MarketStructureEngine(EngineConfig(swing_window=5))
    mse_engine.load(df.iloc[-ml_cfg.window_size:])
    mse_engine.analyze()
    prediction = engine.predict(mse_engine.market_state())
    assert any("isotonic" in line for line in prediction.explanation)


def test_engine_never_imports_strategy_or_linear_regression() -> None:
    """The Logistic Regression Engine must never depend on the Strategy
    Engine's outputs or on the Linear Regression Engine -- three
    independent analytical systems (per the spec's explicit requirement)."""
    import logistic_regression
    import pkgutil

    for module_info in pkgutil.iter_modules(logistic_regression.__path__):
        module = __import__(f"logistic_regression.{module_info.name}", fromlist=["_"])
        source_file = getattr(module, "__file__", None)
        if not source_file:
            continue
        with open(source_file, encoding="utf-8") as fh:
            source = fh.read()
        assert "import strategy" not in source
        assert "from strategy" not in source
        assert "import linear_regression" not in source
        assert "from linear_regression" not in source
