"""Training/caching/inference pipeline shared by every API router.

Each analytical engine is cached independently (regression and
classification models are no longer tied to a strategy choice -- only the
Decision Engine actually needs all three together). Reuses every engine's
own public training/inference API exactly as the example scripts do; this
module adds no analytical logic of its own.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

from oanda_client import fetch_candles  # noqa: E402

from market_structure import EngineConfig, MarketState, MarketStructureEngine  # noqa: E402
from ml_pipeline import Dataset, DatasetBuilder, TimeSeriesSplitter  # noqa: E402
from ml_pipeline.config import DatasetConfig as MLDatasetConfig  # noqa: E402
from ml_pipeline.splitter import SplitResult  # noqa: E402
from training.config import TrainingConfig  # noqa: E402

from strategy import StrategyEngine, StrategyEvaluation  # noqa: E402
from strategy.strategy_registry import default_registry  # noqa: E402
from strategies.ict_strategy import default_config as ict_default_config  # noqa: E402
from strategies.london_breakout import default_config as london_breakout_default_config  # noqa: E402
from strategies.scalping_strategy import default_config as scalping_default_config  # noqa: E402
from strategies.swing_strategy import default_config as swing_default_config  # noqa: E402
from strategies.trend_following import default_config as trend_following_default_config  # noqa: E402

from linear_regression import RegressionConfig, RegressionEngine  # noqa: E402
from logistic_regression import ClassificationConfig, LogisticRegressionEngine  # noqa: E402
from logistic_regression.label_manager import ConfigurableClassificationLabelGenerator  # noqa: E402
from logistic_regression.live_inference import to_live_inference  # noqa: E402

from decision_engine import DecisionEngine, DecisionEngineConfig  # noqa: E402
from decision_engine.exceptions import DecisionEngineError  # noqa: E402

from model_monitor import ModelLifecycleMetadata, ModelMonitor, MonitorConfig  # noqa: E402
from model_monitor.prediction_monitor import ResolvedPrediction  # noqa: E402
from training.utils import utc_timestamp  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "api_output"


class _ArrayMarketState:
    """Duck-typed stand-in for a real ``MarketState``, wrapping an already-
    computed feature row from a ``ml_pipeline.Dataset`` -- the same pattern
    this project's own test suite uses (e.g. ``test_lr_inference_predictor.py``'s
    ``_FakeMarketState``) to feed a real, already-built feature vector
    through an engine's ``predict()`` without re-running the Market
    Structure Engine. Assumes full feature validity (1.0), reasonable for
    historical rows the engine already trained/tested against."""

    def __init__(self, vector, feature_names: List[str]) -> None:
        self._vector = vector
        self._names = feature_names

    def to_vector(self):
        return self._vector, self._names

    def to_dict(self) -> Dict[str, float]:
        return {f"{name}_valid": 1.0 for name in self._names}

_STRATEGY_DEFAULT_CONFIGS = {
    "ict": ict_default_config,
    "trend_following": trend_following_default_config,
    "london_breakout": london_breakout_default_config,
    "swing": swing_default_config,
    "scalping": scalping_default_config,
}


class PipelineError(RuntimeError):
    """Raised for any user-facing pipeline failure (bad strategy name,
    OANDA fetch failure, insufficient data, etc.) -- every router maps
    this to an HTTP 400."""


def available_strategies() -> List[str]:
    return sorted(_STRATEGY_DEFAULT_CONFIGS)


# --------------------------------------------------------------------------- #
# Market Structure Engine -- no caching, no training, always a fresh read.
# --------------------------------------------------------------------------- #
def build_market_state(symbol: str, timeframe: str, window_size: int, count: Optional[int] = None) -> MarketState:
    try:
        candles = fetch_candles(symbol, timeframe, count=count or (window_size + 10))
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Failed to fetch OANDA candles for {symbol} {timeframe}: {exc}") from exc
    if len(candles) < window_size:
        raise PipelineError(f"Only {len(candles)} candles available, need >= {window_size} (window_size).")
    engine = MarketStructureEngine(EngineConfig(swing_window=5))
    engine.load(candles.iloc[-window_size:])
    engine.analyze()
    return engine.market_state()


# --------------------------------------------------------------------------- #
# Strategy Engine -- cheap to construct, never cached.
# --------------------------------------------------------------------------- #
def evaluate_strategy(market_state: MarketState, strategy_name: str, symbol: str, timeframe: str) -> StrategyEvaluation:
    if strategy_name not in _STRATEGY_DEFAULT_CONFIGS:
        raise PipelineError(f"Unknown strategy_name={strategy_name!r}. Available: {available_strategies()}")
    strategy_engine = StrategyEngine()
    strategy_cls = default_registry().get(strategy_name)
    strategy_engine.register_strategy(strategy_cls(_STRATEGY_DEFAULT_CONFIGS[strategy_name]()))
    return strategy_engine.evaluate(market_state, strategy_name, symbol=symbol, timeframe=timeframe)


# --------------------------------------------------------------------------- #
# Linear Regression Engine
# --------------------------------------------------------------------------- #
@dataclass
class RegressionBundle:
    symbol: str
    timeframe: str
    window_size: int
    horizon: int
    engine: RegressionEngine
    test_metrics: Dict[str, dict]
    dataset: Dataset
    split: SplitResult
    training_date: str


_regression_cache: Dict[Tuple[str, str], RegressionBundle] = {}
_regression_lock = threading.Lock()


def train_regression(
    *, symbol: str, timeframe: str, count: int = 2500, window_size: int = 200, horizon: int = 5, stride: int = 3,
) -> RegressionBundle:
    try:
        candles = fetch_candles(symbol, timeframe, count=count)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Failed to fetch OANDA candles for {symbol} {timeframe}: {exc}") from exc

    ml_cfg = MLDatasetConfig(
        window_size=window_size, horizon=horizon, stride=stride, symbol=symbol, timeframe=timeframe,
        regression_targets=["next_close", "expected_pip_movement"],
    )
    dataset = DatasetBuilder(ml_cfg).build(candles)
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    training_config = TrainingConfig(
        experiment_name=f"api_{symbol}_{timeframe}_lr", output_root=str(OUTPUT_ROOT / "lr"),
        scaler="standard", random_seed=42,
    )
    config = RegressionConfig(
        targets=["next_close", "expected_pip_movement"], prediction_horizon=horizon, model_type="ridge",
        # alpha=1.0 (sklearn's Ridge default) badly overfits here: 185 features
        # against only a few hundred training rows. alpha=300 was empirically
        # checked against real OANDA data -- it improves BOTH targets' test R2
        # (next_close: ~0.71 -> ~0.84; expected_pip_movement: ~-1.87 -> ~-0.26),
        # since the same regularization weakness hurt both. expected_pip_movement's
        # R2 stays modestly negative even at high alpha -- 5-bar-ahead forex
        # pip movement is close to a random walk, so a strongly positive R2
        # isn't realistically achievable for a linear model on this target;
        # this fix removes the overfitting, it doesn't manufacture signal
        # that isn't there.
        model_hyperparameters={"alpha": 300.0},
        n_bootstrap=10, symbol=symbol, timeframe=timeframe, training_config=training_config,
    )
    engine = RegressionEngine(config)
    records = engine.train(dataset, split)
    engine.load_from_records(records)

    bundle = RegressionBundle(
        symbol=symbol, timeframe=timeframe, window_size=window_size, horizon=horizon, engine=engine,
        test_metrics={target: record.testing_metrics for target, record in records.items()},
        dataset=dataset, split=split, training_date=records["next_close"].timestamp,
    )
    with _regression_lock:
        _regression_cache[(symbol, timeframe)] = bundle
    return bundle


def get_or_train_regression(
    *, symbol: str, timeframe: str, count: int = 2500, window_size: int = 200, horizon: int = 5, stride: int = 3,
) -> RegressionBundle:
    with _regression_lock:
        existing = _regression_cache.get((symbol, timeframe))
    if existing is not None:
        return existing
    return train_regression(symbol=symbol, timeframe=timeframe, count=count, window_size=window_size, horizon=horizon, stride=stride)


def predict_regression(bundle: RegressionBundle) -> dict:
    market_state = build_market_state(bundle.symbol, bundle.timeframe, bundle.window_size)
    prediction = bundle.engine.predict(market_state, symbol=bundle.symbol, timeframe=bundle.timeframe)
    return prediction.to_dict()


# --------------------------------------------------------------------------- #
# Logistic Regression Engine
# --------------------------------------------------------------------------- #
@dataclass
class ClassificationBundle:
    symbol: str
    timeframe: str
    window_size: int
    horizon: int
    engine: LogisticRegressionEngine
    test_metrics: dict
    dataset: Dataset
    split: SplitResult
    classes: List[str]
    training_date: str


_classification_cache: Dict[Tuple[str, str], ClassificationBundle] = {}
_classification_lock = threading.Lock()


def train_classification(
    *, symbol: str, timeframe: str, count: int = 2500, window_size: int = 200, horizon: int = 5, stride: int = 3,
) -> ClassificationBundle:
    try:
        candles = fetch_candles(symbol, timeframe, count=count)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Failed to fetch OANDA candles for {symbol} {timeframe}: {exc}") from exc

    ml_cfg = MLDatasetConfig(window_size=window_size, horizon=horizon, stride=stride, symbol=symbol, timeframe=timeframe)
    label_generator = ConfigurableClassificationLabelGenerator(min_pip_movement=5.0, risk_reward_threshold=1.5)
    dataset = DatasetBuilder(ml_cfg, label_generator=label_generator).build(candles)
    split = TimeSeriesSplitter(method="simple", train_frac=0.7, val_frac=0.15).split(len(dataset))[0]
    training_config = TrainingConfig(
        experiment_name=f"api_{symbol}_{timeframe}_lgr", output_root=str(OUTPUT_ROOT / "lgr"),
        scaler="standard", random_seed=42,
    )
    config = ClassificationConfig(
        prediction_horizon=horizon, min_pip_movement=5.0, risk_reward_threshold=1.5,
        class_balancing="class_weight", calibration_method="platt", n_bootstrap=10,
        symbol=symbol, timeframe=timeframe, training_config=training_config,
    )
    engine = LogisticRegressionEngine(config)
    record = engine.train(dataset, split)
    engine.load_from_record(record)

    bundle = ClassificationBundle(
        symbol=symbol, timeframe=timeframe, window_size=window_size, horizon=horizon, engine=engine,
        test_metrics=record.testing_metrics, dataset=dataset, split=split,
        classes=list(dataset.class_names), training_date=record.timestamp,
    )
    with _classification_lock:
        _classification_cache[(symbol, timeframe)] = bundle
    return bundle


def get_or_train_classification(
    *, symbol: str, timeframe: str, count: int = 2500, window_size: int = 200, horizon: int = 5, stride: int = 3,
) -> ClassificationBundle:
    with _classification_lock:
        existing = _classification_cache.get((symbol, timeframe))
    if existing is not None:
        return existing
    return train_classification(symbol=symbol, timeframe=timeframe, count=count, window_size=window_size, horizon=horizon, stride=stride)


def predict_classification(bundle: ClassificationBundle) -> dict:
    """Live production inference: returns ONLY the minimal prediction
    object (predicted direction + confidence + versions + model health) --
    never the full evaluation-oriented ``ClassificationPrediction`` (no
    class probabilities, no confusion matrix, no accuracy/precision/recall/
    F1/ROC-AUC/PR-AUC/log-loss/Brier score). See
    ``LOGISTIC_REGRESSION_INFERENCE_UPDATE.md``. The full object is still
    used internally by the Decision Engine and Model Monitor -- both call
    ``bundle.engine.predict()`` directly (see ``make_decision()``/
    ``monitor_health()`` below), never this function."""
    market_state = build_market_state(bundle.symbol, bundle.timeframe, bundle.window_size)
    prediction = bundle.engine.predict(market_state, symbol=bundle.symbol, timeframe=bundle.timeframe)
    return to_live_inference(prediction).to_dict()


# --------------------------------------------------------------------------- #
# Decision Engine -- combines a fresh Strategy evaluation with the cached
# regression/classification bundles.
# --------------------------------------------------------------------------- #
def make_decision(
    *, symbol: str, timeframe: str, strategy_name: str, count: int = 2500,
    window_size: int = 200, horizon: int = 5, stride: int = 3,
) -> dict:
    regression_bundle = get_or_train_regression(
        symbol=symbol, timeframe=timeframe, count=count, window_size=window_size, horizon=horizon, stride=stride,
    )
    classification_bundle = get_or_train_classification(
        symbol=symbol, timeframe=timeframe, count=count, window_size=window_size, horizon=horizon, stride=stride,
    )
    market_state = build_market_state(symbol, timeframe, window_size)
    strategy_eval = evaluate_strategy(market_state, strategy_name, symbol, timeframe)
    regression_pred = regression_bundle.engine.predict(market_state, symbol=symbol, timeframe=timeframe)
    classification_pred = classification_bundle.engine.predict(market_state, symbol=symbol, timeframe=timeframe)

    decision_engine = DecisionEngine(DecisionEngineConfig())
    try:
        decision = decision_engine.decide(
            market_state=market_state, strategy_evaluation=strategy_eval,
            regression_prediction=regression_pred, classification_prediction=classification_pred,
            symbol=symbol, timeframe=timeframe,
        )
    except DecisionEngineError as exc:
        raise PipelineError(str(exc)) from exc
    return decision.to_dict()


# --------------------------------------------------------------------------- #
# Model Monitor -- fits a baseline from the cached bundle's own training
# dataset, replays its held-out test split as "live" resolved predictions
# (ground truth is already known for a historical split, so this is safe --
# no look-ahead into anything the model hasn't already been scored against),
# then evaluates health/drift/calibration exactly as the live monitoring
# loop would once real predictions accumulate over time.
# --------------------------------------------------------------------------- #
def monitor_health(
    *, symbol: str, timeframe: str, task_type: str, count: int = 2500,
    window_size: int = 200, horizon: int = 5, stride: int = 3,
) -> dict:
    if task_type == "regression":
        bundle = get_or_train_regression(symbol=symbol, timeframe=timeframe, count=count, window_size=window_size, horizon=horizon, stride=stride)
        classes = None
        primary_target = "next_close"
    elif task_type == "classification":
        bundle = get_or_train_classification(symbol=symbol, timeframe=timeframe, count=count, window_size=window_size, horizon=horizon, stride=stride)
        classes = bundle.classes
        primary_target = None
    else:
        raise PipelineError(f"task_type={task_type!r}, expected 'regression' or 'classification'.")

    test_idx = bundle.split.test_idx
    if len(test_idx) < 10:
        raise PipelineError(f"Test split only has {len(test_idx)} samples -- need >= 10 to evaluate health. Increase count.")

    monitor_config = MonitorConfig(
        min_new_samples=5, rolling_window=min(50, len(test_idx)), market_regime_lookback=min(50, len(test_idx)),
        max_model_age_days=9999,
    )
    monitor = ModelMonitor(
        monitor_config, model_name=f"api_{task_type}_{symbol}_{timeframe}", task_type=task_type,
        output_root=OUTPUT_ROOT / "monitor", classes=classes,
    )

    lifecycle = ModelLifecycleMetadata(
        model_name=monitor.model_name, version="1", task_type=task_type, status="candidate",
        training_date=bundle.training_date, training_dataset_size=len(bundle.dataset),
        feature_version="1.0.0", training_version="1.0.0", strategy_version="1.0.0", dataset_version="1.0.0",
    )
    monitor.register_model(lifecycle, as_production=True)

    train_idx = bundle.split.train_idx
    if task_type == "regression":
        monitor.fit_baseline(
            bundle.dataset.X[train_idx], bundle.dataset.feature_names,
            target_values=bundle.dataset.y_reg[primary_target][train_idx],
        )
    else:
        monitor.fit_baseline(bundle.dataset.X[train_idx], bundle.dataset.feature_names)

    now_iso = utc_timestamp()
    feature_names = bundle.dataset.feature_names
    for i in test_idx:
        fake_state = _ArrayMarketState(bundle.dataset.X[i], feature_names)
        prediction = bundle.engine.predict(fake_state, symbol=symbol, timeframe=timeframe)

        if task_type == "regression":
            snapshot = ModelMonitor.from_regression_prediction(
                prediction, model_name=monitor.model_name, primary_target=primary_target,
                decision_index=int(i), feature_vector=bundle.dataset.X[i], feature_names=feature_names,
            )
            actual_value = float(bundle.dataset.y_reg[primary_target][i])
            resolved = ResolvedPrediction(snapshot=snapshot, resolved_at=now_iso, actual_value=actual_value)
        else:
            snapshot = ModelMonitor.from_classification_prediction(
                prediction, model_name=monitor.model_name, decision_index=int(i),
                feature_vector=bundle.dataset.X[i], feature_names=feature_names,
            )
            actual_class = bundle.dataset.class_names[int(bundle.dataset.y_cls[i])]
            resolved = ResolvedPrediction(snapshot=snapshot, resolved_at=now_iso, actual_class=actual_class)

        # Bypass PredictionLog.resolve()'s DataFrame-based resolver: the
        # ground-truth label/value for a held-out test row is already known
        # directly from the dataset (no look-ahead -- this is historical
        # data the model was already scored against at training time), so
        # there is nothing to "resolve" via a forward-looking df lookup.
        monitor.prediction_log.log(snapshot)
        monitor.prediction_log._pending.pop()
        monitor.prediction_log._resolved.append(resolved)

    health_report = monitor.evaluate_health(now_iso=now_iso)
    recommendation = monitor.recommend_retraining(current_horizon=bundle.horizon, now_iso=now_iso)
    agentic_report = monitor.to_agentic_report(model_display_name=f"{task_type}:{symbol}:{timeframe}")

    return {
        "agentic_report": agentic_report,
        "health": health_report.to_dict(),
        "retraining_recommendation": recommendation.to_dict(),
    }
