"""``DecisionEngine`` -- the production entry point.

Combines one ``StrategyEvaluation`` (required) with an optional
``RegressionPrediction`` and an optional ``ClassificationPrediction`` into
one :class:`~decision_engine.decision_result.DecisionResult`. Every
upstream object is consumed strictly through its own public fields
(duck-typed -- this module never subclasses or mutates them); the engine
itself computes no indicator, reads no candle, performs no market
structure analysis, executes no trade, and calculates no risk or position
size.

Live operation, entirely in memory::

    market_state = mse_engine.analyze()
    strategy_eval = strategy_engine.evaluate(market_state, "trend_following", symbol="EUR_USD", timeframe="M5")
    regression_pred = regression_engine.predict(market_state, symbol="EUR_USD", timeframe="M5")
    classification_pred = logistic_engine.predict(market_state, symbol="EUR_USD", timeframe="M5")

    decision = DecisionEngine(DecisionEngineConfig()).decide(
        market_state=market_state, strategy_evaluation=strategy_eval,
        regression_prediction=regression_pred, classification_prediction=classification_pred,
        symbol="EUR_USD", timeframe="M5",
    )
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from market_structure import MarketState
from training.utils import utc_timestamp

from .config import DecisionEngineConfig
from .decision_result import (
    DecisionMetadata,
    DecisionResult,
    LinearRegressionAnalysis,
    LogisticRegressionAnalysis,
    PositionSizePlaceholder,
)
from .explainability import build_explainability
from .market_analysis import build_market_analysis
from .recommendation import build_strategy_analysis
from .strategy_verdict import build_strategy_verdict
from .trade_plan_builder import build_trade_plan
from .validator import validate_decision_inputs
from .version import DECISION_ENGINE_VERSION


def _linear_regression_analysis(prediction: Optional[Any]) -> LinearRegressionAnalysis:
    if prediction is None:
        return LinearRegressionAnalysis(available=False)
    return LinearRegressionAnalysis(
        available=True, expected_close=prediction.expected_close, expected_high=prediction.expected_high,
        expected_low=prediction.expected_low, expected_return=prediction.expected_return,
        expected_pip_movement=prediction.expected_pip_move, expected_volatility=prediction.expected_volatility,
        expected_MFE=prediction.expected_MFE, expected_MAE=prediction.expected_MAE,
        prediction_confidence=prediction.prediction_confidence,
        prediction_interval=dict(prediction.prediction_interval),
    )


def _logistic_regression_analysis(prediction: Optional[Any]) -> LogisticRegressionAnalysis:
    if prediction is None:
        return LogisticRegressionAnalysis(available=False)
    return LogisticRegressionAnalysis(
        available=True, predicted_class=prediction.predicted_class,
        buy_probability=prediction.buy_probability, sell_probability=prediction.sell_probability,
        no_trade_probability=prediction.no_trade_probability,
        classification_confidence=prediction.prediction_confidence,
        probability_margin=prediction.probability_margin, entropy=prediction.prediction_entropy,
    )


class DecisionEngine:
    """Orchestration-only: combines Strategy/Linear/Logistic outputs into
    one ``DecisionResult``. Computes no indicator, no market structure, no
    trade execution, no risk/position sizing."""

    def __init__(self, config: Optional[DecisionEngineConfig] = None) -> None:
        self.config = config or DecisionEngineConfig()

    def decide(
        self, *, market_state: MarketState, strategy_evaluation: Any,
        regression_prediction: Optional[Any] = None, classification_prediction: Optional[Any] = None,
        symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN",
        historical_win_rate: Optional[float] = None, feature_version: Optional[str] = None,
    ) -> DecisionResult:
        validate_decision_inputs(strategy_evaluation)

        linear_regression = _linear_regression_analysis(regression_prediction)
        logistic_regression = _logistic_regression_analysis(classification_prediction)

        strategy_analysis, final_recommendation, _model_net = build_strategy_analysis(
            strategy_name=strategy_evaluation.strategy_name,
            strategy_recommendation=strategy_evaluation.recommendation,
            market_bias=strategy_evaluation.market_bias,
            strategy_compliance=strategy_evaluation.strategy_compliance,
            strategy_confidence=strategy_evaluation.strategy_confidence,
            strategy_overall_score=strategy_evaluation.overall_score,
            expected_return=linear_regression.expected_return,
            expected_pip_movement=linear_regression.expected_pip_movement,
            regression_confidence=linear_regression.prediction_confidence,
            predicted_class=logistic_regression.predicted_class,
            classification_confidence=logistic_regression.classification_confidence,
            config=self.config, mse_compliance=strategy_evaluation.mse_compliance,
        )

        prediction_horizon = None
        if regression_prediction is not None:
            prediction_horizon = regression_prediction.prediction_horizon
        elif classification_prediction is not None:
            prediction_horizon = classification_prediction.prediction_horizon

        trade_plan = build_trade_plan(
            final_recommendation=final_recommendation, market_state=market_state,
            linear_regression=linear_regression, consensus_score=strategy_analysis.consensus_score,
            opportunity_score=strategy_analysis.opportunity_score, prediction_horizon=prediction_horizon,
            timeframe=timeframe, config=self.config,
        )
        strategy_analysis.trade_quality_score = trade_plan.trade_quality_score

        market_analysis = build_market_analysis(
            market_state=market_state, market_bias=strategy_analysis.market_bias,
            consensus_score=strategy_analysis.consensus_score, regression_available=linear_regression.available,
            regression_prediction_confidence=linear_regression.prediction_confidence,
            regression_confidence_breakdown=(
                dict(regression_prediction.confidence_breakdown) if regression_prediction is not None else None
            ),
            classification_confidence_breakdown=(
                dict(classification_prediction.confidence_breakdown) if classification_prediction is not None else None
            ),
        )

        strategy_verdict = build_strategy_verdict(
            strategy_name=strategy_evaluation.strategy_name, strategy_overall_score=strategy_evaluation.overall_score,
            strategy_compliance=strategy_evaluation.strategy_compliance,
            market_quality_score=strategy_evaluation.market_quality_score,
            forecast_alignment=strategy_analysis.forecast_alignment,
            probability_alignment=strategy_analysis.probability_alignment,
            consensus_score=strategy_analysis.consensus_score, historical_win_rate=historical_win_rate,
        )

        reasoning = build_explainability(
            final_recommendation=final_recommendation, strategy_recommendation=strategy_evaluation.recommendation,
            market_bias=strategy_analysis.market_bias, strategy_name=strategy_evaluation.strategy_name,
            strategy_compliance=strategy_evaluation.strategy_compliance,
            strategy_confidence=strategy_evaluation.strategy_confidence,
            linear_regression=linear_regression, logistic_regression=logistic_regression,
            forecast_alignment=strategy_analysis.forecast_alignment,
            probability_alignment=strategy_analysis.probability_alignment,
            consensus_score=strategy_analysis.consensus_score, mse_alignment=strategy_analysis.mse_alignment,
        )

        resolved_feature_version = feature_version
        if resolved_feature_version is None:
            if regression_prediction is not None:
                resolved_feature_version = regression_prediction.feature_version
            elif classification_prediction is not None:
                resolved_feature_version = classification_prediction.feature_version
            else:
                resolved_feature_version = "unknown"

        metadata = DecisionMetadata(
            currency_pair=symbol, timeframe=timeframe, timestamp=utc_timestamp(),
            strategy_version=strategy_evaluation.strategy_version, feature_version=resolved_feature_version,
            market_structure_version=_market_structure_version(),
            linear_regression_version=(regression_prediction.model_version if regression_prediction is not None else "unknown"),
            logistic_regression_version=(classification_prediction.model_version if classification_prediction is not None else "unknown"),
            decision_engine_version=DECISION_ENGINE_VERSION,
        )

        return DecisionResult(
            recommendation=final_recommendation, strategy=strategy_analysis, linear_regression=linear_regression,
            logistic_regression=logistic_regression, trade_plan=trade_plan,
            position_size=PositionSizePlaceholder(),
            market_analysis=market_analysis, strategy_verdict=strategy_verdict, reasoning=reasoning, metadata=metadata,
        )


def _market_structure_version() -> str:
    import market_structure
    return market_structure.__version__
