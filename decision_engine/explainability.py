"""Deterministic explanation generation (EXPLAINABILITY section).

Every sentence is built from already-computed values (strategy compliance/
confidence/bias, regression/classification outputs, alignment/consensus
scores) -- nothing here is randomized or re-derives anything; the same
inputs always produce the same explanation strings.
"""
from __future__ import annotations

from typing import List

from .decision_result import Explainability, LinearRegressionAnalysis, LogisticRegressionAnalysis


def _strategy_compliance_explanation(strategy_name: str, market_bias: str, strategy_compliance: float, strategy_confidence: float) -> str:
    return (
        f"Strategy '{strategy_name}' reads the market as {market_bias} with "
        f"{strategy_compliance:.1f}% rule compliance and {strategy_confidence:.1f}% confidence."
    )


def _forecast_explanation(lr: LinearRegressionAnalysis) -> str:
    if not lr.available:
        return "Linear Regression forecast was not supplied for this decision."
    if lr.expected_pip_movement is not None:
        direction = "gain" if lr.expected_pip_movement >= 0 else "loss"
        return (
            f"Linear Regression forecasts a {abs(lr.expected_pip_movement):.1f} pip {direction} "
            f"over the prediction horizon (confidence {lr.prediction_confidence:.1f}%)."
        )
    return f"Linear Regression prediction confidence is {lr.prediction_confidence:.1f}%."


def _probability_explanation(lgr: LogisticRegressionAnalysis) -> str:
    if not lgr.available:
        return "Logistic Regression classification was not supplied for this decision."
    return (
        f"Logistic Regression assigns BUY={lgr.buy_probability:.1%}, SELL={lgr.sell_probability:.1%}, "
        f"NO_TRADE={lgr.no_trade_probability:.1%} (predicted class: {lgr.predicted_class}, "
        f"confidence {lgr.classification_confidence:.1f}%)."
    )


def _why(
    direction_label: str, market_bias: str, lr: LinearRegressionAnalysis, lgr: LogisticRegressionAnalysis,
    bias_matches: bool, mse_alignment: float,
) -> str:
    reasons: List[str] = []
    if bias_matches:
        reasons.append(f"strategy bias is {market_bias}")
    if direction_label == "BUY":
        if lr.available and lr.expected_pip_movement is not None and lr.expected_pip_movement > 0:
            reasons.append(f"Linear Regression forecasts +{lr.expected_pip_movement:.1f} pips")
        if lgr.available and lgr.buy_probability is not None:
            reasons.append(f"Logistic Regression BUY probability is {lgr.buy_probability:.1%}")
    elif direction_label == "SELL":
        if lr.available and lr.expected_pip_movement is not None and lr.expected_pip_movement < 0:
            reasons.append(f"Linear Regression forecasts {lr.expected_pip_movement:.1f} pips")
        if lgr.available and lgr.sell_probability is not None:
            reasons.append(f"Logistic Regression SELL probability is {lgr.sell_probability:.1%}")
    else:  # WAIT
        if lgr.available and lgr.no_trade_probability is not None:
            reasons.append(f"Logistic Regression NO_TRADE probability is {lgr.no_trade_probability:.1%}")
        if not bias_matches:
            if mse_alignment < 70.0:
                reasons.append(
                    f"the Market Structure Engine's own independent trend only confirms this "
                    f"{market_bias} bias at {mse_alignment:.0f}% (needs 70%)"
                )
            else:
                reasons.append(f"strategy bias ({market_bias}) does not clearly support this direction")
    if not reasons:
        return f"No strong evidence currently supports {direction_label}."
    return f"{direction_label} is supported by: " + "; ".join(reasons) + "."


def build_explainability(
    *, final_recommendation: str, strategy_recommendation: str, market_bias: str,
    strategy_name: str, strategy_compliance: float, strategy_confidence: float,
    linear_regression: LinearRegressionAnalysis, logistic_regression: LogisticRegressionAnalysis,
    forecast_alignment: float, probability_alignment: float, consensus_score: float,
    mse_alignment: float = 100.0,
) -> Explainability:
    supporting: List[str] = []
    opposing: List[str] = []

    if forecast_alignment == 100.0:
        supporting.append("Linear Regression forecast direction agrees with the strategy's market bias.")
    elif forecast_alignment == 0.0:
        opposing.append("Linear Regression forecast direction opposes the strategy's market bias.")

    if probability_alignment == 100.0:
        supporting.append("Logistic Regression's predicted class agrees with the strategy's recommendation.")
    elif probability_alignment == 0.0:
        opposing.append("Logistic Regression's predicted class opposes the strategy's recommendation.")

    if market_bias != "NEUTRAL" and mse_alignment >= 70.0:
        supporting.append(
            f"Market Structure Engine's independent trend confirms the strategy's {market_bias} "
            f"bias ({mse_alignment:.0f}% alignment)."
        )
    elif market_bias != "NEUTRAL" and mse_alignment < 70.0:
        opposing.append(
            f"Market Structure Engine's independent trend does NOT confirm the strategy's "
            f"{market_bias} bias ({mse_alignment:.0f}% alignment, needs 70%)."
        )

    if strategy_compliance >= 70.0:
        supporting.append(f"Strategy rule compliance is high ({strategy_compliance:.1f}%).")
    elif strategy_compliance < 40.0:
        opposing.append(f"Strategy rule compliance is low ({strategy_compliance:.1f}%).")

    if final_recommendation != strategy_recommendation:
        opposing.append(
            f"Final recommendation was downgraded from the strategy's own {strategy_recommendation} to "
            f"{final_recommendation} due to model disagreement."
        )

    summary = (
        f"Final recommendation: {final_recommendation} (from strategy recommendation {strategy_recommendation}), "
        f"consensus score {consensus_score:.1f}/100."
    )

    bullish_bias = market_bias in ("BULLISH", "STRONG_BULLISH")
    bearish_bias = market_bias in ("BEARISH", "STRONG_BEARISH")

    return Explainability(
        supporting_factors=supporting, opposing_factors=opposing, summary=summary,
        why_buy=_why("BUY", market_bias, linear_regression, logistic_regression, bullish_bias, mse_alignment),
        why_sell=_why("SELL", market_bias, linear_regression, logistic_regression, bearish_bias, mse_alignment),
        why_wait=_why(
            "WAIT", market_bias, linear_regression, logistic_regression,
            not (bullish_bias or bearish_bias), mse_alignment,
        ),
        strategy_compliance_explanation=_strategy_compliance_explanation(strategy_name, market_bias, strategy_compliance, strategy_confidence),
        forecast_explanation=_forecast_explanation(linear_regression),
        probability_explanation=_probability_explanation(logistic_regression),
    )
