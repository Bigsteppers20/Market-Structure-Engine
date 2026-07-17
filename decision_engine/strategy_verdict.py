"""Strategy Verdict construction (STRATEGY VERDICT section).

Validates the user's own Strategy Lab strategy against the live market
(Strategy Engine's own compliance/market-quality scores) and against the
two ML engines (forecast/probability alignment) -- it never overrides the
strategy itself; ``recommended_action`` is advice about the strategy's
current fitness, not a BUY/SELL/WAIT trade instruction.
"""
from __future__ import annotations

from typing import Optional

from .decision_result import StrategyVerdict


def _validation_status(strategy_compliance: float, consensus_score: float) -> str:
    if strategy_compliance >= 70.0 and consensus_score >= 60.0:
        return "Validated"
    if strategy_compliance >= 70.0 and consensus_score < 40.0:
        return "Conflicting"
    if strategy_compliance < 40.0:
        return "Not Validated"
    return "Partially Validated"


_RECOMMENDED_ACTIONS = {
    "Validated": "Strategy is well-aligned with current market conditions and model forecasts -- continue using it as configured.",
    "Conflicting": "Strategy rules are currently satisfied, but the Linear/Logistic Regression forecasts disagree with its bias -- review before increasing exposure.",
    "Not Validated": "Current market conditions do not satisfy this strategy's own rules -- avoid trading this setup until compliance improves.",
    "Partially Validated": "Mixed signals across the strategy and the two model forecasts -- proceed with caution and confirm with additional analysis.",
}


def build_strategy_verdict(
    *, strategy_name: str, strategy_overall_score: float, strategy_compliance: float,
    market_quality_score: float, forecast_alignment: float, probability_alignment: float,
    consensus_score: float, historical_win_rate: Optional[float],
) -> StrategyVerdict:
    validation_status = _validation_status(strategy_compliance, consensus_score)
    return StrategyVerdict(
        strategy_name=strategy_name,
        overall_strategy_quality=strategy_overall_score,
        historical_success_probability=historical_win_rate,
        live_market_alignment=(strategy_compliance + market_quality_score) / 2.0,
        model_alignment=(forecast_alignment + probability_alignment) / 2.0,
        validation_status=validation_status,
        recommended_action=_RECOMMENDED_ACTIONS[validation_status],
    )
