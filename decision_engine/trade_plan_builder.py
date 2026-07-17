"""Trade Plan construction (TRADE PLAN section).

Price levels are geometric placements derived from the Market Structure
Engine's already-computed ATR (``market_state.indicators["atr"]``) and
current close (``market_state.price_action.current_close``) -- this module
never computes an indicator, reads a raw candle, or performs market
structure analysis itself. It also never calculates position size or
account risk (see ``decision_result.PositionSizePlaceholder`` -- that stays
the Risk Manager's job).
"""
from __future__ import annotations

from typing import Optional

from market_structure import MarketState

from .config import DecisionEngineConfig
from .decision_result import LinearRegressionAnalysis, TradePlan
from .recommendation import _renormalized_blend  # same package, same reuse convention as trainer.py::_brier_macro


def build_trade_plan(
    *, final_recommendation: str, market_state: MarketState, linear_regression: LinearRegressionAnalysis,
    consensus_score: float, opportunity_score: float, prediction_horizon: Optional[int], timeframe: str,
    config: DecisionEngineConfig,
) -> TradePlan:
    if final_recommendation not in ("BUY", "SELL"):
        return TradePlan(
            direction="NONE",
            trade_quality_score=_renormalized_blend(
                {"consensus": consensus_score, "opportunity": opportunity_score}, config.trade_quality_weights,
            ),
        )

    direction = 1 if final_recommendation == "BUY" else -1
    entry_price = market_state.price_action.current_close
    atr = market_state.indicators.get("atr")
    atr_valid = market_state.indicator_validity.get("atr", 0.0) >= 1.0

    if not atr or atr <= 0 or not atr_valid:
        # No reliable ATR yet (indicator warm-up) -- report direction/entry only,
        # never fabricate a stop/target from an invalid or missing ATR reading.
        return TradePlan(
            direction=final_recommendation, entry_price=entry_price, target_feasibility=0.0,
            trade_quality_score=_renormalized_blend(
                {"consensus": consensus_score, "opportunity": opportunity_score}, config.trade_quality_weights,
            ),
        )

    stop_distance = atr * config.stop_atr_multiple
    stop_loss = entry_price - direction * stop_distance
    r1, r2, r3 = config.take_profit_r_multiples
    take_profit_1 = entry_price + direction * stop_distance * r1
    take_profit_2 = entry_price + direction * stop_distance * r2
    take_profit_3 = entry_price + direction * stop_distance * r3
    risk_reward_ratio = r2  # TP2 is, by construction, r2 multiples of the stop distance

    expected_pip_gain = abs(take_profit_2 - entry_price) / config.pip_size
    if linear_regression.available and linear_regression.expected_pip_movement is not None:
        same_direction = (linear_regression.expected_pip_movement > 0) == (direction > 0)
        if same_direction:
            # Prefer the model's own forecast magnitude when it agrees with
            # the trade's direction -- a more informative headline number
            # than the fixed geometric TP2 distance.
            expected_pip_gain = abs(linear_regression.expected_pip_movement)

    if linear_regression.available and linear_regression.expected_MAE is not None:
        expected_maximum_drawdown = abs(linear_regression.expected_MAE) / config.pip_size
    else:
        expected_maximum_drawdown = stop_distance / config.pip_size

    planned_gain_price_units = abs(take_profit_2 - entry_price)
    if linear_regression.available and linear_regression.expected_MFE and linear_regression.expected_MFE > 0:
        target_feasibility = min(100.0, 100.0 * linear_regression.expected_MFE / planned_gain_price_units)
    else:
        # No forecasted favorable-excursion reference available -- fall back
        # to cross-engine agreement as the best available feasibility proxy.
        target_feasibility = consensus_score

    expected_holding_time = (
        f"{prediction_horizon} candle{'s' if prediction_horizon != 1 else ''} ({timeframe})"
        if prediction_horizon else "Unknown"
    )

    trade_quality_score = _renormalized_blend(
        {
            "consensus": consensus_score, "opportunity": opportunity_score,
            "target_feasibility": target_feasibility, "risk_reward": min(100.0, risk_reward_ratio / 3.0 * 100.0),
        },
        config.trade_quality_weights,
    )

    return TradePlan(
        direction=final_recommendation, entry_price=entry_price, stop_loss=stop_loss,
        take_profit_1=take_profit_1, take_profit_2=take_profit_2, take_profit_3=take_profit_3,
        risk_reward_ratio=risk_reward_ratio, expected_holding_time=expected_holding_time,
        expected_pip_gain=expected_pip_gain, expected_maximum_drawdown=expected_maximum_drawdown,
        target_feasibility=target_feasibility, trade_quality_score=trade_quality_score,
    )
