"""Configuration for the Decision Engine.

Same convention as every other engine on this platform
(``market_structure.EngineConfig``, ``strategy.StrategyConfig``,
``logistic_regression.ClassificationConfig``): one explicit, serializable
dataclass, every threshold overridable, nothing read from global state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from .exceptions import InvalidConfigError


@dataclass(slots=True)
class DecisionEngineConfig:
    """Tunable parameters for combining Strategy/Linear/Logistic outputs
    into one :class:`~decision_engine.decision_result.DecisionResult`.

    Attributes
    ----------
    pip_size:
        Price units per pip (0.0001 for most FX pairs, 0.01 for JPY pairs) --
        used only to express already-known price distances in pips, never
        to compute a new indicator.
    stop_atr_multiple:
        Stop-loss distance from entry, in multiples of the Market Structure
        Engine's already-computed ATR (``market_state.indicators["atr"]``).
        This is geometric trade-plan construction, not a risk/position-size
        calculation -- the Risk Manager still owns how many lots to risk.
    take_profit_r_multiples:
        Three risk-multiple targets (e.g. ``(1.0, 2.0, 3.0)``) applied to
        the stop distance to place TP1/TP2/TP3.
    downgrade_opposition_threshold:
        A strategy BUY/SELL is downgraded to WAIT when the combined,
        confidence-weighted model vote opposes it by at least this much
        (0-1 scale) -- the Decision Engine may add caution, but per the
        spec it must never *upgrade* a WAIT/NO_TRADE into a trade the
        strategy itself didn't clear.
    decision_confidence_weights:
        Blend weights (``strategy``/``regression``/``classification``) for
        ``decision_confidence`` -- renormalized over whichever sources are
        actually present for this call.
    opportunity_score_weights:
        Blend weights (``compliance``/``overall_score``/``consensus``/
        ``forecast_strength``) for ``opportunity_score``.
    trade_quality_weights:
        Blend weights (``consensus``/``opportunity``/``target_feasibility``/
        ``risk_reward``) for ``trade_quality_score``.
    """

    pip_size: float = 0.0001
    stop_atr_multiple: float = 1.5
    take_profit_r_multiples: Tuple[float, float, float] = (1.0, 2.0, 3.0)
    downgrade_opposition_threshold: float = 0.5
    decision_confidence_weights: Dict[str, float] = field(
        default_factory=lambda: {"strategy": 0.4, "regression": 0.3, "classification": 0.3}
    )
    opportunity_score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "compliance": 0.3, "overall_score": 0.3, "consensus": 0.25, "forecast_strength": 0.15,
        }
    )
    trade_quality_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "consensus": 0.25, "opportunity": 0.35, "target_feasibility": 0.25, "risk_reward": 0.15,
        }
    )

    def __post_init__(self) -> None:
        if self.pip_size <= 0:
            raise InvalidConfigError("pip_size must be > 0.")
        if self.stop_atr_multiple <= 0:
            raise InvalidConfigError("stop_atr_multiple must be > 0.")
        if len(self.take_profit_r_multiples) != 3:
            raise InvalidConfigError("take_profit_r_multiples must have exactly 3 entries (TP1/TP2/TP3).")
        if list(self.take_profit_r_multiples) != sorted(self.take_profit_r_multiples):
            raise InvalidConfigError("take_profit_r_multiples must be strictly increasing (TP1 < TP2 < TP3).")
        if not (0.0 <= self.downgrade_opposition_threshold <= 1.0):
            raise InvalidConfigError("downgrade_opposition_threshold must be in [0, 1].")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["take_profit_r_multiples"] = list(self.take_profit_r_multiples)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionEngineConfig":
        d = dict(d)
        if "take_profit_r_multiples" in d:
            d["take_profit_r_multiples"] = tuple(d["take_profit_r_multiples"])
        return cls(**d)
