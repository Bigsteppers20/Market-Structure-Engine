"""Rule interface and the built-in rule library.

Every rule consumes only a ``market_structure.MarketState`` instance -- never
raw candles, never the broker, never a hand-computed indicator. All values a
rule reads already exist on ``MarketState`` (``trend``, ``structure``,
``zones``, ``liquidity``, ``fvg``, ``order_blocks``, ``spread``, ``session``,
``volatility``, ``microstructure``, ``price_action``, ``indicators``,
``indicator_validity``). A rule that finds its required data unavailable
(e.g. a ``_valid`` flag is 0, or no zone/gap/block exists yet) returns
``NOT_APPLICABLE`` rather than guessing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Tuple

from market_structure import MarketState


class RuleStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(slots=True)
class RuleResult:
    """Outcome of one rule's evaluation against a single ``MarketState``.

    Attributes
    ----------
    score:
        0-100, how strongly the rule's condition is satisfied (not merely
        binary -- used for weighted aggregation even on a FAIL).
    confidence:
        0-100, how certain the rule is about its own reading (driven by the
        underlying engine's ``_valid`` flags, signal freshness, and how
        clear-cut vs. borderline the reading is).
    weight / weighted_score:
        Filled in by :class:`~strategy.rule_engine.RuleEngine` after
        evaluation -- a bare ``Rule`` never knows its own weight.
    metadata:
        Free-form extras; the reserved key ``"direction"`` (``-1``/``0``/``1``)
        is read by bias aggregation and confidence's indicator-agreement
        component for directional rules.
    """

    rule_name: str
    category: str
    status: RuleStatus
    score: float
    confidence: float
    reason: str
    weight: float = 0.0
    weighted_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> int:
        return int(self.metadata.get("direction", 0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name, "category": self.category,
            "status": self.status.value, "score": round(self.score, 2),
            "confidence": round(self.confidence, 2), "reason": self.reason,
            "weight": round(self.weight, 4), "weighted_score": round(self.weighted_score, 4),
            "metadata": self.metadata,
        }


class Rule(ABC):
    """A single, independent, market-structure-only trading rule."""

    name: ClassVar[str]
    category: ClassVar[str]  # "technical" | "market_quality" | "risk"

    @abstractmethod
    def evaluate(self, market_state: MarketState) -> RuleResult:
        """Evaluate this rule against ``market_state``. Never mutates it."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _result(
    rule: "Rule", status: RuleStatus, score: float, confidence: float, reason: str,
    direction: int = 0, **metadata: Any,
) -> RuleResult:
    return RuleResult(
        rule_name=rule.name, category=rule.category, status=status,
        score=max(0.0, min(100.0, score)), confidence=max(0.0, min(100.0, confidence)),
        reason=reason, metadata={"direction": direction, **metadata},
    )


def _not_applicable(rule: "Rule", reason: str) -> RuleResult:
    return RuleResult(
        rule_name=rule.name, category=rule.category, status=RuleStatus.NOT_APPLICABLE,
        score=0.0, confidence=0.0, reason=reason, metadata={"direction": 0},
    )


def get_indicator(market_state: MarketState, name: str) -> Tuple[float, bool]:
    """Read one indicator plus its validity flag; missing key => invalid."""
    value = market_state.indicators.get(name)
    valid = market_state.indicator_validity.get(name, 0.0) >= 0.5
    if value is None:
        return 0.0, False
    return float(value), valid


# --------------------------------------------------------------------------- #
# 1. Trend Rule
# --------------------------------------------------------------------------- #
class TrendRule(Rule):
    name = "trend"
    category = "technical"

    def __init__(self, min_strength: float = 0.3) -> None:
        self.min_strength = min_strength

    def evaluate(self, market_state: MarketState) -> RuleResult:
        trend = market_state.trend
        if trend is None or not trend.valid:
            return _not_applicable(self, "Insufficient swing structure to assess trend.")
        direction = int(trend.direction)
        score = trend.strength * 100.0
        if direction == 0:
            score = min(score, 40.0)
        status = RuleStatus.PASS if direction != 0 and trend.strength >= self.min_strength else RuleStatus.FAIL
        confidence = min(100.0, 50.0 + trend.duration_bars * 2.0)
        label = "Bullish" if direction > 0 else "Bearish" if direction < 0 else "Sideways"
        reason = f"{label} trend, strength {trend.strength:.2f}, {trend.duration_bars} bars old."
        return _result(self, status, score, confidence, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 2. EMA Alignment Rule
# --------------------------------------------------------------------------- #
class EmaAlignmentRule(Rule):
    name = "ema_alignment"
    category = "technical"

    def evaluate(self, market_state: MarketState) -> RuleResult:
        ema20, v20 = get_indicator(market_state, "ema_20")
        ema50, v50 = get_indicator(market_state, "ema_50")
        ema200, v200 = get_indicator(market_state, "ema_200")
        if not (v20 and v50 and v200):
            return _not_applicable(self, "EMA(20/50/200) not yet warmed up.")
        close = market_state.price_action.current_close
        bullish = close > ema20 > ema50 > ema200
        bearish = close < ema20 < ema50 < ema200
        if bullish or bearish:
            direction = 1 if bullish else -1
            score, confidence = 100.0, 90.0
        else:
            direction = 1 if ema20 > ema50 else -1 if ema20 < ema50 else 0
            score, confidence = 45.0, 50.0
        status = RuleStatus.PASS if (bullish or bearish) else RuleStatus.FAIL
        label = "bullish (close>20>50>200)" if bullish else "bearish (close<20<50<200)" if bearish else "mixed/unaligned"
        reason = f"EMA stack {label}."
        return _result(self, status, score, confidence, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 3. Swing Structure Rule
# --------------------------------------------------------------------------- #
class SwingStructureRule(Rule):
    name = "swing_structure"
    category = "technical"

    def __init__(self, min_impulse_ratio: float = 1.2, max_retracement_pct: float = 70.0) -> None:
        self.min_impulse_ratio = min_impulse_ratio
        self.max_retracement_pct = max_retracement_pct

    def evaluate(self, market_state: MarketState) -> RuleResult:
        micro = market_state.microstructure
        trend = market_state.trend
        if micro is None or not micro.valid or trend is None or not trend.valid:
            return _not_applicable(self, "Not enough swings to assess structure quality.")
        clean = micro.impulse_ratio >= self.min_impulse_ratio and micro.retracement_pct <= self.max_retracement_pct
        if clean:
            score = min(100.0, 60.0 + micro.impulse_ratio * 10.0)
            confidence = 80.0
        else:
            score = max(0.0, 50.0 - max(0.0, micro.retracement_pct - self.max_retracement_pct))
            confidence = 55.0
        status = RuleStatus.PASS if clean else RuleStatus.FAIL
        reason = f"Impulse/correction ratio {micro.impulse_ratio:.2f}, retracement {micro.retracement_pct:.1f}%."
        return _result(self, status, score, confidence, reason, direction=int(trend.direction))


# --------------------------------------------------------------------------- #
# 4. Break Of Structure Rule
# --------------------------------------------------------------------------- #
class BreakOfStructureRule(Rule):
    name = "break_of_structure"
    category = "technical"

    def __init__(self, max_bars_since: int = 15) -> None:
        self.max_bars_since = max_bars_since

    def evaluate(self, market_state: MarketState) -> RuleResult:
        s = market_state.structure
        if s is None or s.last_bos_direction == 0.0:
            return _not_applicable(self, "No confirmed break of structure yet.")
        direction = int(s.last_bos_direction)
        bars_since = s.bars_since_bos
        fresh = 0 <= bars_since <= self.max_bars_since
        base = max(0.0, 100.0 - (bars_since / max(self.max_bars_since, 1)) * 60.0) if fresh else 25.0
        score = min(100.0, base + min(s.last_bos_strength, 2.0) * 15.0)
        status = RuleStatus.PASS if fresh and s.last_bos_strength > 0 else RuleStatus.FAIL
        confidence = 85.0 if fresh else 45.0
        label = "Bullish" if direction > 0 else "Bearish"
        reason = f"{label} BOS {int(bars_since)} bars ago (strength {s.last_bos_strength:.2f} ATR)."
        return _result(self, status, score, confidence, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 5. CHOCH Rule
# --------------------------------------------------------------------------- #
class ChochRule(Rule):
    name = "choch"
    category = "technical"

    def __init__(self, max_bars_since: int = 20) -> None:
        self.max_bars_since = max_bars_since

    def evaluate(self, market_state: MarketState) -> RuleResult:
        s = market_state.structure
        if s is None or s.last_choch_direction == 0.0:
            return _not_applicable(self, "No change of character detected yet.")
        direction = int(s.last_choch_direction)
        bars_since = s.bars_since_choch
        fresh = 0 <= bars_since <= self.max_bars_since
        base = max(0.0, 100.0 - (bars_since / max(self.max_bars_since, 1)) * 70.0) if fresh else 20.0
        score = min(100.0, base + min(s.last_choch_strength, 2.0) * 10.0)
        status = RuleStatus.PASS if fresh else RuleStatus.FAIL
        confidence = 75.0 if fresh else 40.0
        label = "Bullish" if direction > 0 else "Bearish"
        reason = f"{label} CHOCH {int(bars_since)} bars ago ({int(s.choch_count)} total flips)."
        return _result(self, status, score, confidence, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 6. Support Rule
# --------------------------------------------------------------------------- #
class SupportRule(Rule):
    name = "support"
    category = "market_quality"

    def __init__(self, max_distance_atr: float = 1.5) -> None:
        self.max_distance_atr = max_distance_atr

    def evaluate(self, market_state: MarketState) -> RuleResult:
        z = market_state.zones
        if z is None or z.nearest_support is None:
            return _not_applicable(self, "No support zone identified.")
        dist = z.distance_to_support
        near = 0.0 <= dist <= self.max_distance_atr
        base = max(0.0, 100.0 - (dist / max(self.max_distance_atr, 0.01)) * 60.0) if near else 30.0
        score = min(100.0, base + min(z.nearest_support.strength, 5.0) * 5.0)
        status = RuleStatus.PASS if near else RuleStatus.FAIL
        confidence = min(100.0, 50.0 + z.nearest_support.touches * 10.0)
        reason = (
            f"Nearest support {dist:.2f} ATR away, strength {z.nearest_support.strength:.2f}, "
            f"{z.nearest_support.touches} touches."
        )
        return _result(self, status, score, confidence, reason, direction=1)


# --------------------------------------------------------------------------- #
# 7. Resistance Rule
# --------------------------------------------------------------------------- #
class ResistanceRule(Rule):
    name = "resistance"
    category = "market_quality"

    def __init__(self, max_distance_atr: float = 1.5) -> None:
        self.max_distance_atr = max_distance_atr

    def evaluate(self, market_state: MarketState) -> RuleResult:
        z = market_state.zones
        if z is None or z.nearest_resistance is None:
            return _not_applicable(self, "No resistance zone identified.")
        dist = z.distance_to_resistance
        near = 0.0 <= dist <= self.max_distance_atr
        base = max(0.0, 100.0 - (dist / max(self.max_distance_atr, 0.01)) * 60.0) if near else 30.0
        score = min(100.0, base + min(z.nearest_resistance.strength, 5.0) * 5.0)
        status = RuleStatus.PASS if near else RuleStatus.FAIL
        confidence = min(100.0, 50.0 + z.nearest_resistance.touches * 10.0)
        reason = (
            f"Nearest resistance {dist:.2f} ATR away, strength {z.nearest_resistance.strength:.2f}, "
            f"{z.nearest_resistance.touches} touches."
        )
        return _result(self, status, score, confidence, reason, direction=-1)


# --------------------------------------------------------------------------- #
# 8. Liquidity Sweep Rule
# --------------------------------------------------------------------------- #
class LiquiditySweepRule(Rule):
    name = "liquidity_sweep"
    category = "technical"

    def __init__(self, max_bars_since: int = 10) -> None:
        self.max_bars_since = max_bars_since

    def evaluate(self, market_state: MarketState) -> RuleResult:
        lq = market_state.liquidity
        if lq is None or lq.last_sweep is None:
            return _not_applicable(self, "No liquidity sweep detected.")
        sweep = lq.last_sweep
        # A sweep of buy-side liquidity ("above") often precedes a bearish
        # reversal (smart money took out longs' stops); a sweep "below" a
        # bearish reversal precursor is bullish (shorts' stops taken out).
        direction = -1 if sweep.direction == "above" else 1
        bars_since = market_state.n_candles - 1 - sweep.index
        fresh = 0 <= bars_since <= self.max_bars_since
        base = max(0.0, 100.0 - (bars_since / max(self.max_bars_since, 1)) * 70.0) if fresh else 20.0
        score = min(100.0, base + min(sweep.size, 2.0) * 15.0)
        status = RuleStatus.PASS if fresh else RuleStatus.FAIL
        confidence = 80.0 if fresh else 40.0
        bias = "bearish" if direction < 0 else "bullish"
        reason = f"Liquidity swept {sweep.direction} {bars_since} bars ago (size {sweep.size:.2f} ATR) -- {bias} signal."
        return _result(self, status, score, confidence, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 9. Fair Value Gap Rule
# --------------------------------------------------------------------------- #
class FairValueGapRule(Rule):
    name = "fair_value_gap"
    category = "technical"

    def __init__(self, max_distance_atr: float = 2.0) -> None:
        self.max_distance_atr = max_distance_atr

    def evaluate(self, market_state: MarketState) -> RuleResult:
        g = market_state.fvg
        if g is None or g.nearest is None:
            return _not_applicable(self, "No unfilled fair value gap nearby.")
        direction = 1 if g.nearest.direction == "bullish" else -1
        near = g.distance_to_nearest <= self.max_distance_atr
        base = max(0.0, 100.0 - (g.distance_to_nearest / max(self.max_distance_atr, 0.01)) * 70.0) if near else 20.0
        score = min(100.0, base + min(g.nearest.size_atr, 2.0) * 10.0)
        status = RuleStatus.PASS if near else RuleStatus.FAIL
        confidence = 70.0 if near else 40.0
        reason = (
            f"Nearest unfilled {g.nearest.direction} FVG {g.distance_to_nearest:.2f} ATR away "
            f"(size {g.nearest.size_atr:.2f} ATR)."
        )
        return _result(self, status, score, confidence, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 10. Order Block Rule
# --------------------------------------------------------------------------- #
class OrderBlockRule(Rule):
    name = "order_block"
    category = "technical"

    def __init__(self, max_distance_atr: float = 2.0, min_freshness: float = 0.5) -> None:
        self.max_distance_atr = max_distance_atr
        self.min_freshness = min_freshness

    def evaluate(self, market_state: MarketState) -> RuleResult:
        ob = market_state.order_blocks
        if ob is None or ob.nearest is None:
            return _not_applicable(self, "No unmitigated order block nearby.")
        direction = 1 if ob.nearest.direction == "bullish" else -1
        near = ob.distance_to_nearest <= self.max_distance_atr
        base = max(0.0, 100.0 - (ob.distance_to_nearest / max(self.max_distance_atr, 0.01)) * 60.0) if near else 25.0
        score = min(100.0, base + ob.nearest.freshness * 30.0 + min(ob.nearest.strength, 2.0) * 5.0)
        status = RuleStatus.PASS if near and ob.nearest.freshness >= self.min_freshness else RuleStatus.FAIL
        confidence = min(100.0, 40.0 + ob.nearest.freshness * 60.0)
        reason = (
            f"Nearest unmitigated {ob.nearest.direction} order block {ob.distance_to_nearest:.2f} ATR away, "
            f"freshness {ob.nearest.freshness:.2f}."
        )
        return _result(self, status, score, confidence, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 11. Volume Rule
# --------------------------------------------------------------------------- #
class VolumeRule(Rule):
    name = "volume"
    category = "market_quality"

    def __init__(self, min_relative_volume: float = 0.8) -> None:
        self.min_relative_volume = min_relative_volume

    def evaluate(self, market_state: MarketState) -> RuleResult:
        rel, valid = get_indicator(market_state, "relative_volume")
        spike, spike_valid = get_indicator(market_state, "volume_spike")
        if not valid:
            return _not_applicable(self, "Relative volume not available.")
        score = min(100.0, rel * 50.0)
        status = RuleStatus.PASS if rel >= self.min_relative_volume else RuleStatus.FAIL
        confidence = 70.0
        spike_note = ", spike detected" if spike_valid and spike >= 0.5 else ""
        reason = f"Relative volume {rel:.2f}x average{spike_note}."
        return _result(self, status, score, confidence, reason)


# --------------------------------------------------------------------------- #
# 12. ATR Rule
# --------------------------------------------------------------------------- #
class AtrRule(Rule):
    name = "atr"
    category = "risk"

    def evaluate(self, market_state: MarketState) -> RuleResult:
        atr, valid = get_indicator(market_state, "atr")
        vol = market_state.volatility
        if not valid or vol is None or not vol.valid:
            return _not_applicable(self, "ATR/volatility regime not available.")
        if vol.expansion:
            score, status, note = 35.0, RuleStatus.FAIL, "expansion regime (elevated risk)"
        elif vol.compression:
            score, status, note = 65.0, RuleStatus.PASS, "compression regime (tight ranges)"
        else:
            score, status, note = 85.0, RuleStatus.PASS, "normal regime"
        reason = f"ATR={atr:.5f}, volatility {note}."
        return _result(self, status, score, 75.0, reason)


# --------------------------------------------------------------------------- #
# 13. RSI Rule
# --------------------------------------------------------------------------- #
class RsiRule(Rule):
    name = "rsi"
    category = "technical"

    def __init__(self, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self.oversold = oversold
        self.overbought = overbought

    def evaluate(self, market_state: MarketState) -> RuleResult:
        rsi, valid = get_indicator(market_state, "rsi")
        if not valid:
            return _not_applicable(self, "RSI not yet warmed up.")
        if rsi >= self.overbought:
            direction = -1
            score = max(0.0, 100.0 - (rsi - self.overbought) * 2.0)
            status = RuleStatus.FAIL
            reason = f"RSI {rsi:.1f} overbought (>{self.overbought:.0f})."
        elif rsi <= self.oversold:
            direction = 1
            score = max(0.0, 100.0 - (self.oversold - rsi) * 2.0)
            status = RuleStatus.FAIL
            reason = f"RSI {rsi:.1f} oversold (<{self.oversold:.0f})."
        else:
            direction = 1 if rsi >= 50.0 else -1
            score = min(100.0, 60.0 + abs(rsi - 50.0))
            status = RuleStatus.PASS
            reason = f"RSI {rsi:.1f} within acceptable range."
        return _result(self, status, score, 70.0, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 14. MACD Rule
# --------------------------------------------------------------------------- #
class MacdRule(Rule):
    name = "macd"
    category = "technical"

    def evaluate(self, market_state: MarketState) -> RuleResult:
        macd, v1 = get_indicator(market_state, "macd")
        signal, v2 = get_indicator(market_state, "macd_signal")
        if not (v1 and v2):
            return _not_applicable(self, "MACD not yet warmed up.")
        histogram = macd - signal
        direction = 1 if histogram > 0 else -1 if histogram < 0 else 0
        magnitude = abs(histogram) / max(abs(macd), abs(signal), 1e-9)
        score = min(100.0, 50.0 + magnitude * 200.0) if direction != 0 else 50.0
        status = RuleStatus.PASS if direction != 0 else RuleStatus.FAIL
        label = "bullish" if direction > 0 else "bearish" if direction < 0 else "flat"
        reason = f"MACD {label} crossover (histogram={histogram:+.6f})."
        return _result(self, status, score, 65.0, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 15. Session Rule
# --------------------------------------------------------------------------- #
class SessionRule(Rule):
    name = "session"
    category = "market_quality"

    def __init__(self, preferred_sessions: Tuple[str, ...] = ("is_london", "is_newyork")) -> None:
        self.preferred_sessions = preferred_sessions

    def evaluate(self, market_state: MarketState) -> RuleResult:
        s = market_state.session
        if s is None:
            return _not_applicable(self, "Session data unavailable.")
        all_sessions = ("is_sydney", "is_asian", "is_london", "is_newyork")
        active = [n for n in all_sessions if getattr(s, n) >= 0.5]
        preferred_active = [n for n in self.preferred_sessions if getattr(s, n, 0.0) >= 0.5]
        overlap = s.is_london >= 0.5 and s.is_newyork >= 0.5
        if overlap:
            score, status = 100.0, RuleStatus.PASS
        elif preferred_active:
            score, status = 75.0, RuleStatus.PASS
        else:
            score, status = 30.0, RuleStatus.FAIL
        reason = f"Active sessions: {active or ['none']}, hour={int(s.hour)} UTC."
        return _result(self, status, score, 90.0, reason)


# --------------------------------------------------------------------------- #
# 16. Spread Rule
# --------------------------------------------------------------------------- #
class SpreadRule(Rule):
    name = "spread"
    category = "risk"

    def __init__(self, max_percentile: float = 0.8) -> None:
        self.max_percentile = max_percentile

    def evaluate(self, market_state: MarketState) -> RuleResult:
        sp = market_state.spread
        if sp is None or not sp.valid:
            return _not_applicable(self, "Spread data unavailable (no broker spread supplied).")
        if sp.spike >= 0.5:
            score, status = 20.0, RuleStatus.FAIL
            reason = f"Spread spike detected (current={sp.current:.6f}, {sp.percentile * 100:.0f}th percentile)."
        elif sp.percentile <= self.max_percentile:
            score, status = 100.0 - sp.percentile * 50.0, RuleStatus.PASS
            reason = f"Spread normal ({sp.percentile * 100:.0f}th percentile)."
        else:
            score, status = 45.0, RuleStatus.FAIL
            reason = f"Spread elevated ({sp.percentile * 100:.0f}th percentile)."
        return _result(self, status, score, 80.0, reason)


# --------------------------------------------------------------------------- #
# 17. Volatility Rule
# --------------------------------------------------------------------------- #
class VolatilityRule(Rule):
    name = "volatility"
    category = "risk"

    def __init__(self, min_hv: float = 0.00005, max_hv: float = 0.01) -> None:
        self.min_hv = min_hv
        self.max_hv = max_hv

    def evaluate(self, market_state: MarketState) -> RuleResult:
        vol = market_state.volatility
        if vol is None or not vol.valid:
            return _not_applicable(self, "Historical volatility unavailable.")
        hv = vol.historical_volatility
        in_range = self.min_hv <= hv <= self.max_hv
        score = 85.0 if in_range else (30.0 if hv > self.max_hv else 40.0)
        status = RuleStatus.PASS if in_range else RuleStatus.FAIL
        reason = f"Historical volatility {hv:.6f} {'within' if in_range else 'outside'} acceptable range."
        return _result(self, status, score, 70.0, reason)


# --------------------------------------------------------------------------- #
# 18. Momentum Rule
# --------------------------------------------------------------------------- #
class MomentumRule(Rule):
    name = "momentum"
    category = "technical"

    def evaluate(self, market_state: MarketState) -> RuleResult:
        momentum, v1 = get_indicator(market_state, "momentum")
        roc, v2 = get_indicator(market_state, "roc")
        if not (v1 and v2):
            return _not_applicable(self, "Momentum/ROC not yet warmed up.")
        direction = 1 if roc > 0 else -1 if roc < 0 else 0
        score = min(100.0, 50.0 + abs(roc) * 20.0)
        status = RuleStatus.PASS if direction != 0 else RuleStatus.FAIL
        reason = f"ROC {roc:+.2f}%, momentum {momentum:+.6f}."
        return _result(self, status, score, 60.0, reason, direction=direction)


# --------------------------------------------------------------------------- #
# 19. Risk Rule -- composite risk-quality gate
# --------------------------------------------------------------------------- #
class RiskRule(Rule):
    name = "risk"
    category = "risk"

    def evaluate(self, market_state: MarketState) -> RuleResult:
        issues = []
        sp = market_state.spread
        if sp is not None and sp.valid and sp.spike >= 0.5:
            issues.append("spread spike")
        vol = market_state.volatility
        if vol is not None and vol.valid and vol.expansion:
            issues.append("volatility expansion")
        s = market_state.session
        if s is not None and s.is_sydney >= 0.5 and s.is_asian < 0.5 and s.is_london < 0.5 and s.is_newyork < 0.5:
            issues.append("low-liquidity session")

        if issues:
            score = max(10.0, 70.0 - len(issues) * 25.0)
            status = RuleStatus.FAIL
            reason = "Risk concerns: " + ", ".join(issues) + "."
        else:
            score, status = 90.0, RuleStatus.PASS
            reason = "No elevated risk conditions detected."
        return _result(self, status, score, 75.0, reason)


#: name -> class, used by StrategyLoader / concrete strategies to build rule
#: instances by name without importing every class individually.
BUILTIN_RULES: Dict[str, type] = {
    cls.name: cls  # type: ignore[attr-defined]
    for cls in (
        TrendRule, EmaAlignmentRule, SwingStructureRule, BreakOfStructureRule, ChochRule,
        SupportRule, ResistanceRule, LiquiditySweepRule, FairValueGapRule, OrderBlockRule,
        VolumeRule, AtrRule, RsiRule, MacdRule, SessionRule, SpreadRule, VolatilityRule,
        MomentumRule, RiskRule,
    )
}


def build_all_rules(rule_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Rule]:
    """Instantiate every built-in rule, applying any per-rule constructor
    overrides from ``rule_params`` (``{rule_name: {kwarg: value}}``).

    Every concrete strategy's ``build_rules()`` can simply return this --
    the full library is always *available*; a strategy's ``StrategyConfig``
    controls which subset actually runs and with what weight.
    """
    rule_params = rule_params or {}
    return {name: cls(**rule_params.get(name, {})) for name, cls in BUILTIN_RULES.items()}
