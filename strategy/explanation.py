"""Structured, human-readable explanations for a strategy evaluation.

Turns the raw rule results plus compliance/confidence into an ordered list
of short sentences suitable for display in a UI, a log line, or a prompt fed
to the existing Agentic AI -- e.g.::

    Bullish market structure detected.
    EMA alignment confirmed.
    Liquidity sweep completed.
    Bullish BOS confirmed.
    Order Block respected.
    RSI within acceptable range.
    Compliance: 91%
    Confidence: 88%
"""
from __future__ import annotations

from typing import List

from .rule_base import RuleResult, RuleStatus

#: How many FAILing rules (beyond PASSes) to surface as explicit caveats.
MAX_FAILURE_CAVEATS = 3


def generate_explanations(
    rule_results: List[RuleResult], compliance: float, confidence: float, market_bias: str,
    warnings: List[str], mse_compliance: float | None = None,
) -> List[str]:
    """Build the ordered explanation list for one ``StrategyEvaluation``."""
    lines: List[str] = []

    bias_label = market_bias.replace("_", " ").title()
    lines.append(f"{bias_label} market structure detected.")

    passing = [r for r in rule_results if r.status == RuleStatus.PASS]
    passing.sort(key=lambda r: r.weight, reverse=True)
    for r in passing:
        lines.append(r.reason)

    failing = [r for r in rule_results if r.status == RuleStatus.FAIL]
    failing.sort(key=lambda r: r.weight, reverse=True)
    for r in failing[:MAX_FAILURE_CAVEATS]:
        lines.append(f"Caveat: {r.reason}")

    for w in warnings:
        lines.append(f"Warning: {w}")

    lines.append(f"Compliance: {compliance:.0f}%")
    lines.append(f"Confidence: {confidence:.0f}%")
    if mse_compliance is not None:
        lines.append(f"MSE Compliance: {mse_compliance:.0f}%")
    return lines
