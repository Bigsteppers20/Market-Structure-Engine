"""Compliance: how closely the current market satisfies the strategy.

Compliance is a straightforward weight-renormalized average of every
*applicable* rule's score -- ``NOT_APPLICABLE`` rules are excluded from both
the numerator and the denominator so a strategy isn't penalized just because
one of its rules had nothing to evaluate (e.g. no order block exists yet).
This is deliberately the *simple* aggregate; :mod:`confidence` computes a
genuinely independent, differently-shaped signal (see that module).
"""
from __future__ import annotations

from typing import List

from .rule_base import RuleResult, RuleStatus


def compute_compliance(rule_results: List[RuleResult]) -> float:
    """Weighted average score (0-100) across applicable rules only."""
    applicable = [r for r in rule_results if r.status != RuleStatus.NOT_APPLICABLE]
    if not applicable:
        return 0.0
    total_weight = sum(r.weight for r in applicable)
    if total_weight <= 0:
        return 0.0
    return sum(r.weight * r.score for r in applicable) / total_weight
