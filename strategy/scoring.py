"""Category and overall scoring.

Every rule declares a ``category`` (``"technical"``, ``"market_quality"``, or
``"risk"``). This module aggregates rule results two different ways:

- **Category scores** (``technical_score``, ``market_quality_score``,
  ``risk_quality_score``): weight-renormalized average *within* each
  category, mirroring how :mod:`compliance` aggregates across all rules.
- **``weighted_score``**: a single flat weighted average across *every*
  configured rule (using the strategy's full, un-renormalized weights --
  ``NOT_APPLICABLE`` rules contribute 0, so they still pull this number
  down, unlike compliance/category scores which exclude them). This makes
  ``weighted_score`` answer "how well is the market satisfying the complete,
  as-configured rule set right now" rather than "of the rules that could
  fire, how well did they do."
- **``overall_score``**: a second-stage blend of the three category scores
  (default weights 50% technical / 30% market quality / 20% risk),
  independent of how many rules happen to fall in each category.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .rule_base import RuleResult, RuleStatus

DEFAULT_OVERALL_WEIGHTS: Dict[str, float] = {
    "technical": 0.5, "market_quality": 0.3, "risk": 0.2,
}


@dataclass(slots=True)
class ScoreBreakdown:
    technical_score: float
    market_quality_score: float
    risk_quality_score: float
    overall_score: float
    weighted_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "technical_score": round(self.technical_score, 2),
            "market_quality_score": round(self.market_quality_score, 2),
            "risk_quality_score": round(self.risk_quality_score, 2),
            "overall_score": round(self.overall_score, 2),
            "weighted_score": round(self.weighted_score, 2),
        }


def _category_score(rule_results: List[RuleResult], category: str) -> float:
    in_category = [r for r in rule_results if r.category == category]
    applicable = [r for r in in_category if r.status != RuleStatus.NOT_APPLICABLE]
    if not applicable:
        return 0.0
    total_weight = sum(r.weight for r in applicable)
    if total_weight <= 0:
        return 0.0
    return sum(r.weight * r.score for r in applicable) / total_weight


def compute_weighted_score(rule_results: List[RuleResult]) -> float:
    """Flat weighted average across *every* configured rule (NOT_APPLICABLE
    counts as 0 contribution, still consuming its share of weight)."""
    total_weight = sum(r.weight for r in rule_results)
    if total_weight <= 0:
        return 0.0
    contribution = sum(r.weight * r.score for r in rule_results if r.status != RuleStatus.NOT_APPLICABLE)
    return contribution / total_weight


def compute_scores(
    rule_results: List[RuleResult], overall_weights: Dict[str, float] | None = None
) -> ScoreBreakdown:
    weights = overall_weights or DEFAULT_OVERALL_WEIGHTS
    technical = _category_score(rule_results, "technical")
    market_quality = _category_score(rule_results, "market_quality")
    risk = _category_score(rule_results, "risk")
    overall = (
        technical * weights.get("technical", 0.0)
        + market_quality * weights.get("market_quality", 0.0)
        + risk * weights.get("risk", 0.0)
    )
    weighted = compute_weighted_score(rule_results)
    return ScoreBreakdown(
        technical_score=technical, market_quality_score=market_quality,
        risk_quality_score=risk, overall_score=overall, weighted_score=weighted,
    )
