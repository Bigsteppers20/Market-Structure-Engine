"""Versioning for the Decision Engine.

The Decision Engine trains nothing and has no dataset/schema of its own to
version -- its ``metadata.decision_engine_version`` exists purely so a
consumer can tell which combination logic produced a given
:class:`~decision_engine.decision_result.DecisionResult`. Every other
version in ``metadata`` (strategy/feature/market-structure/linear-
regression/logistic-regression) is read directly from the upstream
engines' own already-computed outputs -- never recomputed here.
"""
from __future__ import annotations

#: Bump on any change to the combination/scoring/trade-plan logic that
#: could affect a previously-produced DecisionResult's interpretation.
DECISION_ENGINE_VERSION = "1.0.0"
