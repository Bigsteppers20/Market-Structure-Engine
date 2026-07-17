"""Strategy Engine endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from .. import pipeline

router = APIRouter(prefix="/strategy", tags=["Strategy Engine"])


@router.get("/list")
def list_strategies() -> dict:
    return {"strategies": pipeline.available_strategies()}


@router.get("/evaluate")
def evaluate(
    request: Request,
    symbol: str = Query(default="EUR_USD"),
    timeframe: str = Query(default="M5"),
    strategy_name: str = Query(default="trend_following", description="One of the registered Strategy Lab strategies -- see GET /strategy/list"),
    window_size: int = Query(default=200, ge=50),
    count: int = Query(default=250, ge=50, le=5000),
) -> dict:
    """Evaluate one registered strategy against the current live market.
    No training involved -- purely rule-based, deterministic, real-time.
    The live MarketState is built once by ``MarketStateMiddleware`` from
    these same query params and attached to ``request.state.market_state``.

    Example: ``GET /strategy/evaluate?strategy_name=trend_following``
    """
    try:
        evaluation = pipeline.evaluate_strategy(request.state.market_state, strategy_name, symbol, timeframe)
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return evaluation.to_dict()
