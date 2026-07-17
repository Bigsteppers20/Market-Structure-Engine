"""Decision Engine endpoints -- combines Strategy + Linear Regression +
Logistic Regression into one DecisionResult."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .. import pipeline
from ..schemas import DecisionRequest

router = APIRouter(prefix="/decision", tags=["Decision Engine"])


@router.post("/predict")
def predict(body: DecisionRequest, request: Request) -> dict:
    """Full pipeline: Market Structure Engine -> Strategy Engine + Linear
    Regression + Logistic Regression -> Decision Engine. Auto-trains (and
    caches) the two ML models for this symbol/timeframe on first call
    (~10-30s); every subsequent call is fast (~2s). Returns the complete
    ``DecisionResult`` JSON. The live MarketState is built once by
    ``MarketStateMiddleware`` from this same request body and attached to
    ``request.state.market_state``."""
    try:
        return pipeline.make_decision(
            symbol=body.symbol, timeframe=body.timeframe, strategy_name=body.strategy_name,
            market_state=request.state.market_state,
            count=body.count, window_size=body.window_size, horizon=body.horizon, stride=body.stride,
        )
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
