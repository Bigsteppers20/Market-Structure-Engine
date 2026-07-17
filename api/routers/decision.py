"""Decision Engine endpoints -- combines Strategy + Linear Regression +
Logistic Regression into one DecisionResult."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import pipeline
from ..schemas import DecisionRequest

router = APIRouter(prefix="/decision", tags=["Decision Engine"])


@router.post("/predict")
def predict(request: DecisionRequest) -> dict:
    """Full pipeline: Market Structure Engine -> Strategy Engine + Linear
    Regression + Logistic Regression -> Decision Engine. Auto-trains (and
    caches) the two ML models for this symbol/timeframe on first call
    (~10-30s); every subsequent call is fast (~2s). Returns the complete
    ``DecisionResult`` JSON."""
    try:
        return pipeline.make_decision(
            symbol=request.symbol, timeframe=request.timeframe, strategy_name=request.strategy_name,
            count=request.count, window_size=request.window_size, horizon=request.horizon, stride=request.stride,
        )
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
