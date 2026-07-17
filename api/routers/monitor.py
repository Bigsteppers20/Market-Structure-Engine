"""Model Monitor endpoints.

Fits a drift/performance baseline from the cached model's own training
split, then evaluates health/drift/calibration against its held-out test
split (ground truth already known -- no look-ahead) -- the same
computation a live monitoring loop would perform once real predictions
accumulate over time, packaged into a single request/response."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import pipeline
from ..schemas import MonitorHealthRequest

router = APIRouter(prefix="/monitor", tags=["Model Monitor"])


@router.post("/health")
def health(request: MonitorHealthRequest) -> dict:
    """Auto-trains (and caches) the requested model if needed, then returns
    its health score, drift report, retraining recommendation, and an
    Agentic-AI-style summary."""
    try:
        return pipeline.monitor_health(
            symbol=request.symbol, timeframe=request.timeframe, task_type=request.task_type,
            count=request.count, window_size=request.window_size, horizon=request.horizon, stride=request.stride,
        )
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
