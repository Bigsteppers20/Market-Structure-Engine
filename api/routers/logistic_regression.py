"""Logistic Regression Engine endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .. import pipeline
from ..schemas import PredictRequest, TrainRequest

router = APIRouter(prefix="/logistic-regression", tags=["Logistic Regression Engine"])


@router.post("/train")
def train(request: TrainRequest) -> dict:
    """(Re)train and cache a Logistic Regression model (SELL/NO_TRADE/BUY)
    for this symbol/timeframe. Takes ~10-30s."""
    try:
        bundle = pipeline.train_classification(
            symbol=request.symbol, timeframe=request.timeframe, count=request.count,
            window_size=request.window_size, horizon=request.horizon, stride=request.stride,
        )
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "symbol": bundle.symbol, "timeframe": bundle.timeframe, "classes": bundle.classes,
        "test_metrics": bundle.test_metrics,
    }


@router.post("/predict")
def predict(body: PredictRequest, request: Request) -> dict:
    """Predict from the current live market. Auto-trains (and caches) on
    first call for this symbol/timeframe -- slow the first time (~10-30s),
    fast afterward. Returns the full ``ClassificationPrediction``. The live
    MarketState is built once by ``MarketStateMiddleware`` from this same
    request body and attached to ``request.state.market_state``."""
    try:
        bundle = pipeline.get_or_train_classification(
            symbol=body.symbol, timeframe=body.timeframe, count=body.count,
            window_size=body.window_size, horizon=body.horizon, stride=body.stride,
        )
        return pipeline.predict_classification(bundle, request.state.market_state)
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
