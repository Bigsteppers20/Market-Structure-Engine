"""FastAPI app exposing every analytical engine for local testing (Postman,
curl, Swagger UI). Not a production trading API -- no auth, no rate
limiting, single-process in-memory model cache.

Run:

    .venv\\Scripts\\python.exe -m uvicorn api.main:app --reload --port 8000

Then:
    Swagger UI   -> http://127.0.0.1:8000/docs
    OpenAPI spec -> http://127.0.0.1:8000/openapi.json (import into Postman
                    via Import > Link, or save and Import > File)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import pipeline
from .middleware import EngineVersionsMiddleware
from .routers import decision, linear_regression, logistic_regression, market, monitor, strategy
from .schemas import DecisionRequest, TrainRequest

app = FastAPI(
    title="Forex Dynamics -- Platform API",
    description=(
        "Local testing API exposing every analytical engine: Market Structure, "
        "Strategy, Linear Regression, Logistic Regression, Model Monitor, and "
        "the combined Decision Engine. Fetches real OANDA practice-account "
        "candles -- this is for manual testing only, not a production or "
        "live-trading endpoint. No trade is ever executed."
    ),
    version="1.0.0",
)

app.add_middleware(EngineVersionsMiddleware)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(market.router)
app.include_router(strategy.router)
app.include_router(linear_regression.router)
app.include_router(logistic_regression.router)
app.include_router(decision.router)
app.include_router(monitor.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "Forex Dynamics Platform API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "GET /health": "Liveness check.",
            "GET /strategies": "List registered Strategy Lab strategies (alias of GET /strategy/list).",
            "GET /market/analyze": "Market Structure Engine -- live MarketState, no ML/strategy layered on top.",
            "GET /strategy/list": "List registered Strategy Lab strategies.",
            "GET /strategy/evaluate": "Strategy Engine -- evaluate one strategy against the live market.",
            "POST /linear-regression/train": "Train + cache a Linear Regression model.",
            "POST /linear-regression/predict": "Predict from the live market (auto-trains on first call).",
            "POST /logistic-regression/train": "Train + cache a Logistic Regression model.",
            "POST /logistic-regression/predict": (
                "Live production prediction (auto-trains on first call): returns ONLY "
                "{prediction: BUY|SELL|WAIT, prediction_confidence, prediction_horizon, "
                "model_version, feature_version, training_version, model_health, timestamp} "
                "-- no class probabilities or evaluation metrics."
            ),
            "POST /decision/predict": "Full Decision Engine pipeline (Strategy + both ML models combined).",
            "POST /monitor/health": "Model health/drift/retraining report for a cached model.",
            "POST /train": "[Deprecated alias] Trains both ML models together -- see /linear-regression/train + /logistic-regression/train.",
            "POST /decision": "[Deprecated alias] Same as POST /decision/predict.",
        },
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/strategies", tags=["meta"])
def strategies() -> dict:
    """Alias of GET /strategy/list, kept for backward compatibility."""
    return {"strategies": pipeline.available_strategies()}


# --------------------------------------------------------------------------- #
# Deprecated flat aliases -- kept so any existing client/collection built
# against the original two-endpoint API keeps working unchanged.
# --------------------------------------------------------------------------- #
@app.post("/train", tags=["deprecated"], deprecated=True)
def train_alias(request: TrainRequest) -> dict:
    """Deprecated: trains both Linear Regression and Logistic Regression
    together. Prefer POST /linear-regression/train and
    POST /logistic-regression/train."""
    try:
        reg_bundle = pipeline.train_regression(
            symbol=request.symbol, timeframe=request.timeframe, count=request.count,
            window_size=request.window_size, horizon=request.horizon, stride=request.stride,
        )
        cls_bundle = pipeline.train_classification(
            symbol=request.symbol, timeframe=request.timeframe, count=request.count,
            window_size=request.window_size, horizon=request.horizon, stride=request.stride,
        )
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "symbol": request.symbol, "timeframe": request.timeframe,
        "regression_test_metrics": reg_bundle.test_metrics,
        "classification_test_metrics": cls_bundle.test_metrics,
    }


@app.post("/decision", tags=["deprecated"], deprecated=True)
def decision_alias(request: DecisionRequest) -> dict:
    """Deprecated alias of POST /decision/predict."""
    try:
        return pipeline.make_decision(
            symbol=request.symbol, timeframe=request.timeframe, strategy_name=request.strategy_name,
            count=request.count, window_size=request.window_size, horizon=request.horizon, stride=request.stride,
        )
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
