"""ASGI middleware for the trading API.

- ``EngineVersionsMiddleware`` attaches each engine package's own
  ``__version__`` to every HTTP response as a header. Replaces the earlier
  ``forex_dynamics`` bundle-package approach: instead of an importable
  namespace a consumer would ``pip install``, "which library versions
  produced this response" is now visible directly on the response itself,
  for any HTTP client, with zero import required.

- ``MarketStateMiddleware`` builds the live ``MarketState`` once per
  request (for the routes that need one) and attaches it to
  ``request.state.market_state``, so route handlers stop each calling
  ``pipeline.build_market_state()`` themselves -- the fetch-candles +
  run-Market-Structure-Engine step was previously duplicated across five
  routers.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

import decision_engine
import linear_regression
import logistic_regression
import market_structure
import ml_pipeline
import model_monitor
import strategy
import training

from . import pipeline

#: header name -> the owning package's own __version__. `strategies` (the
#: concrete Strategy Lab strategies) and `api` itself are deliberately
#: excluded: neither defines a single package-level __version__ -- each
#: strategy versions itself independently, and api's version is already
#: FastAPI's own `app.version` (see /openapi.json).
_VERSION_HEADERS: dict[str, str] = {
    "X-Market-Structure-Version": market_structure.__version__,
    "X-Ml-Pipeline-Version": ml_pipeline.__version__,
    "X-Training-Version": training.__version__,
    "X-Strategy-Version": strategy.__version__,
    "X-Linear-Regression-Version": linear_regression.__version__,
    "X-Logistic-Regression-Version": logistic_regression.__version__,
    "X-Model-Monitor-Version": model_monitor.__version__,
    "X-Decision-Engine-Version": decision_engine.__version__,
}


class EngineVersionsMiddleware(BaseHTTPMiddleware):
    """Adds one ``X-*-Version`` response header per engine package to every
    request this app serves, regardless of route."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for header, version in _VERSION_HEADERS.items():
            response.headers[header] = version
        return response


#: Only routes whose handler actually consumes a *live* MarketState
#: directly -- NOT /*/train or /monitor/health, which only ever build a
#: historical training Dataset (a completely different, much larger fetch)
#: and never touch a live market_state at all; running this middleware for
#: them would be a wasted OANDA call + MSE computation nothing reads.
#:
#: `count_applies_to_market_state` mirrors each route's own pre-existing
#: behavior exactly: GET routes' `count` query param always fed the live
#: market_state fetch; the POST predict/decision routes' `count` field only
#: ever sized the (separately cached) *training* dataset -- their live
#: market_state fetch always left `count` at build_market_state()'s own
#: default (`window_size + 10`). Changing that mapping would silently
#: change fetch size, so it's preserved as-is here.
_MARKET_STATE_ROUTES: dict[tuple[str, str], bool] = {
    ("GET", "/market/analyze"): True,
    ("GET", "/strategy/evaluate"): True,
    ("POST", "/linear-regression/predict"): False,
    ("POST", "/logistic-regression/predict"): False,
    ("POST", "/decision/predict"): False,
    ("POST", "/decision"): False,  # deprecated alias of /decision/predict
}

_DEFAULT_SYMBOL = "EUR_USD"
_DEFAULT_TIMEFRAME = "M5"
_DEFAULT_WINDOW_SIZE = 200
_DEFAULT_COUNT = 250


class MarketStateMiddleware(BaseHTTPMiddleware):
    """Builds ``MarketState`` once per request for the routes that need a
    live one, attaching it to ``request.state.market_state``. Routes not in
    ``_MARKET_STATE_ROUTES`` (including ones with no symbol/timeframe at
    all, like ``/health`` or ``/docs``) pass through untouched."""

    async def dispatch(self, request: Request, call_next) -> Response:
        route_key = (request.method, request.url.path)
        if route_key not in _MARKET_STATE_ROUTES:
            return await call_next(request)
        count_applies = _MARKET_STATE_ROUTES[route_key]

        try:
            if request.method == "GET":
                params = request.query_params
                symbol = params.get("symbol", _DEFAULT_SYMBOL)
                timeframe = params.get("timeframe", _DEFAULT_TIMEFRAME)
                window_size = int(params.get("window_size", _DEFAULT_WINDOW_SIZE))
                count = int(params.get("count", _DEFAULT_COUNT)) if count_applies else None
            else:
                try:
                    body = await request.json()
                except ValueError:
                    body = {}
                symbol = body.get("symbol", _DEFAULT_SYMBOL)
                timeframe = body.get("timeframe", _DEFAULT_TIMEFRAME)
                window_size = int(body.get("window_size", _DEFAULT_WINDOW_SIZE))
                count = int(body["count"]) if count_applies and "count" in body else None
        except (TypeError, ValueError):
            # Malformed window_size/count (e.g. non-numeric). Don't error out
            # here -- pass through unmodified so FastAPI's own Query()/Pydantic
            # validation runs in the handler and returns its normal, correctly
            # -shaped 422; the handler never reaches request.state.market_state
            # in that case since FastAPI rejects the request before the
            # handler body executes.
            return await call_next(request)

        try:
            request.state.market_state = pipeline.build_market_state(
                symbol, timeframe, window_size, count=count,
            )
        except pipeline.PipelineError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        return await call_next(request)
