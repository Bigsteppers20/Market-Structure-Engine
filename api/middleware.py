"""ASGI middleware attaching each engine package's own ``__version__`` to
every HTTP response as a header.

Replaces the earlier ``forex_dynamics`` bundle-package approach: instead of
an importable namespace a consumer would ``pip install``, "which library
versions produced this response" is now visible directly on the response
itself, for any HTTP client, with zero import required.

Each version is read once at import time from the package's own
``__version__`` -- never recomputed per request, never guessed or
hand-maintained separately from the source of truth.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

import decision_engine
import linear_regression
import logistic_regression
import market_structure
import ml_pipeline
import model_monitor
import strategy
import training

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
