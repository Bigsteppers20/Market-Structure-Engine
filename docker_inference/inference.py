"""Loads the Market Structure model in a BACKGROUND thread, started from
main.py's startup hook -- not at import time. Nothing outside this
container should import this module directly -- router.py (same folder)
is the only allowed caller.

Why background, not import-time: on Railway there's no volume mount, so a
future real-weights model needs to download weights first (network I/O,
can take a while). If that happened synchronously at import time, the
ASGI app would never finish constructing, uvicorn would never bind the
port, and /health would be unreachable (connection refused) instead of a
proper 503 -- Railway's healthcheck couldn't distinguish "still starting"
from "crashed". Running it in a background thread lets uvicorn start
serving immediately; is_ready() (and therefore /health, see main.py) stays
False/503 until loading actually finishes.

run() itself still calls ``market_structure.MarketStructureEngine`` from
the ``market-structure-engine`` distribution -- a deterministic,
rule-based feature engine, not a trained ML model. The artifact bundle
downloaded below (model.joblib, scaler.joblib, feature_pipeline.joblib,
config.json, feature_schema.json, metadata.json -- the full contents of
the private HF_REPO_ID model repo) is downloaded and loaded into
``_artifacts`` for interface/infrastructure completeness, but run() does
NOT yet consume it -- wiring real predictions from this bundle (mapping
the engine's feature output into the classifier's expected input, adding
prediction-time code) is a separate, not-yet-scoped change.

``USE_MOCK_MODEL=1`` skips both the download and the market_structure
import entirely and returns a small fixed response instead -- for
exercising the gateway/stack without the real (heavier) dependency chain,
e.g. in CI or a fast local loop.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

#: Directory snapshot_download populates (and, on repeat runs, reuses --
#: it dedupes against already-present files via the remote's own hashes,
#: so a volume-mounted MODEL_WEIGHTS_PATH that already has the bundle
#: doesn't re-hit the network). Empty string counts as unset, same
#: convention as gateway/registry.py's _resolve().
MODEL_WEIGHTS_PATH: str = os.environ.get("MODEL_WEIGHTS_PATH") or "/weights"
#: The private HF model repo holding the artifact bundle.
HF_REPO_ID: str = os.environ.get("HF_REPO_ID") or "IfaDevGeek/market_structure_engine"
#: Required whenever USE_MOCK_MODEL isn't set -- HF_REPO_ID above is
#: private. NEVER logged -- passed straight to snapshot_download, which
#: uses it to build its own Authorization header.
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
USE_MOCK_MODEL: bool = os.environ.get("USE_MOCK_MODEL", "0") == "1"

#: Every file in the published bundle (see HF_REPO_ID) -- all loaded and
#: logged individually, even though run() doesn't consume any of them yet
#: (see module docstring).
_JOBLIB_ARTIFACTS = ("model", "scaler", "feature_pipeline")
_JSON_ARTIFACTS = ("config", "feature_schema", "metadata")

_engine_cls = None
_config_cls = None
_artifacts: Dict[str, Any] = {}
_load_error: Optional[str] = None
_ready = False
_state_lock = threading.Lock()


def is_ready() -> bool:
    """Whether the model finished loading successfully. Reflects the
    background loader's current state -- see start_loading()."""
    with _state_lock:
        return _ready


def load_error() -> Optional[str]:
    """Human-readable reason is_ready() is False due to a failure. None
    while still loading (not yet failed) or once ready."""
    with _state_lock:
        return _load_error


def _mark_ready() -> None:
    global _ready
    with _state_lock:
        _ready = True


def _mark_failed(message: str) -> None:
    global _load_error
    logger.error(message)
    with _state_lock:
        _load_error = message


def _download_and_load_artifacts() -> None:
    """Pull the full HF_REPO_ID snapshot into MODEL_WEIGHTS_PATH (reusing
    any already-present, unchanged files via snapshot_download's own
    hash-based caching) and load every artifact into ``_artifacts``.
    Raises with a clear message on any failure -- the caller (_load())
    turns that into a failed load state, never a silent partial one.
    """
    if not HF_TOKEN:
        raise RuntimeError(
            f"HF_TOKEN is required to download the private model repo {HF_REPO_ID!r} -- "
            "refusing to attempt an unauthenticated download. Set HF_TOKEN (a "
            "read-scoped Hugging Face access token) and restart."
        )

    logger.info("Downloading model repo %s to %s ...", HF_REPO_ID, MODEL_WEIGHTS_PATH)
    start_time = time.monotonic()
    try:
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            local_dir=MODEL_WEIGHTS_PATH,
        )
    # Broad catch is deliberate: any failure here must produce a clear,
    # actionable message rather than a bare traceback. The message below
    # never includes the token -- only the repo id (not a secret) and the
    # underlying exception's own text (huggingface_hub's own errors, e.g.
    # "401 Client Error", never echo the token back).
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download model repo {HF_REPO_ID!r}: {exc}") from exc

    elapsed = time.monotonic() - start_time
    logger.info("Download complete: %s in %.1fs", local_dir, elapsed)

    root = Path(local_dir)
    for name in _JOBLIB_ARTIFACTS:
        path = root / f"{name}.joblib"
        if not path.is_file():
            raise RuntimeError(f"Expected artifact {path} was not present after download.")
        _artifacts[name] = joblib.load(path)
        logger.info("Loaded artifact: %s", path.name)

    for name in _JSON_ARTIFACTS:
        path = root / f"{name}.json"
        if not path.is_file():
            raise RuntimeError(f"Expected artifact {path} was not present after download.")
        _artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded artifact: %s", path.name)


def _load() -> None:
    """Runs in a background thread (see start_loading()): download and
    load every artifact from HF_REPO_ID, then import the market_structure
    engine. is_ready() stays False the entire time; /health stays 503 (see
    main.py) until this returns having called _mark_ready()."""
    global _engine_cls, _config_cls

    if USE_MOCK_MODEL:
        logger.warning("USE_MOCK_MODEL=1 -- skipping artifact download and market_structure import.")
        _mark_ready()
        return

    try:
        _download_and_load_artifacts()
    except Exception as exc:  # noqa: BLE001
        _mark_failed(str(exc))
        return

    try:
        from market_structure import EngineConfig as _EngineConfig
        from market_structure import MarketStructureEngine as _MarketStructureEngine

        _engine_cls = _MarketStructureEngine
        _config_cls = _EngineConfig
        logger.info("Market Structure Engine loaded (market_structure package importable).")
    # Broad catch is deliberate: any load-time failure means "not ready",
    # never a crash.
    except Exception as exc:  # noqa: BLE001
        _mark_failed(
            "Failed to import the market_structure package. Install it with: "
            "pip install git+https://github.com/Bigsteppers20/Market-Structure-Engine.git "
            f"(underlying error: {exc})"
        )
        return

    logger.info("Model ready.")
    _mark_ready()


def start_loading() -> None:
    """Kick off _load() in a background thread. Called once from main.py's
    startup hook -- never at import time, so the ASGI app can start
    accepting connections (and reporting 503 via /health) immediately
    instead of blocking on a potentially slow download."""
    threading.Thread(target=_load, name="model-loader", daemon=True).start()


def _mock_run(input: Dict[str, Any]) -> Dict[str, Any]:
    candles: List[Dict[str, Any]] = input.get("candles", [])
    return {
        "market_state": {"n_candles": float(len(candles)), "mock": 1.0},
        "n_candles": len(candles),
    }


def run(input: Dict[str, Any]) -> Dict[str, Any]:
    """Run the model on one already-validated request payload (a plain dict
    from ``PredictRequest.model_dump()``). Raises on failure -- the caller
    (``router.py``) is responsible for turning that into an HTTP response."""
    if USE_MOCK_MODEL:
        return _mock_run(input)

    if not is_ready():
        raise RuntimeError(load_error() or "Market Structure model is not ready.")

    import pandas as pd  # local import: keeps this heavy dependency out of the module's import-time cost

    candles: List[Dict[str, Any]] = input["candles"]
    window_size: int = input.get("window_size", 200)
    swing_window: int = input.get("swing_window", 5)

    df = pd.DataFrame(candles)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if window_size and len(df) > window_size:
        df = df.iloc[-window_size:].reset_index(drop=True)

    engine = _engine_cls(_config_cls(swing_window=swing_window))
    engine.load(df)
    engine.analyze()
    state = engine.market_state()

    return {"market_state": state.to_dict(), "n_candles": state.n_candles}
