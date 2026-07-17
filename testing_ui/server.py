"""Personal-testing-only local server for ``testing_ui/``.

Never imports or calls any model/engine code -- this is purely a static
file server for ``index.html`` plus a tiny local JSON-file-backed store for
paper positions (a browser page cannot write to disk on its own). The real
trading API (``uvicorn api.main:app``) is called only from the browser's
JS, never from here -- this process has no idea what a ``MarketState`` or a
``RegressionPrediction`` even is.

Run (from the project root, with the trading API already running
separately on its own port)::

    .venv\\Scripts\\python.exe testing_ui\\server.py [--port 8765] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "index.html"
POSITIONS_FILE = ROOT / "positions.json"

_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _pip_size(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("_JPY") else 0.0001


PENDING_DIRECTIONS = {"BUY_STOP", "BUY_LIMIT", "SELL_STOP", "SELL_LIMIT"}


def _load() -> Dict[str, List[Dict[str, Any]]]:
    with _lock:
        if not POSITIONS_FILE.exists():
            return {"pending": [], "open": [], "history": []}
        try:
            data = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"pending": [], "open": [], "history": []}
        data.setdefault("pending", [])
        data.setdefault("open", [])
        data.setdefault("history", [])
        return data


def _save(data: Dict[str, List[Dict[str, Any]]]) -> None:
    with _lock:
        POSITIONS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _realized_pips(position: Dict[str, Any], exit_price: float) -> float:
    entry = float(position["entry_price"])
    pip = _pip_size(position["symbol"])
    diff = exit_price - entry
    if position["direction"].upper() in ("SELL", "SHORT"):
        diff = -diff
    return round(diff / pip, 1)


class Handler(BaseHTTPRequestHandler):
    server_version = "TestingUI/1.0"

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter, prefixed default logging
        print(f"[testing_ui] {self.address_string()} - {fmt % args}")

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_index()
        elif self.path == "/api/positions":
            self._send_json(HTTPStatus.OK, _load())
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path {self.path!r}"})

    def do_POST(self) -> None:
        close_match = re.fullmatch(r"/api/positions/([^/]+)/close", self.path)
        trigger_match = re.fullmatch(r"/api/pending/([^/]+)/trigger", self.path)
        if self.path == "/api/positions":
            self._create_position()
        elif self.path == "/api/pending":
            self._create_pending()
        elif close_match:
            self._close_position(close_match.group(1))
        elif trigger_match:
            self._trigger_pending(trigger_match.group(1))
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path {self.path!r}"})

    def do_DELETE(self) -> None:
        delete_match = re.fullmatch(r"/api/positions/([^/]+)", self.path)
        pending_delete_match = re.fullmatch(r"/api/pending/([^/]+)", self.path)
        if delete_match:
            self._delete_position(delete_match.group(1))
        elif pending_delete_match:
            self._delete_pending(pending_delete_match.group(1))
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path {self.path!r}"})

    # ------------------------------------------------------------------ #
    def _serve_index(self) -> None:
        if not INDEX_FILE.exists():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "index.html not found next to server.py"})
            return
        body = INDEX_FILE.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _create_position(self) -> None:
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON body"})
            return
        required = ("symbol", "timeframe", "direction", "entry_price")
        missing = [k for k in required if not body.get(k)]
        if missing:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"missing fields: {missing}"})
            return
        record = {
            "id": str(uuid.uuid4()), "symbol": body["symbol"], "timeframe": body["timeframe"],
            "direction": body["direction"], "entry_price": float(body["entry_price"]),
            "stop_loss": body.get("stop_loss"), "take_profit": body.get("take_profit"),
            "size": body.get("size"), "thesis": body.get("thesis", ""),
            "model_recommendation": body.get("model_recommendation"),
            "opened_at": _utc_now(),
        }
        data = _load()
        data["open"].append(record)
        _save(data)
        self._send_json(HTTPStatus.CREATED, record)

    def _create_pending(self) -> None:
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON body"})
            return
        required = ("symbol", "timeframe", "direction", "trigger_price")
        missing = [k for k in required if not body.get(k)]
        if missing:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"missing fields: {missing}"})
            return
        direction = str(body["direction"]).upper()
        if direction not in PENDING_DIRECTIONS:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"direction must be one of {sorted(PENDING_DIRECTIONS)}"},
            )
            return
        record = {
            "id": str(uuid.uuid4()), "symbol": body["symbol"], "timeframe": body["timeframe"],
            "direction": direction, "trigger_price": float(body["trigger_price"]),
            "stop_loss": body.get("stop_loss"), "take_profit": body.get("take_profit"),
            "size": body.get("size"), "thesis": body.get("thesis", ""),
            "model_recommendation": body.get("model_recommendation"),
            "created_at": _utc_now(),
        }
        data = _load()
        data["pending"].append(record)
        _save(data)
        self._send_json(HTTPStatus.CREATED, record)

    def _trigger_pending(self, order_id: str) -> None:
        """Promote a pending order into an open position at a client-reported
        fill price. The browser -- not this server -- decides when a trigger
        condition is met, since only it talks to the trading API for live
        prices; this endpoint just performs the pending -> open transition."""
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON body"})
            return
        if "fill_price" not in body:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "fill_price is required"})
            return
        data = _load()
        idx = next((i for i, o in enumerate(data["pending"]) if o["id"] == order_id), None)
        if idx is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"pending order {order_id!r} not found"})
            return
        order = data["pending"].pop(idx)
        filled_direction = "BUY" if order["direction"].startswith("BUY") else "SELL"
        position = {
            "id": order["id"], "symbol": order["symbol"], "timeframe": order["timeframe"],
            "direction": filled_direction, "entry_price": float(body["fill_price"]),
            "stop_loss": order.get("stop_loss"), "take_profit": order.get("take_profit"),
            "size": order.get("size"), "thesis": order.get("thesis", ""),
            "model_recommendation": order.get("model_recommendation"),
            "opened_at": _utc_now(),
            "filled_from_pending": order["direction"], "trigger_price": order["trigger_price"],
        }
        data["open"].append(position)
        _save(data)
        self._send_json(HTTPStatus.OK, position)

    def _delete_pending(self, order_id: str) -> None:
        data = _load()
        before = len(data["pending"])
        data["pending"] = [o for o in data["pending"] if o["id"] != order_id]
        if len(data["pending"]) == before:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"pending order {order_id!r} not found"})
            return
        _save(data)
        self._send_json(HTTPStatus.OK, {"deleted": order_id})

    def _close_position(self, position_id: str) -> None:
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON body"})
            return
        if "exit_price" not in body:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "exit_price is required"})
            return
        data = _load()
        idx = next((i for i, p in enumerate(data["open"]) if p["id"] == position_id), None)
        if idx is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"open position {position_id!r} not found"})
            return
        position = data["open"].pop(idx)
        exit_price = float(body["exit_price"])
        position["exit_price"] = exit_price
        position["closed_at"] = _utc_now()
        position["realized_pips"] = _realized_pips(position, exit_price)
        data["history"].append(position)
        _save(data)
        self._send_json(HTTPStatus.OK, position)

    def _delete_position(self, position_id: str) -> None:
        data = _load()
        before = len(data["open"])
        data["open"] = [p for p in data["open"] if p["id"] != position_id]
        if len(data["open"]) == before:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"open position {position_id!r} not found"})
            return
        _save(data)
        self._send_json(HTTPStatus.OK, {"deleted": position_id})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personal testing UI local server (static file server + paper-position "
        "store). Never touches the model/engine code -- talks to the real trading API only "
        "from the browser."
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not POSITIONS_FILE.exists():
        _save({"pending": [], "open": [], "history": []})

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Personal testing UI: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    print(f"Positions stored at: {POSITIONS_FILE}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
