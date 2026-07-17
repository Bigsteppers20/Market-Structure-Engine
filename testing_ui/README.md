# Personal Testing UI

A personal tool for trying out the platform's predictions and paper-trading them, kept **completely
separate from the model/engine code**. This directory imports nothing from `linear_regression/`,
`logistic_regression/`, `decision_engine/`, `market_structure/`, or any other platform package — it
only talks HTTP to the already-running trading API, exactly like Postman would.

- `index.html` — the entire UI (markup + styles + JS). No build step, no framework.
- `server.py` — a tiny local server (Python standard library only, no new dependency). Serves
  `index.html` and persists your paper positions to `positions.json` — a static HTML page cannot
  write to disk on its own, so this is only a static file server + a JSON-file-backed store, never a
  caller of any model code.
- `positions.json` — created automatically on first run. Your personal test data, not part of the
  product.

## Run it

**Easiest:** double-click `testing_ui\start.bat`. It starts the trading API and this UI's server
each in their own window and opens the UI in your browser. Close those two windows (or Ctrl+C in
each) to stop everything.

**Manual (two terminals, from the project root):**

```
# 1. The real trading API (unmodified) -- if it's not already running
.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000

# 2. This UI's local server
.venv\Scripts\python.exe testing_ui\server.py
```

Then open **http://127.0.0.1:8765** in a browser.

If your trading API runs on a different host/port, change the "Trading API base URL" field at the
top of the page (saved in your browser's `localStorage`).

The pill in the top-right corner of the page pings `GET /health` every 15s and tells you plainly
whether the trading API is reachable, rather than waiting for you to hit **Analyze** and see a
cryptic "Failed to fetch."

## How to use it

1. Type a thesis, e.g. *"there is a bullish trend on EU, confirm if that's accurate and give me an
   entry on the 15 min timeframe"* — or just leave it blank and use the Symbol/Timeframe/Strategy
   dropdowns directly.
2. Click **Analyze**. The first call for a new symbol/timeframe pair trains both ML models
   server-side (up to ~30s, per `POST /decision/predict`'s own documented latency); every call after
   that for the same pair is fast (a couple of seconds).
3. The **Model Verdict** panel tells you whether your stated thesis was confirmed by the model's own
   market bias, plus its recommendation and reasoning. The **Suggested Entry** panel shows the
   Decision Engine's ready-made trade plan (entry/stop/take-profit/risk-reward).
4. Click **Place Position** and choose either a market order (**BUY**/**SELL**, fills immediately at
   the price you type) or a pending order (**BUY_STOP**/**BUY_LIMIT**/**SELL_STOP**/**SELL_LIMIT**,
   fills automatically once the price crosses your trigger). Pending orders show up under
   **Pending Orders**; market orders go straight to **Open Positions**.
5. Click **Refresh Prices** any time to pull the live current price (`GET /market/analyze`'s
   `pa_current_close` field) for every symbol you have open or pending. This also checks each
   pending order's trigger condition and, if hit, fills it into an open position automatically
   (`BUY_STOP`/`SELL_STOP` trigger when price crosses *through* the level in the breakout direction;
   `BUY_LIMIT`/`SELL_LIMIT` trigger on a pullback *to* the level). You can also **Cancel** a pending
   order before it triggers.
6. Click **Close** on an open position, enter an exit price (defaults to the last refreshed price),
   and it moves to **History** with a realized P&L.

## Notes

- Position size is just a plain number you type for your own bookkeeping — there's no Risk Manager
  in this platform yet, so no real position-sizing math happens here.
- P&L is reported in pips only (0.0001 per pip, 0.01 for `*_JPY` pairs), matching the rest of the
  platform's own convention.
- Nothing here is called by, or calls into, the Decision Engine, Model Monitor, or either ML engine's
  actual code — deleting this entire directory has zero effect on the rest of the platform.
