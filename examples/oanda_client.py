"""Shared OANDA v20 REST client helper, reused by every example/script that
needs real historical candles (never mock data).

Credentials are read from environment variables (see market_structure/.env):
    OANDA_API_KEY
    OANDA_ACCOUNT_ID
    OANDA_PRACTICE   ("True" / "False")
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / "market_structure" / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.environ.get("OANDA_API_KEY")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
PRACTICE = os.environ.get("OANDA_PRACTICE", "True").strip().lower() in ("1", "true", "yes")

BASE_URL = "https://api-fxpractice.oanda.com" if PRACTICE else "https://api-fxtrade.oanda.com"


def fetch_candles(instrument: str, granularity: str, count: int) -> pd.DataFrame:
    """Pull `count` most recent COMPLETE candles for `instrument` from OANDA.

    Requests bid/mid/ask pricing so we can populate the optional `spread`
    column from real quoted spread rather than leaving it out.
    """
    if not API_KEY or not ACCOUNT_ID:
        raise RuntimeError(
            "Missing OANDA_API_KEY / OANDA_ACCOUNT_ID environment variables. "
            f"Expected them to be loaded from {ENV_PATH}"
        )

    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {
        "count": count,
        "granularity": granularity,
        "price": "BAM",  # bid, ask, mid
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for c in payload["candles"]:
        if not c.get("complete", False):
            continue  # drop the still-forming candle
        mid = c["mid"]
        bid = c["bid"]
        ask = c["ask"]
        rows.append(
            {
                "timestamp": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": float(c["volume"]),
                "spread": float(ask["c"]) - float(bid["c"]),
                "tick_volume": float(c["volume"]),
            }
        )

    if not rows:
        raise RuntimeError(f"OANDA returned no complete candles for {instrument} {granularity}.")

    return pd.DataFrame(rows)
