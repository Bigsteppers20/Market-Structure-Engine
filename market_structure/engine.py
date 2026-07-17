"""Market Structure Engine — public orchestrator.

Usage::

    engine = MarketStructureEngine()
    engine.load(data)              # DataFrame or List[Candle]
    engine.analyze()
    state = engine.market_state()  # MarketState dataclass
    vec, names = engine.feature_vector()

The engine transforms raw OHLCV data into numerical market-structure
features. It never produces BUY / SELL / NO_TRADE decisions.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .bos import BosEngine
from .candle_patterns import CandlePatternEngine
from .choch import ChochEngine
from .config import EngineConfig
from .data_loader import DataLoader, InputData
from .feature_vector import (
    MarketState,
    build_microstructure,
    build_price_action,
    build_session,
    build_structure,
    build_volatility,
)
from .fvg import FvgEngine
from .indicators import IndicatorEngine
from .liquidity import LiquidityEngine
from .order_blocks import OrderBlockEngine
from .spread import SpreadEngine
from .support_resistance import SupportResistanceEngine
from .swings import SwingDetector
from .trend import TrendEngine


class EngineStateError(RuntimeError):
    """Raised when the public API is called out of order."""


class MarketStructureEngine:
    """Coordinates every sub-engine and assembles the :class:`MarketState`."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self._loader = DataLoader(self.config)
        self._swing_detector = SwingDetector(self.config)
        self._trend_engine = TrendEngine(self.config)
        self._bos_engine = BosEngine(self.config)
        self._choch_engine = ChochEngine(self.config)
        self._sr_engine = SupportResistanceEngine(self.config)
        self._liquidity_engine = LiquidityEngine(self.config)
        self._fvg_engine = FvgEngine(self.config)
        self._ob_engine = OrderBlockEngine(self.config)
        self._indicator_engine = IndicatorEngine(self.config)
        self._pattern_engine = CandlePatternEngine(self.config)
        self._spread_engine = SpreadEngine(self.config)
        self._df: Optional[pd.DataFrame] = None
        self._state: Optional[MarketState] = None

    # ------------------------------------------------------------------ #
    def load(self, data: InputData) -> "MarketStructureEngine":
        """Validate and store input data; returns self for chaining."""
        self._df = self._loader.load(data)
        self._state = None
        return self

    def analyze(self) -> MarketState:
        """Run every sub-engine and build the market state."""
        if self._df is None:
            raise EngineStateError("Call load(data) before analyze().")
        df = self._df

        swings = self._swing_detector.detect(df)
        trend = self._trend_engine.analyze(df, swings)
        breaks = self._bos_engine.detect(df, swings)
        chochs = self._choch_engine.detect(breaks)
        zones = self._sr_engine.analyze(df, swings)
        liquidity = self._liquidity_engine.analyze(df, swings)
        fvg = self._fvg_engine.analyze(df)
        order_blocks = self._ob_engine.analyze(df)
        panel = self._indicator_engine.analyze(df)
        patterns = self._pattern_engine.analyze(df)
        spread = self._spread_engine.analyze(df)

        state = MarketState(
            n_candles=len(df),
            trend=trend,
            structure=build_structure(len(df), breaks, chochs),
            price_action=build_price_action(df, panel),
            volatility=build_volatility(df, panel, self.config),
            microstructure=build_microstructure(df, swings, trend),
            session=build_session(df, self.config),
            zones=zones,
            liquidity=liquidity,
            fvg=fvg,
            order_blocks=order_blocks,
            spread=spread,
            indicators=panel.snapshot,
            indicator_validity=panel.valid,
            patterns=patterns.snapshot,
            swings=swings,
            breaks=breaks,
            chochs=chochs,
        )
        self._state = state
        return state

    def market_state(self) -> MarketState:
        """Return the last computed :class:`MarketState`."""
        if self._state is None:
            raise EngineStateError("Call analyze() before market_state().")
        return self._state

    def feature_vector(self) -> Tuple[np.ndarray, List[str]]:
        """Return the flat numeric feature vector and its feature names."""
        return self.market_state().to_vector()

    @property
    def data(self) -> pd.DataFrame:
        """The validated candle DataFrame currently loaded."""
        if self._df is None:
            raise EngineStateError("No data loaded.")
        return self._df
