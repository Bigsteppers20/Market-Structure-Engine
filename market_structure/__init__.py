"""Market Structure Engine (MSE).

Transforms raw OHLCV forex data into a comprehensive numerical representation
of the current market state (trend, structure, liquidity, zones, indicators,
patterns, sessions). Emits features only — never trade decisions.
"""
from .bos import BosEngine, StructureBreak
from .candle import Candle, candles_to_dataframe, dataframe_to_candles
from .candle_patterns import CandlePatternEngine, PatternPanel
from .choch import ChochEngine, ChochEvent
from .config import EngineConfig
from .data_loader import DataLoader, DataValidationError
from .engine import EngineStateError, MarketStructureEngine
from .feature_vector import (
    MarketState,
    MicrostructureFeatures,
    PriceActionFeatures,
    SessionFeatures,
    StructureFeatures,
    VolatilityFeatures,
)
from .fvg import FairValueGap, FvgEngine, FvgState
from .indicators import IndicatorEngine, IndicatorPanel
from .liquidity import LiquidityEngine, LiquidityPool, LiquiditySweep, LiquidityState
from .order_blocks import OrderBlock, OrderBlockEngine, OrderBlockState
from .spread import SpreadEngine, SpreadFeatures
from .support_resistance import SupportResistanceEngine, Zone, ZoneSummary
from .swings import SwingDetector, SwingPoint
from .trend import TrendDirection, TrendEngine, TrendState

__version__ = "1.0.0"

__all__ = [
    "MarketStructureEngine",
    "EngineConfig",
    "EngineStateError",
    "MarketState",
    "Candle",
    "candles_to_dataframe",
    "dataframe_to_candles",
    "DataLoader",
    "DataValidationError",
    "SwingDetector",
    "SwingPoint",
    "TrendEngine",
    "TrendState",
    "TrendDirection",
    "BosEngine",
    "StructureBreak",
    "ChochEngine",
    "ChochEvent",
    "SupportResistanceEngine",
    "Zone",
    "ZoneSummary",
    "LiquidityEngine",
    "LiquidityPool",
    "LiquiditySweep",
    "LiquidityState",
    "FvgEngine",
    "FairValueGap",
    "FvgState",
    "OrderBlockEngine",
    "OrderBlock",
    "OrderBlockState",
    "SpreadEngine",
    "SpreadFeatures",
    "IndicatorEngine",
    "IndicatorPanel",
    "CandlePatternEngine",
    "PatternPanel",
    "PriceActionFeatures",
    "VolatilityFeatures",
    "MicrostructureFeatures",
    "SessionFeatures",
    "StructureFeatures",
]
