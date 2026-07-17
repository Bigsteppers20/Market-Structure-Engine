"""Change of Character (CHOCH) detection.

A CHOCH is the *first* break of structure against the prevailing structural
direction: after a run of bullish breaks, the first bearish break is a
bearish CHOCH (and vice versa). Subsequent same-direction breaks are plain
BOS continuations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import pandas as pd

from .bos import StructureBreak
from .config import EngineConfig

Direction = Literal["bullish", "bearish"]


@dataclass(slots=True)
class ChochEvent:
    """A confirmed change of character.

    Attributes
    ----------
    direction:
        Direction of the *new* character (``"bullish"`` = shift up).
    index:
        Bar index where the reversal break closed.
    price:
        Structural level whose break flipped the character.
    strength:
        ATR-normalized breach distance inherited from the causal break.
    timestamp:
        Timestamp of the reversal candle.
    """

    direction: Direction
    index: int
    price: float
    strength: float
    timestamp: pd.Timestamp


class ChochEngine:
    """Derives CHOCH events from an ordered list of structure breaks."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def detect(self, breaks: List[StructureBreak]) -> List[ChochEvent]:
        """Return every direction flip in the break sequence."""
        events: List[ChochEvent] = []
        prev_dir: Direction | None = None
        for brk in breaks:
            if prev_dir is not None and brk.direction != prev_dir:
                events.append(
                    ChochEvent(
                        direction=brk.direction,
                        index=brk.index,
                        price=brk.price,
                        strength=brk.strength,
                        timestamp=brk.timestamp,
                    )
                )
            prev_dir = brk.direction
        return events
