"""Optional visualization of the analyzed market structure.

Requires matplotlib (an optional dependency). Renders candles with swings,
structure breaks, S/R zones, FVGs and order blocks overlaid — useful for
debugging and documentation, never required by the analysis pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .feature_vector import MarketState


def plot_market_structure(
    df: pd.DataFrame,
    state: MarketState,
    last_n: int = 300,
    save_path: Optional[Path] = None,
):
    """Plot candles and overlays for the last ``last_n`` bars.

    Parameters
    ----------
    df:
        Validated OHLCV DataFrame (``engine.data``).
    state:
        Result of ``engine.analyze()``.
    last_n:
        Number of most recent candles to display.
    save_path:
        When given, the figure is written to disk instead of shown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "matplotlib is required for visualization: pip install matplotlib") from exc

    start = max(len(df) - last_n, 0)
    view = df.iloc[start:]
    x = view.index.to_numpy()

    fig, ax = plt.subplots(figsize=(14, 7))
    up = view["close"] >= view["open"]
    ax.vlines(x, view["low"], view["high"], color="#666", linewidth=0.6, zorder=1)
    ax.bar(x[up], (view["close"] - view["open"])[up], bottom=view["open"][up],
           width=0.7, color="#2e7d32", zorder=2)
    ax.bar(x[~up], (view["close"] - view["open"])[~up], bottom=view["open"][~up],
           width=0.7, color="#c62828", zorder=2)

    for s in state.swings:
        if s.index < start:
            continue
        marker, color = ("v", "#c62828") if s.kind == "high" else ("^", "#2e7d32")
        ax.scatter(s.index, s.price, marker=marker, color=color, s=36, zorder=3)

    for b in state.breaks:
        if b.index < start:
            continue
        color = "#1565c0" if b.direction == "bullish" else "#ef6c00"
        ax.axhline(b.price, xmin=0, xmax=1, color=color, linewidth=0.5, alpha=0.35)
        ax.annotate("BOS", (b.index, b.close), fontsize=7, color=color)

    def _zone(lower: float, upper: float, i0: int, color: str) -> None:
        i0 = max(i0, start)
        ax.add_patch(Rectangle((i0, lower), x[-1] - i0, max(upper - lower, 1e-9),
                               facecolor=color, alpha=0.15, zorder=0))

    if state.zones:
        for z in state.zones.support_zones[:5]:
            _zone(z.lower, z.upper, z.last_touch_index, "#2e7d32")
        for z in state.zones.resistance_zones[:5]:
            _zone(z.lower, z.upper, z.last_touch_index, "#c62828")
    if state.fvg:
        for g in state.fvg.gaps:
            if not g.filled and g.index >= start:
                _zone(g.lower, g.upper, g.index, "#6a1b9a")
    if state.order_blocks:
        for ob in state.order_blocks.blocks:
            if not ob.mitigated and ob.index >= start:
                _zone(ob.lower, ob.upper, ob.index, "#1565c0")

    ax.set_title("Market Structure")
    ax.set_xlabel("bar index")
    ax.set_ylabel("price")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    return fig
