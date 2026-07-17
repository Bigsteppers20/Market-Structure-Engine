"""Smoke test for visualization.py (skipped when matplotlib is absent)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_structure import EngineConfig, MarketStructureEngine

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from market_structure.visualization import plot_market_structure  # noqa: E402


def test_plot_renders_and_saves(random_df: pd.DataFrame, tmp_path: Path) -> None:
    engine = MarketStructureEngine(EngineConfig(swing_window=3)).load(random_df)
    state = engine.analyze()
    out = tmp_path / "structure.png"
    fig = plot_market_structure(engine.data, state, last_n=200, save_path=out)
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)
