"""Concrete, ready-to-use trading strategies built on the Strategy Engine.

Each module here defines one ``TradingStrategy`` subclass plus a
``default_config()`` factory returning a validated ``StrategyConfig`` (rule
weights summing to 100%) as a sensible starting point -- copy and edit it to
create your own named/versioned variant via ``strategy.StrategyLoader``.
"""
