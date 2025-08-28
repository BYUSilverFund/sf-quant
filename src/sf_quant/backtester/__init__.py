"""
The ``backtester`` module provides tools for simulating trading strategies
over historical data. It exposes a clean public API for running backtests,
generating performance metrics, and analyzing trade signals, while keeping
internal execution details hidden.
"""
from .sequential import backtest_sequential
from .parallel import backtest_parallel

__all__ = [
    "backtest_sequential",
    "backtest_parallel"
]
