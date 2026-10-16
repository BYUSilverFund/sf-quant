"""
The ``backtester`` module provides tools for simulating trading strategies
over historical data. It exposes a clean public API for running sequential,
parallel (Ray) and dynamic mean-variance backtests. The ``slurm`` subpackage
distributes these backtests across a SLURM cluster, one task per year.
"""

from .sequential import backtest_sequential
from .parallel import backtest_parallel
from .dynamic_parallel import dynamic_backtest_parallel

__all__ = ["backtest_sequential", "backtest_parallel", "dynamic_backtest_parallel"]
