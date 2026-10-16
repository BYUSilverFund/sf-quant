"""SLURM backtesting: SLURM-based parallel backtesting for quantitative finance."""

from sf_quant.backtester.slurm.config import BacktestConfig, BacktestDynamicConfig, SlurmConfig
from sf_quant.backtester.slurm.runner import BacktestRunner, BacktestDynamicRunner

__version__ = "0.1.1"
__all__ = ["BacktestConfig", "BacktestDynamicConfig", "BacktestRunner", "BacktestDynamicRunner", "SlurmConfig"]
