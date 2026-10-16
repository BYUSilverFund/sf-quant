"""
SLURM-based parallel backtesting. Distributes ``sf_quant.backtester`` runs across
compute nodes, one year per SLURM array task. Also available as the ``sf_backtester`` CLI.
"""

from sf_quant.backtester.slurm.config import BacktestConfig, BacktestDynamicConfig, SlurmConfig
from sf_quant.backtester.slurm.runner import BacktestRunner, BacktestDynamicRunner

__all__ = ["BacktestConfig", "BacktestDynamicConfig", "BacktestRunner", "BacktestDynamicRunner", "SlurmConfig"]
