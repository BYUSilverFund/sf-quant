"""
The ``research`` module provides tools for quantitative research and analysis.
It includes functions for generating quantile portfolios, scaling returns by
volatility and beta, running Fama-French regressions, and other research utilities.
"""

from .quantile_portfolios import (
    generate_quantile_ports,
    vol_scale_ports,
    beta_scale_ports,
)
from .ff_regression import run_ff_regression
from .signal_analysis import signal_stats, signal_distribution

__all__ = [
    "generate_quantile_ports",
    "vol_scale_ports",
    "beta_scale_ports",
    "run_ff_regression",
    "signal_stats",
    "signal_distribution",
]
