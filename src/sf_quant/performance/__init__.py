"""
The ``performance`` module provides tools for evaluating and summarizing
portfolio and strategy performance. It exposes a clean public API for
calculating returns, risk metrics, and visualizing performance, while keeping
internal computation details hidden.
"""

from .returns import legacy_generate_returns_from_weights, legacy_generate_multi_returns_from_weights
from sf_quant.simulation import SimulationConfig, SimulationResult, simulate_portfolio
from .leverage import generate_leverage_from_weights
from .drawdown import generate_drawdown_from_returns
from .tables import (
    generate_returns_summary_table,
    generate_multi_returns_summary_table,
    generate_leverage_summary_table,
    generate_drawdown_summary_table,
)
from .charts import (
    generate_returns_chart,
    generate_multi_returns_chart,
    generate_leverage_chart,
    generate_drawdown_chart,
    generate_ic_chart,
)
from .ics import (
    generate_alpha_ics,
)
from .turnover import (
    get_turnover_stats,
    plot_turnover,
    get_trade_turnover_stats,
)

__all__ = [
    "SimulationConfig", "SimulationResult", "simulate_portfolio", "get_trade_turnover_stats",
    # Returns
    "legacy_generate_returns_from_weights",
    "legacy_generate_multi_returns_from_weights",
    # Leverage
    "generate_leverage_from_weights",
    # Drawdown
    "generate_drawdown_from_returns",
    # Summary tables
    "generate_returns_summary_table",
    "generate_multi_returns_summary_table",
    "generate_leverage_summary_table",
    "generate_drawdown_summary_table",
    # Charts
    "generate_returns_chart",
    "generate_multi_returns_chart",
    "generate_leverage_chart",
    "generate_drawdown_chart",
    # Alpha
    "generate_alpha_ics",
    "generate_ic_chart",
    # Turnover
    "get_turnover_stats",
    "plot_turnover",
]


def __getattr__(name):
    if name in ("generate_returns_from_weights", "generate_multi_returns_from_weights"):
        raise AttributeError(
            f"{name} has been retired because it ignores portfolio drift. "
            "Use simulate_portfolio in sf_quant.simulation instead."
        )
    raise AttributeError(name)
