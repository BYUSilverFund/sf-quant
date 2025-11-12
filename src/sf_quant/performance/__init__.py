"""
The ``performance`` module provides tools for evaluating and summarizing
portfolio and strategy performance. It exposes a clean public API for
calculating returns, risk metrics, and visualizing performance, while keeping
internal computation details hidden.
"""

from .returns import generate_returns_from_weights, generate_multi_returns_from_weights
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
)

__all__ = [
    # Returns
    "generate_returns_from_weights",
    "generate_multi_returns_from_weights",
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
]
