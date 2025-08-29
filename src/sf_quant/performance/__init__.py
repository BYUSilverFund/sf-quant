"""
The ``performance`` module provides tools for evaluating and summarizing
portfolio and strategy performance. It exposes a clean public API for
calculating returns, risk metrics, and visualizing performance, while keeping
internal computation details hidden.
"""

from .returns import generate_returns_from_weights
from .tables import generate_summary_table
from .charts import generate_returns_chart

__all__ = [
    "generate_returns_from_weights",
    "generate_summary_table",
    "generate_returns_chart",
]
