"""Archived weight arithmetic. All portfolio backtests must use simulation.

Use sf_quant.simulation.simulate_portfolio and result.returns()/multi_returns().
The former public function names have been removed, with no callable aliases.
"""
import warnings

import polars as pl
from sf_quant.data.assets import load_assets
from sf_quant.data.benchmark import load_benchmark
from sf_quant.schema.portfolio_schema import PortfolioSchema
from sf_quant.schema.returns_schema import PortfolioRetSchema, MultiPortfolioRetSchema

LEGACY_RETURN_WARNING = 'NOTE this does not calculate correct returns, as it ignores portfolio drift. Do not use it. Use simulate_portfolio in sf_quant.simulation instead.'

# Retired public API (intentionally commented out):
# def generate_returns_from_weights(weights):
#     """NOTE this does not calculate correct returns, as it ignores portfolio drift. Do not use it. Use simulate_portfolio in sf_quant.simulation instead."""

# Retired public API (intentionally commented out):
# def generate_multi_returns_from_weights(weights):
#     """NOTE this does not calculate correct returns, as it ignores portfolio drift. Do not use it. Use simulate_portfolio in sf_quant.simulation instead."""

def legacy_generate_multi_returns_from_weights(weights: PortfolioSchema) -> MultiPortfolioRetSchema:
    """NOTE this does not calculate correct returns, as it ignores portfolio drift. Do not use it. Use simulate_portfolio in sf_quant.simulation instead.

    Archived same-date weighted-return arithmetic for reproducing legacy results
    only. This function does not simulate holdings, drift, execution timing, cash,
    or transaction costs. Never use it to evaluate a new portfolio backtest.
    """
    warnings.warn(
        LEGACY_RETURN_WARNING,
        RuntimeWarning,
        stacklevel=2,
    )
    start = weights["date"].min()
    end = weights["date"].max()

    columns = ["date", "barrid", "return"]

    returns = load_assets(start=start, end=end, in_universe=True, columns=columns)

    benchmark = load_benchmark(start=start, end=end)

    returns = returns.sort("barrid", "date")

    joined = (
        weights.join(returns, on=["date", "barrid"], how="left")
        .join(benchmark, on=["date", "barrid"], how="left", suffix="_bmk")
        .with_columns(
            pl.col("return").truediv(100),
        )
        .with_columns(pl.col("weight").sub("weight_bmk").alias("weight_act"))
        .rename({"weight": "total", "weight_bmk": "benchmark", "weight_act": "active"})
        .unpivot(
            index=["date", "barrid", "return"],
            variable_name="portfolio",
            value_name="weight",
        )
    )

    # Build aggregation expressions
    agg_exprs = []
    agg_exprs.append(pl.col("return").mul("weight").sum().alias("return"))

    result = (
        joined.group_by("date", "portfolio")
        .agg(agg_exprs)
        .sort("date", "portfolio")
    )
    return MultiPortfolioRetSchema.validate(result)


def legacy_generate_returns_from_weights(weights: PortfolioSchema) -> PortfolioRetSchema:
    """NOTE this does not calculate correct returns, as it ignores portfolio drift. Do not use it. Use simulate_portfolio in sf_quant.simulation instead.

    Archived same-date weighted-return arithmetic for reproducing legacy results
    only. This function does not simulate holdings, drift, execution timing, cash,
    or transaction costs. Never use it to evaluate a new portfolio backtest.
    """
    warnings.warn(
        LEGACY_RETURN_WARNING,
        RuntimeWarning,
        stacklevel=2,
    )
    start = weights["date"].min()
    end = weights["date"].max()

    columns = ["date", "barrid", "return"]

    returns = load_assets(start=start, end=end, in_universe=True, columns=columns)

    returns = returns.sort("barrid", "date")

    joined = (
        weights.join(returns, on=["date", "barrid"], how="left")
        .with_columns(
            pl.col("return").truediv(100),
        )
    )

    result = (
        joined.group_by("date")
        .agg([
            pl.col("return").mul("weight").sum().alias("return"),
        ])
        .sort("date")
    )
    return PortfolioRetSchema.validate(result)


def __getattr__(name):
    if name in ("generate_returns_from_weights", "generate_multi_returns_from_weights"):
        raise AttributeError(
            f"{name} has been retired because it ignores portfolio drift. "
            "Use simulate_portfolio in sf_quant.simulation instead."
        )
    raise AttributeError(name)
