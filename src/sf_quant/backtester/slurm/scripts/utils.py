"""Shared code for the mvo and mvo_dynamic worker scripts."""

import argparse
import datetime as dt
import inspect
from typing import Callable

import polars as pl

from sf_quant.optimizer import constraints as _constraints

# Registry of available constraints, built from sf_quant.optimizer.constraints
CONSTRAINT_REGISTRY: dict[str, type] = {
    name: cls
    for name, cls in vars(_constraints).items()
    if inspect.isclass(cls)
    and cls.__module__ == _constraints.__name__
    and name != "Constraint"
}


def get_constraints(constraint_names: list[str]) -> list:
    """Convert constraint names to constraint objects.

    Args:
        constraint_names: List of constraint class names.

    Returns:
        List of instantiated constraint objects.

    Raises:
        KeyError: If a constraint name is not in the registry.
    """
    constraints = []
    for name in constraint_names:
        if name not in CONSTRAINT_REGISTRY:
            available = ", ".join(CONSTRAINT_REGISTRY.keys())
            raise KeyError(f"Unknown constraint '{name}'. Available: {available}")
        constraints.append(CONSTRAINT_REGISTRY[name]())
    return constraints


def run_backtest_by_year(
    df: pl.LazyFrame,
    year: int,
    output_dir: str,
    n_cpus: int,
    constraints: list[str],
    backtest_fn: Callable[..., pl.DataFrame],
    **backtest_kwargs,
) -> None:
    """Run a backtest for a single year and write the weights to parquet.

    Args:
        df: LazyFrame containing the data.
        year: Year to process.
        output_dir: Directory to write ``{year}.parquet``.
        n_cpus: Number of CPUs for parallel execution.
        constraints: List of constraint names.
        backtest_fn: ``sf_quant.backtester`` function to run (e.g. ``backtest_parallel``).
        **backtest_kwargs: Extra arguments forwarded to ``backtest_fn``.
    """
    filtered = df.filter(
        pl.col("date").is_between(dt.date(year, 1, 1), dt.date(year, 12, 31))
    ).collect()

    print(f"Processing year {year}: {len(filtered)} rows")

    weights = backtest_fn(
        data=filtered,
        constraints=get_constraints(constraints),
        n_cpus=n_cpus,
        **backtest_kwargs,
    )

    output_path = f"{output_dir}/{year}.parquet"
    weights.write_parquet(output_path)
    print(f"Wrote weights to {output_path}")


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create an argument parser with the arguments shared by all workers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_path", required=True, help="Path to parquet file containing the data.")
    parser.add_argument("--year", type=int, required=True, help="Year to process.")
    parser.add_argument("--output_dir", required=True, help="Directory to write output parquet file.")
    parser.add_argument("--n_cpus", type=int, required=True, help="Number of CPUs for parallel execution.")
    parser.add_argument(
        "--constraints",
        nargs="+",
        required=True,
        help="List of constraint names (e.g., ZeroBeta ZeroInvestment).",
    )
    return parser
