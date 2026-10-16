"""MVO dynamic worker script for SLURM jobs.

This script is executed by each SLURM array task to process a single year.
"""

import argparse
import datetime as dt

import polars as pl
import sf_quant.backtester as sfb
from sf_quant.backtester.slurm.scripts.utils import get_constraints


def run_backtest_by_year(
    df: pl.LazyFrame,
    initial_gamma: float,
    year: int,
    output_dir: str,
    n_cpus: int,
    constraints: list[str],
    active_weights: bool,
    target_active_risk: float,
) -> None:
    """Run MVO backtest for a single year.

    Args:
        df: LazyFrame containing the data.
        initial_gamma: Risk aversion parameter.
        year: Year to process.
        output_dir: Directory to write output.
        n_cpus: Number of CPUs for parallel execution.
        constraints: List of constraint names.
        active_weights: Active weights flag.
        target_active_risk: Target active risk.
    """
    year_start = dt.date(year, 1, 1)
    year_end = dt.date(year, 12, 31)

    filtered = (
        df.filter(pl.col("date").is_between(year_start, year_end))
        .collect()
    )

    print(f"Processing year {year}: {len(filtered)} rows")

    constraint_objects = get_constraints(constraints)

    weights = sfb.dynamic_backtest_parallel(
        data=filtered,
        constraints=constraint_objects,
        initial_gamma=initial_gamma,
        target_active_risk=target_active_risk,
        active_weights=active_weights,
        n_cpus=n_cpus,
    )

    output_path = f"{output_dir}/{year}.parquet"
    weights.write_parquet(output_path)
    print(f"Wrote weights to {output_path}")


def main() -> None:
    """Main entry point for the worker script."""
    parser = argparse.ArgumentParser(
        description="Run MVO backtest for a single year."
    )

    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to parquet file containing the data.",
    )
    parser.add_argument(
        "--initial_gamma",
        type=float,
        required=True,
        help="Risk aversion parameter for MVO.",
    )
    parser.add_argument(
        "--target_active_risk",
        type=float,
        required=True,
        help="Target active risk.",
    )
    parser.add_argument(
        "--active_weights",
        type=lambda x: x.lower() == "true",
        required=True,
        help="Active weights flag (pass 'true' or 'false').",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output parquet file.",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        required=True,
        help="Number of CPUs for parallel execution.",
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        required=True,
        help="List of constraint names (e.g., ZeroBeta ZeroInvestment).",
    )

    args = parser.parse_args()

    df = pl.scan_parquet(args.data_path)

    run_backtest_by_year(
        df=df,
        initial_gamma=args.initial_gamma,
        year=args.year,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus,
        constraints=args.constraints,
        active_weights=args.active_weights,
        target_active_risk=args.target_active_risk
    )


if __name__ == "__main__":
    main()
