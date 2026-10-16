"""MVO worker script for SLURM jobs.

This script is executed by each SLURM array task to process a single year.
"""

import polars as pl
import sf_quant.backtester as sfb

from sf_quant.backtester.slurm.scripts.utils import make_parser, run_backtest_by_year


def main() -> None:
    """Main entry point for the worker script."""
    parser = make_parser("Run a fixed-gamma MVO backtest for a single year.")
    parser.add_argument("--gamma", type=float, required=True, help="Risk aversion parameter for MVO.")
    args = parser.parse_args()

    run_backtest_by_year(
        df=pl.scan_parquet(args.data_path),
        year=args.year,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus,
        constraints=args.constraints,
        backtest_fn=sfb.backtest_parallel,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
