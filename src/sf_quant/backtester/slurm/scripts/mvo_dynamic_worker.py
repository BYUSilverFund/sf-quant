"""MVO dynamic worker script for SLURM jobs.

This script is executed by each SLURM array task to process a single year.
"""

import polars as pl
import sf_quant.backtester as sfb

from sf_quant.backtester.slurm.scripts.utils import make_parser, run_backtest_by_year


def main() -> None:
    """Main entry point for the worker script."""
    parser = make_parser("Run a dynamic MVO backtest for a single year.")
    parser.add_argument("--initial_gamma", type=float, required=True, help="Initial risk aversion parameter.")
    parser.add_argument("--target_active_risk", type=float, required=True, help="Target active risk.")
    parser.add_argument(
        "--active_weights",
        type=lambda x: x.lower() == "true",
        required=True,
        help="Active weights flag (pass 'true' or 'false').",
    )
    args = parser.parse_args()

    run_backtest_by_year(
        df=pl.scan_parquet(args.data_path),
        year=args.year,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus,
        constraints=args.constraints,
        backtest_fn=sfb.dynamic_backtest_parallel,
        initial_gamma=args.initial_gamma,
        target_active_risk=args.target_active_risk,
        active_weights=args.active_weights,
    )


if __name__ == "__main__":
    main()
