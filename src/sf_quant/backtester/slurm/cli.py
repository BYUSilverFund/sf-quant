"""Command-line interface for SLURM backtesting."""

import click

from sf_quant.backtester.slurm.runner import BacktestDynamicRunner, BacktestRunner


@click.group()
def main() -> None:
    """SLURM backtesting: Multi-node parallelized backtesting tool."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SBATCH script without submitting.",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Override the data path from config.",
)
@click.option(
    "--gamma",
    type=float,
    help="Override gamma from config.",
)
def run(
    config_path: str,
    dry_run: bool,
    data_path: str | None,
    gamma: float | None,
) -> None:
    """Run a parallel backtest on SLURM from a YAML config file.

    CONFIG_PATH is the path to a YAML configuration file.

    Example:
        sf_backtester run config.yml
        sf_backtester run config.yml --dry-run
        sf_backtester run config.yml --gamma 0.5
    """
    runner = BacktestRunner.from_yaml(config_path)

    # Apply overrides
    if data_path is not None:
        runner.config.data_path = data_path
    if gamma is not None:
        runner.config.gamma = gamma
        # Update output paths that depend on gamma
        runner.config.output_dir = (
            f"{runner.config.project_root}/weights/{runner.config.signal_name}/{gamma}"
        )
        runner.config.logs_dir = f"logs/{runner.config.signal_name}/{gamma}"

    runner.submit(dry_run=dry_run)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the SBATCH script without submitting.",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    help="Override the data path from config.",
)
@click.option(
    "--initial_gamma",
    type=float,
    help="Override initial_gamma from config.",
)
@click.option(
    "--target_active_risk",
    type=float,
    help="Override target_active_risk from config.",
)
@click.option(
    "--active_weights",
    type=bool,
    help="Override active_weights flag from config.",
)
def run_dynamic(
    config_path: str,
    dry_run: bool,
    data_path: str | None,
    initial_gamma: float | None,
    target_active_risk: float | None,
    active_weights: bool | None,
) -> None:
    """Run a dynamic parallel backtest on SLURM from a YAML config file.

    CONFIG_PATH is the path to a YAML configuration file.

    Example:
        sf_backtester run-dynamic config.yml
        sf_backtester run-dynamic config.yml --dry-run
        sf_backtester run-dynamic config.yml --initial_gamma 0.5
    """
    runner = BacktestDynamicRunner.from_yaml(config_path)

    # Apply overrides
    if data_path is not None:
        runner.config.data_path = data_path

    if initial_gamma is not None:
        runner.config.initial_gamma = initial_gamma

    if target_active_risk is not None:
        runner.config.target_active_risk = target_active_risk
        # Update output paths that depend on target_active_risk
        runner.config.output_dir = (
            f"{runner.config.project_root}/weights/{runner.config.signal_name}/{target_active_risk}"
        )
        runner.config.logs_dir = f"logs/{runner.config.signal_name}/{target_active_risk}"

    if active_weights is not None:
        runner.config.active_weights = active_weights

    runner.submit(dry_run=dry_run)
