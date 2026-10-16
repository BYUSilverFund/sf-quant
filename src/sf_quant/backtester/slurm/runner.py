"""Main backtest runner."""

import os
from pathlib import Path

import polars as pl

from sf_quant.backtester.slurm.config import BacktestConfig, BacktestDynamicConfig
from sf_quant.backtester.slurm.slurm import generate_sbatch_script, submit_job


class BacktestRunner:
    """Orchestrates parallel backtesting via SLURM."""

    config_cls = BacktestConfig
    kind = ""

    def __init__(self, config: BacktestConfig) -> None:
        """Initialize the runner with configuration.

        Args:
            config: Backtest configuration.
        """
        self.config = config
        self._data: pl.DataFrame | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path):
        """Create a runner from a YAML config file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Configured runner instance.
        """
        return cls(cls.config_cls.from_yaml(config_path))

    def load_data(self, data: pl.DataFrame | None = None) -> pl.DataFrame:
        """Load data from config path or use provided DataFrame.

        Args:
            data: Optional DataFrame to use instead of loading from disk.

        Returns:
            The loaded or provided DataFrame.
        """
        if data is not None:
            self._data = data
        elif self._data is None:
            self._data = pl.read_parquet(self.config.data_path)
        return self._data

    def get_years(self, data: pl.DataFrame) -> list[int]:
        """Extract unique years from the data.

        Args:
            data: DataFrame with a 'date' column.

        Returns:
            Sorted list of unique years.
        """
        years = (
            data.select(pl.col("date").dt.year().alias("year"))
            .unique()
            .sort("year")
            .to_series()
            .to_list()
        )
        return years

    def prepare(self, data: pl.DataFrame | None = None) -> None:
        """Prepare directories and save data for the backtest.

        Args:
            data: Optional DataFrame to use instead of loading from config path.
        """
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)

        # Ensure temp directory exists
        temp_dir = Path(self.config.project_root) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # If data provided, save to the configured data_path
        if data is not None:
            data.write_parquet(self.config.data_path)
            self._data = data

    def describe(self) -> list[str]:
        """Config lines printed before submitting."""
        return [
            f"Gamma: {self.config.gamma}",
        ]

    def submit(
        self,
        data: pl.DataFrame | None = None,
        dry_run: bool = False,
    ) -> None:
        """Submit the backtest job to SLURM.

        Args:
            data: Optional DataFrame to backtest. If not provided, loads from config.data_path.
            dry_run: If True, print the script instead of submitting.
        """
        # Prepare directories and optionally save data
        self.prepare(data)

        # Load data to determine years
        df = self.load_data(data)
        years = self.get_years(df)

        print(f"Preparing {self.kind}backtest for {len(years)} years: {years[0]}-{years[-1]}")
        print(f"Signal: {self.config.signal_name}")
        for line in self.describe():
            print(line)
        print(f"Constraints: {self.config.constraints}")
        print(f"Output directory: {self.config.output_dir}")

        # Generate and submit
        script = generate_sbatch_script(self.config, years)
        result = submit_job(script, dry_run=dry_run)

        if result is not None:
            print("Job submitted successfully!")
            print(f"sbatch output: {result.stdout}")
            if result.stderr:
                print(f"sbatch stderr: {result.stderr}")


class BacktestDynamicRunner(BacktestRunner):
    """Orchestrates parallel dynamic backtesting via SLURM."""

    config_cls = BacktestDynamicConfig
    kind = "dynamic "

    def describe(self) -> list[str]:
        return [
            f"Initial gamma: {self.config.initial_gamma}",
            f"Target active risk: {self.config.target_active_risk}",
            f"Active weights: {self.config.active_weights}",
        ]
