"""Configuration management for SLURM backtesting."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import yaml


@dataclass
class SlurmConfig:
    """SLURM job configuration."""

    n_cpus: int = 8
    mem: str = "32G"
    time: str = "06:00:00"
    mail_type: str = "BEGIN,END,FAIL"
    max_concurrent_jobs: int = 31


class _YamlConfig:
    """Shared behavior for backtest configs: derived paths and YAML (de)serialization.

    Subclasses are dataclasses that define ``run_id``, the value (gamma or
    target active risk) used to name the default output and log directories.
    """

    def __post_init__(self) -> None:
        """Set derived paths if not provided and build the SlurmConfig."""
        if self.output_dir is None:
            self.output_dir = f"{self.project_root}/weights/{self.signal_name}/{self.run_id}"
        if self.logs_dir is None:
            self.logs_dir = f"logs/{self.signal_name}/{self.run_id}"

        if isinstance(self.slurm, dict):
            self.slurm = SlurmConfig(**self.slurm)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


@dataclass
class BacktestConfig(_YamlConfig):
    """Configuration for a static (fixed gamma) backtest run."""

    signal_name: str
    data_path: str
    gamma: float

    project_root: str
    byu_email: str

    constraints: list[str]
    slurm: SlurmConfig

    # Default to ``{project_root}/weights/{signal_name}/{gamma}`` and ``logs/...``
    output_dir: str | None = None
    logs_dir: str | None = None

    @property
    def run_id(self) -> float:
        return self.gamma


@dataclass
class BacktestDynamicConfig(_YamlConfig):
    """Configuration for a dynamic (target active risk) backtest run."""

    signal_name: str
    data_path: str
    initial_gamma: float
    target_active_risk: float
    active_weights: bool

    project_root: str
    byu_email: str

    constraints: list[str]
    slurm: SlurmConfig

    # Default to ``{project_root}/weights/{signal_name}/{target_active_risk}`` and ``logs/...``
    output_dir: str | None = None
    logs_dir: str | None = None

    @property
    def run_id(self) -> float:
        return self.target_active_risk
