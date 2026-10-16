"""Configuration management for SLURM backtesting."""

from dataclasses import dataclass, field
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


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    # Required fields
    signal_name: str
    data_path: str
    gamma: float

    # Environment/paths
    project_root: str
    byu_email: str

    # Optional with defaults
    constraints: list[str]
    slurm: SlurmConfig

    # Output configuration
    output_dir: str | None = None
    logs_dir: str | None = None

    def __post_init__(self) -> None:
        """Set derived paths if not provided."""
        if self.output_dir is None:
            self.output_dir = f"{self.project_root}/weights/{self.signal_name}/{self.gamma}"
        if self.logs_dir is None:
            self.logs_dir = f"logs/{self.signal_name}/{self.gamma}"

        # Convert slurm dict to SlurmConfig if needed
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
        data = {
            "signal_name": self.signal_name,
            "data_path": self.data_path,
            "gamma": self.gamma,
            "project_root": self.project_root,
            "byu_email": self.byu_email,
            "constraints": self.constraints,
            "slurm": {
                "n_cpus": self.slurm.n_cpus,
                "mem": self.slurm.mem,
                "time": self.slurm.time,
                "mail_type": self.slurm.mail_type,
                "max_concurrent_jobs": self.slurm.max_concurrent_jobs,
            },
            "output_dir": self.output_dir,
            "logs_dir": self.logs_dir,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@dataclass
class BacktestDynamicConfig:
    """Configuration for a backtest run."""

    # Required fields
    signal_name: str
    data_path: str
    initial_gamma: float
    target_active_risk: float
    active_weights: bool

    # Environment/paths
    project_root: str
    byu_email: str

    # Optional with defaults
    constraints: list[str]
    slurm: SlurmConfig

    # Output configuration
    output_dir: str | None = None
    logs_dir: str | None = None

    def __post_init__(self) -> None:
        """Set derived paths if not provided."""
        if self.output_dir is None:
            self.output_dir = f"{self.project_root}/weights/{self.signal_name}/{self.target_active_risk}"
        if self.logs_dir is None:
            self.logs_dir = f"logs/{self.signal_name}/{self.target_active_risk}"

        # Convert slurm dict to SlurmConfig if needed
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
        data = {
            "signal_name": self.signal_name,
            "data_path": self.data_path,
            "initial_gamma": self.initial_gamma,
            "target_active_risk": self.target_active_risk,
            "active_weights": self.active_weights,
            "project_root": self.project_root,
            "byu_email": self.byu_email,
            "constraints": self.constraints,
            "slurm": {
                "n_cpus": self.slurm.n_cpus,
                "mem": self.slurm.mem,
                "time": self.slurm.time,
                "mail_type": self.slurm.mail_type,
                "max_concurrent_jobs": self.slurm.max_concurrent_jobs,
            },
            "output_dir": self.output_dir,
            "logs_dir": self.logs_dir,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
