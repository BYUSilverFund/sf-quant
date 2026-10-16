"""SLURM script generation and job submission."""

import importlib.resources
import os
import subprocess
import tempfile
from pathlib import Path

from sf_quant.backtester.slurm.config import BacktestConfig, BacktestDynamicConfig


def get_worker_script_path() -> Path:
    """Get the path to the bundled worker script."""
    ref = importlib.resources.files("sf_quant.backtester.slurm.scripts").joinpath("mvo_worker.py")
    # For a real file path (not in a zip), we can use the path directly
    # Otherwise, we'd need to extract it
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def generate_sbatch_script(config: BacktestConfig, years: list[int]) -> str:
    """Generate the SBATCH script content."""
    num_years = len(years)
    years_str = " ".join(str(y) for y in years)
    constraints_str = " ".join(config.constraints)
    worker_script = get_worker_script_path()

    script = f"""#!/bin/bash
#SBATCH --job-name={config.signal_name}_backtest
#SBATCH --output={config.logs_dir}/backtest_%A_%a.out
#SBATCH --error={config.logs_dir}/backtest_%A_%a.err
#SBATCH --array=0-{num_years - 1}%{config.slurm.max_concurrent_jobs}
#SBATCH --cpus-per-task={config.slurm.n_cpus}
#SBATCH --mem={config.slurm.mem}
#SBATCH --time={config.slurm.time}
#SBATCH --mail-user={config.byu_email}
#SBATCH --mail-type={config.slurm.mail_type}

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

DATA_PATH="{config.data_path}"
OUTPUT_DIR="{config.output_dir}"
GAMMA="{config.gamma}"
N_CPUS="{config.slurm.n_cpus}"
CONSTRAINTS="{constraints_str}"

# Years to process
years=({years_str})

num_years=${{#years[@]}}

if [ $SLURM_ARRAY_TASK_ID -ge $num_years ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (max $((num_years-1)))."
    exit 1
fi

year=${{years[$SLURM_ARRAY_TASK_ID]}}

source {config.project_root}/.venv/bin/activate
echo "Running year=$year"
srun python {worker_script} \\
    --data_path "$DATA_PATH" \\
    --gamma "$GAMMA" \\
    --year "$year" \\
    --output_dir "$OUTPUT_DIR" \\
    --n_cpus "$N_CPUS" \\
    --constraints $CONSTRAINTS
"""
    return script


def get_dynamic_worker_script_path() -> Path:
    """Get the path to the bundled dynamic worker script."""
    ref = importlib.resources.files("sf_quant.backtester.slurm.scripts").joinpath("mvo_dynamic_worker.py")
    with importlib.resources.as_file(ref) as path:
        return Path(path)


def generate_sbatch_script_dynamic(config: BacktestDynamicConfig, years: list[int]) -> str:
    """Generate the SBATCH script content for a dynamic backtest."""
    num_years = len(years)
    years_str = " ".join(str(y) for y in years)
    constraints_str = " ".join(config.constraints)
    worker_script = get_dynamic_worker_script_path()
    active_weights_str = str(config.active_weights).lower()

    script = f"""#!/bin/bash
#SBATCH --job-name={config.signal_name}_dynamic_backtest
#SBATCH --output={config.logs_dir}/backtest_%A_%a.out
#SBATCH --error={config.logs_dir}/backtest_%A_%a.err
#SBATCH --array=0-{num_years - 1}%{config.slurm.max_concurrent_jobs}
#SBATCH --cpus-per-task={config.slurm.n_cpus}
#SBATCH --mem={config.slurm.mem}
#SBATCH --time={config.slurm.time}
#SBATCH --mail-user={config.byu_email}
#SBATCH --mail-type={config.slurm.mail_type}

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

DATA_PATH="{config.data_path}"
OUTPUT_DIR="{config.output_dir}"
INITIAL_GAMMA="{config.initial_gamma}"
TARGET_ACTIVE_RISK="{config.target_active_risk}"
ACTIVE_WEIGHTS="{active_weights_str}"
N_CPUS="{config.slurm.n_cpus}"
CONSTRAINTS="{constraints_str}"

# Years to process
years=({years_str})

num_years=${{#years[@]}}

if [ $SLURM_ARRAY_TASK_ID -ge $num_years ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (max $((num_years-1)))."
    exit 1
fi

year=${{years[$SLURM_ARRAY_TASK_ID]}}

source {config.project_root}/.venv/bin/activate
echo "Running year=$year"
srun python {worker_script} \\
    --data_path "$DATA_PATH" \\
    --initial_gamma "$INITIAL_GAMMA" \\
    --target_active_risk "$TARGET_ACTIVE_RISK" \\
    --active_weights "$ACTIVE_WEIGHTS" \\
    --year "$year" \\
    --output_dir "$OUTPUT_DIR" \\
    --n_cpus "$N_CPUS" \\
    --constraints $CONSTRAINTS
"""
    return script


def submit_job(script_content: str, dry_run: bool = False) -> subprocess.CompletedProcess | None:
    """Submit a SLURM job from script content.

    Args:
        script_content: The SBATCH script content.
        dry_run: If True, print the script instead of submitting.

    Returns:
        The subprocess result if submitted, None if dry_run.
    """
    if dry_run:
        print("=" * 60)
        print("DRY RUN - Would submit the following script:")
        print("=" * 60)
        print(script_content)
        print("=" * 60)
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return result
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
