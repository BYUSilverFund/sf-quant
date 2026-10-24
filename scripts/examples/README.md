# Dynamic backtest example

This tutorial demonstrates the dynamic backtester with one calendar year of
Barra momentum data. Part A prepares the input data and submits the yearly
SLURM job. Part B reads its weights and produces daily active-return results.
The tutorial uses its own signal name, so its weights, logs, and results do not
overwrite the existing Barra momentum backtest.

## 1. Clone and install the local checkouts

Place both repositories beside each other in a workspace on the supercomputer that is available at the same path on the login node and SLURM compute nodes:

```text
shared-workspace/
├── sf-quant/
└── sf-backtester/
```

From `sf-quant`, create the environment where the SLURM worker expects it, then
install both local repositories in editable mode:

```bash
cd /path/to/shared-workspace/sf-quant
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -e . -e ../sf-backtester
```

The editable installs point Python at the checked-out source, so imports use
the local code and reflect later edits to either clone. The backtester activates
`<project_root>/.venv/bin/activate` in its generated SLURM script; the example
sets `project_root` to the `sf-quant` checkout. Keep both repositories and this
environment on the cluster's shared filesystem so compute nodes can resolve
the environment and editable source paths.

## 2. Configure data access and the SLURM account

The data loaders require `ROOT` and `DATABASE`. Use "research" as the database. Export them in the shell before running Part A, along with your BYU email for SLURM notifications:

```bash
export ROOT="/path/to/data/root"
export DATABASE="your_database"
export BYU_EMAIL="your_netid@byu.edu"
```

See the [`sf-quant` environment configuration guide](../../README.md#environment-configuration)
for details about these variables and the optional `.env` setup.

The example calls `load_dotenv()` as well, so a project `.env` file can supply
`ROOT` and `DATABASE`. Do not commit credentials. The SLURM job inherits the
submission environment; make sure these variables are available when Part A
submits the job. Part A reads the notification address from `BYU_EMAIL`.

## 3. Run Part A

From the `sf-quant` checkout with the environment activated:

```bash
python scripts/examples/run_dynamic_backtest_a.py
```

Part A loads the 2024 `barra_momentum` alpha, predicted beta, and benchmark
weights, then submits a dynamic backtest with active weights and the explicit
`ZeroInvestment`, `LongOnly`, and `ZeroBeta` constraints. The target active risk
is an editable `0.05` value in both scripts. The expected yearly weight file is
under `scripts/weights/tutorial_barra_momentum/0.05/`, and logs are under
`scripts/logs/tutorial_barra_momentum/0.05/`.

Submission returns before the array finishes. Check the job with your cluster's
SLURM tools and wait until it completes successfully; confirm that the 2024
parquet weight file exists before continuing.

## 4. Run Part B

After Part A finishes, run:

```bash
python scripts/examples/run_dynamic_backtest_b.py
```

Part B calculates daily active returns from the saved active weights and
next-day asset returns. It writes a parquet file, a cumulative-return chart,
and a summary CSV under `scripts/results/tutorial_barra_momentum/`.

## 5. Adapt the example

- To test another alpha, change `SOURCE_SIGNAL` in Part A and use a distinct
  `SIGNAL_NAME` in both scripts so outputs remain isolated.
- To use another risk target, change `TARGET_ACTIVE_RISK` in both scripts; it
  must match the risk directory containing the Part A weights.
- To use a different date range, update `START` and `END` in both scripts.
- For monthly Part C results, invoke the shared helper after Part A completes
  and pass the same signal name and risk value. Save this as
  `scripts/run_tutorial_c.py` and run it from the `sf-quant` checkout:

  ```python
  from monthly_portfolio_c_utils import run_monthly_part_c

  run_monthly_part_c(
      "tutorial_barra_momentum",
      "Tutorial Barra Momentum",
      target_active_risk=0.05,
  )
  ```

  ```bash
  python scripts/run_tutorial_c.py
  ```

- For weekly Part D results, use the corresponding helper:
  save this as `scripts/run_tutorial_d.py` and run it from the `sf-quant`
  checkout:

  ```python
  from monthly_portfolio_c_utils import run_weekly_part_d

  run_weekly_part_d(
      "tutorial_barra_momentum",
      "Tutorial Barra Momentum",
      target_active_risk=0.05,
  )
  ```

  ```bash
  python scripts/run_tutorial_d.py
  ```

Part C and Part D post-process the daily Part A weights with monthly or weekly
hold schedules; they do not submit another optimizer backtest. Their shared
helper uses its own `START` and `END` constants for return and benchmark data,
so adjust those constants in `scripts/monthly_portfolio_c_utils.py` when you
want the periodic analysis limited to a shorter tutorial date range.
