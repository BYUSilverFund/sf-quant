# Silver Fund Quant

A quantitative finance library for the BYU Silver Fund quant team. Provides tools for data loading, portfolio optimization, backtesting, performance analysis, and signal research — all built on Polars DataFrames.

## Installation

```bash
pip install sf-quant
```

## Environment Configuration

`sf-quant` requires two environment variables to locate data: `ROOT` and `DATABASE`.

### Option 1: `.env` file or environment variables

Create a `.env` file in your project root (automatically loaded via `python-dotenv`):

```
ROOT=/path/to/root
DATABASE=your_database
```

Or export them in your shell:

```bash
export ROOT=/path/to/root
export DATABASE=your_database
```

### Option 2: Programmatic configuration

Set values at runtime using `sfd.env()`:

```python
import sf_quant.data as sfd

sfd.env(root="/path/to/root", database="your_database")
```

This will override any values set via environment variables.

## Quick Start

A complete workflow: load asset data, build a momentum alpha signal, construct a covariance matrix, and optimize a portfolio.

```python
import sf_quant.data as sfd
import sf_quant.optimizer as sfo
import datetime as dt
import polars as pl

# 1. Load asset data
df = (
    sfd.load_assets(
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 12, 31),
        columns=["date", "barrid", "price", "return", "specific_risk", "predicted_beta"],
        in_universe=True,
    )
    .with_columns(pl.col("return", "specific_risk").truediv(100))
    .sort("barrid", "date")
)

# 2. Build a momentum alpha signal
df_clean = (
    df
    .with_columns(
        pl.col("return").log1p().rolling_sum(window_size=230).shift(22).over("barrid").alias("momentum"),
    )
    .with_columns(
        pl.col("momentum").sub(pl.col("momentum").mean()).truediv(pl.col("momentum").std()).over("date").alias("score")
    )
    .with_columns(
        pl.lit(0.05).mul(pl.col("score")).mul(pl.col("specific_risk")).alias("alpha")
    )
    .drop_nulls()
    .sort("barrid")
)

# 3. Construct the covariance matrix
date_ = df_clean["date"].last()
barrids = df_clean["barrid"].to_list()

covariance_matrix = (
    sfd.construct_covariance_matrix(date_=date_, barrids=barrids)
    .drop("barrid").to_numpy()
)

# 4. Optimize the portfolio
weights = sfo.mve_optimizer(
    ids=barrids,
    alphas=df_clean["alpha"].to_list(),
    betas=df_clean["predicted_beta"].to_list(),
    covariance_matrix=covariance_matrix,
    constraints=[sfo.ZeroBeta()],
    gamma=100,
)
```

## Modules

### `sf_quant.data`

Load financial data as Polars DataFrames. Use `get_*_columns()` to inspect available columns for any dataset.

| Function | Description |
|---|---|
| `load_assets()` | Asset prices, returns, risk, and metadata for a date range |
| `load_assets_by_date()` | Asset data for a single date |
| `load_crsp_daily()` / `load_crsp_v2_daily()` | CRSP daily stock data |
| `load_crsp_monthly()` / `load_crsp_v2_monthly()` | CRSP monthly stock data |
| `load_exposures()` / `load_exposures_by_date()` | Barra factor exposures (77+ factors) |
| `load_covariances_by_date()` | Factor covariance matrix for a date |
| `construct_covariance_matrix()` | Build an asset-level covariance matrix from the factor model |
| `load_benchmark()` / `load_benchmark_returns()` | Benchmark weights and returns |
| `load_factors()` / `get_factor_names()` | Barra factor returns and metadata |
| `load_fama_french()` | Fama-French 5-factor data |

### `sf_quant.optimizer`

Mean-variance efficient (MVE) portfolio optimizer built on CVXPY.

```python
import sf_quant.optimizer as sfo

weights = sfo.mve_optimizer(
    ids=barrids,
    alphas=alphas,
    covariance_matrix=covariance_matrix,
    constraints=[sfo.FullInvestment(), sfo.LongOnly()],
    gamma=2,
)
```

**Available constraints:**

| Constraint | Effect |
|---|---|
| `FullInvestment()` | Weights sum to 1 |
| `ZeroInvestment()` | Weights sum to 0 |
| `LongOnly()` | No short positions |
| `NoBuyingOnMargin()` | Weights sum to at most 1 |
| `UnitBeta()` | Portfolio beta equals 1 |
| `ZeroBeta()` | Portfolio beta equals 0 |

### `sf_quant.backtester`

Run walk-forward backtests that optimize at each rebalance date.

```python
import sf_quant.backtester as sfb

# Sequential (single core)
portfolio = sfb.backtest_sequential(data=df, constraints=constraints, gamma=2)

# Parallel (multi-core via Ray)
portfolio = sfb.backtest_parallel(data=df, constraints=constraints, gamma=2, n_cpus=4)
```

### `sf_quant.performance`

Compute and visualize portfolio performance metrics.

| Function | Description |
|---|---|
| `generate_returns_from_weights()` | Portfolio returns from weight history |
| `generate_multi_returns_from_weights()` | Total, benchmark, and active returns |
| `generate_leverage_from_weights()` | Sum of absolute weights over time |
| `generate_drawdown_from_returns()` | Drawdown from peak |
| `generate_returns_summary_table()` | Mean return, volatility, Sharpe, total return |
| `generate_alpha_ics()` | Information coefficients between signals and realized returns |
| `get_turnover_stats()` | Rolling turnover statistics |
| `generate_returns_chart()` | Plot cumulative returns |
| `generate_multi_returns_chart()` | Plot total/benchmark/active returns |
| `generate_leverage_chart()` | Plot leverage over time |
| `generate_drawdown_chart()` | Plot drawdowns |
| `generate_ic_chart()` | Plot information coefficients |
| `plot_turnover()` | Plot portfolio turnover |

### `sf_quant.research`

Tools for signal analysis and factor research.

| Function | Description |
|---|---|
| `generate_quantile_ports()` | Sort assets into quantile portfolios with long-short spread |
| `vol_scale_ports()` | Scale portfolio returns to target volatility |
| `beta_scale_ports()` | Scale portfolio returns to target beta |
| `run_ff_regression()` | Fama-French 5-factor regression |
| `run_quantile_ff_regression()` | FF regressions across quantile portfolios |
| `get_signal_stats()` | Summary statistics for a signal |
| `get_signal_distribution()` | Plot signal distribution |

## Documentation Development

To run a local server of the sphinx documentation run

```bash
uv run sphinx-autobuild docs docs/_build/html
```

## Release Process
1. Create PR
2. Merge PR(s)
3. Increment version in pyproject.toml
4. git tag v*.*.*
5. git push origin main --tags
6. Create a release and publish release notes (github)
7. uv build
8. uv publish
