import polars as pl
import os
import ray
import datetime as dt
from ray.experimental import tqdm_ray
from sf_quant.data.benchmark import load_benchmark
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.optimizers import dynamic_mve_optimizer
from sf_quant.optimizer.constraints import Constraint
from sf_quant.schema.portfolio_schema import PortfolioSchema


@ray.remote
def _construct_portfolio(
    date_: dt.date,
    data: pl.DataFrame,
    constraints: list[Constraint],
    gamma: float,
    target_active_risk: float | None = None,
    progress_bar: tqdm_ray.tqdm | None = None,
    active_weights: bool = False
):
    """
    Construct a single optimized portfolio for a given date.

    This is a Ray-remote helper function for parallel portfolio construction.
    It filters data to a specific date, constructs the factor model components,
    and solves a mean-variance optimization problem.

    Parameters
    ----------
    date_ : dt.date
        The date for which to construct the portfolio.
    data : pl.DataFrame
        Full dataset with columns 'date', 'barrid', 'alpha', optionally 'predicted_beta'.
    constraints : list[Constraint]
        Portfolio constraints to enforce.
    gamma : float
        Risk aversion parameter.
    target_active_risk : float, optional
        Target active risk for gamma calibration.
    progress_bar : tqdm_ray.tqdm, optional
        Ray-based progress bar for tracking completion.
    active_weights : bool
        Flag indicating how to treat output weights of optimizer. False (default)
        means that we subtract of benchmark weights before computing active risk.

    Returns
    -------
    pl.DataFrame
        Portfolio weights for the given date, with columns 'date', 'barrid', 'weight'.
    """
    subset = data.filter(pl.col("date").eq(date_)).sort("barrid")
    barrids = subset["barrid"].to_list()
    alphas = subset["alpha"].to_numpy()

    betas = (
        subset["predicted_beta"].to_numpy()
        if "predicted_beta" in subset.columns
        else None
    )

    benchmark_weights = (
        subset["benchmark_weight"].to_numpy()
        if "benchmark_weight" in subset.columns
        else None
    )

    if benchmark_weights is None and active_weights is False:
        raise ValueError("Benchmark weights must be provided if active_weights flag is False (default).")

    (
        factor_exposures,
        factor_covariance,
        specific_risk,
    ) = construct_factor_model_components(date_, barrids)

    portfolio = dynamic_mve_optimizer(
        ids=barrids,
        alphas=alphas,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        constraints=constraints,
        initial_gamma=gamma,
        betas=betas,
        target_active_risk=target_active_risk,
        benchmark_weights=benchmark_weights,
        active_weights=active_weights
    )

    portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
        "date", "barrid", "weight", 'gamma', 'active_risk'
    )

    if progress_bar is not None:
        progress_bar.update.remote(1)

    return portfolio


def dynamic_backtest_parallel(
    data: pl.DataFrame,
    constraints: list[Constraint],
    initial_gamma: float = 100,
    target_active_risk: float = 0.05,
    n_cpus: int | None = None,
    active_weights: bool = False
) -> pl.DataFrame:
    """
    Run a parallelized backtest of portfolio optimization using Ray.

    This function distributes portfolio construction tasks across multiple CPUs
    using Ray, solving mean–variance optimization problems for each date in parallel.
    A Ray-based progress bar tracks computation progress.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataset containing at least the following columns:

        - ``date`` : datetime-like, the date of each observation.
        - ``barrid`` : str, unique identifier for each asset.
        - ``alpha`` : float, expected return (alpha) for each asset.
        - ``predicted_beta`` : float, optional, factor exposures for constraints.

    constraints : list[Constraint]
        List of portfolio constraints to enforce during optimization.
    initial_gamma : float, optional
        Risk aversion parameter. Higher values penalize portfolio variance
        more strongly. Used as the starting gamma for calibration if
        ``target_active_risk`` is specified. Default is 100.
    target_active_risk : float, optional
        If specified, automatically calibrate gamma for each date to achieve this
        target annualized active risk (e.g., 0.05 for 5%). Requires benchmark data
        to be available via :func:`~sf_quant.data.benchmark.load_benchmark`.
    n_cpus : int, optional
        Number of CPUs to allocate to Ray. If ``None``, defaults to
        ``os.cpu_count()`` but is capped at the number of unique dates.
    active_weights : bool
        Flag indicating how to treat output weights of optimizer. False (default)
        means that we subtract of benchmark weights before computing active risk.

    Returns
    -------
    pl.DataFrame
        A PortfolioSchema-validated Polars DataFrame containing optimized
        portfolio weights across all backtest dates, with columns:

        - ``date`` : datetime, portfolio date.
        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.
        - ``gamma`` : float, calibrated risk aversion parameter for each date.
        - ``active_risk`` : float, achieved annualized active risk for each date.

    Notes
    -----
    - Benchmark data is automatically loaded when ``target_active_risk`` is specified,
      via :func:`~sf_quant.data.benchmark.load_benchmark`.
    - Ray is initialized with ``ignore_reinit_error=True``, allowing safe
      re-invocation within the same process.

    See Also
    --------
    backtest_parallel : Sequential version without active risk calibration.
    dynamic_mve_optimizer : Underlying optimizer with active risk calibration.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import sf_quant.backtester as sfb
    >>> import sf_quant.optimizer as sfo
    >>> import polars as pl
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 1, 10)
    >>> benchmark_weights = sfd.load_benchmark(start, end).rename({'weight': 'benchmark_weight'})
    >>> data = (
    ...     sfd.load_assets(
    ...         start=start,
    ...         end=end,
    ...         in_universe=True,
    ...         columns=['date', 'barrid']
    ...     )
    ...     .with_columns(
    ...         pl.lit(0).alias('alpha')
    ...     )
    ...     .join(
    ...         other=benchmark_weights,
    ...         on=['date', 'barrid'],
    ...         how='left'
    ...     )
    ... )
    >>> constraints = [sfo.FullInvestment(), sfo.LongOnly()]
    >>> weights = sfb.dynamic_backtest_parallel(
    ...     data=data,
    ...     constraints=constraints,
    ...     initial_gamma=100,
    ...     target_active_risk=0.05,
    ...     active_weights=False
    ... )
    shape: (5, 5)
    ┌────────────┬─────────┬───────────┬───────┬──────────────┐
    │ date       ┆ barrid  ┆ weight    ┆ gamma ┆ active_risk  │
    │ ---        ┆ ---     ┆ ---       ┆ ---   ┆ ---          │
    │ date       ┆ str     ┆ f64       ┆ f64   ┆ f64          │
    ╞════════════╪═════════╪═══════════╪═══════╪══════════════╡
    │ 2024-01-02 ┆ USA06Z1 ┆ -0.000639 ┆ 100.0 ┆ 0.047        │
    │ 2024-01-02 ┆ USA0771 ┆ -0.000083 ┆ 100.0 ┆ 0.047        │
    │ 2024-01-02 ┆ USA0C11 ┆ -0.003044 ┆ 100.0 ┆ 0.047        │
    │ 2024-01-02 ┆ USA0SY1 ┆ -0.002177 ┆ 100.0 ┆ 0.047        │
    │ 2024-01-02 ┆ USA11I1 ┆ 0.001475  ┆ 100.0 ┆ 0.047        │
    └────────────┴─────────┴───────────┴───────┴──────────────┘
    """
    # Get dates
    dates = data["date"].unique().sort().to_list()

    # Set up ray
    n_cpus = n_cpus or os.cpu_count()
    n_cpus = min(len(dates), n_cpus)
    # Empty runtime_env = no environment management. Workers use parent's Python directly.
    # This prevents Ray from trying to serialize code or rebuild packages on air-gapped clusters.
    ray.init(ignore_reinit_error=True, num_cpus=n_cpus, runtime_env={})

    # Set up ray progress bar
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    progress_bar = remote_tqdm.remote(
        total=len(dates), desc=f"Computing portfolios with {n_cpus} cpus"
    )

    # Dispatch parallel tasks
    portfolio_futures = [
        _construct_portfolio.remote(
            date_=date_,
            data=data,
            constraints=constraints,
            gamma=initial_gamma,
            target_active_risk=target_active_risk,
            progress_bar=progress_bar,
            active_weights=active_weights
        )
        for date_ in dates
    ]

    portfolio_list = ray.get(portfolio_futures)

    progress_bar.close.remote()
    ray.shutdown()

    return pl.concat(portfolio_list)
