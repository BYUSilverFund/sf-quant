import polars as pl
import os
import ray
import datetime as dt
from ray.experimental import tqdm_ray
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.optimizers import dynamic_mve_optimizer
from sf_quant.optimizer.constraints import Constraint


@ray.remote
def _construct_portfolio(
    date_: dt.date,
    data: pl.DataFrame,
    constraints: list[Constraint],
    target_active_risk: float,
    progress_bar: tqdm_ray.tqdm | None = None,
    active_weights: bool = True,
    enforce_zero_beta: bool = True
):
    """
    Construct a single optimized portfolio for a given date.

    This is a Ray-remote helper function for parallel portfolio construction.
    It filters data to a specific date, constructs the factor model components,
    and solves a mean-variance optimization problem in active space.

    Parameters
    ----------
    date_ : dt.date
        The date for which to construct the portfolio.
    data : pl.DataFrame
        Full dataset with columns 'date', 'barrid', 'alpha', and optionally 
        'predicted_beta' and 'benchmark_weight'.
    constraints : list[Constraint]
        Portfolio constraints to enforce during optimization.
    target_active_risk : float
        Target annualized active risk for dual-variable gamma calibration.
    progress_bar : tqdm_ray.tqdm, optional
        Ray-based progress bar for tracking completion across parallel workers.
    active_weights : bool, default True
        Flag indicating the output format. True returns pure active weights. 
        False adds benchmark weights back to return fully invested total weights.
    enforce_zero_beta : bool, default True
        Flag to strictly enforce benchmark beta neutrality on the active overlay.
        Requires 'predicted_beta' in the dataset if set to True.

    Returns
    -------
    pl.DataFrame
        Portfolio weights for the given date, with columns 'date', 'barrid', 
        'weight', 'gamma', and 'active_risk'.
    """
    subset = data.filter(pl.col("date").eq(date_)).sort("barrid")
    barrids = subset["barrid"].to_list()
    alphas = subset["alpha"].to_numpy()

    if "benchmark_weight" not in subset.columns:
        raise ValueError("Column 'benchmark_weight' must be present in data.")
    benchmark_weights = subset["benchmark_weight"].to_numpy()

    if enforce_zero_beta and "predicted_beta" not in subset.columns:
        raise ValueError(
            "Column 'predicted_beta' must be present in data when enforce_zero_beta is True."
        )

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
        raise ValueError("Benchmark weights must be provided if active_weights flag is False.")

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
        target_active_risk=target_active_risk,
        betas=betas,
        benchmark_weights=benchmark_weights,
        active_weights=active_weights,
        enforce_zero_beta=enforce_zero_beta
    )

    portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
        "date", "barrid", "weight", "gamma", "active_risk"
    )

    if progress_bar is not None:
        progress_bar.update.remote(1)

    return portfolio


def dynamic_backtest_parallel(
    data: pl.DataFrame,
    constraints: list[Constraint],
    target_active_risk: float = 0.05,
    n_cpus: int | None = None,
    active_weights: bool = True,
    enforce_zero_beta: bool = True
) -> pl.DataFrame:
    """
    Run a parallelized backtest of portfolio optimization using Ray.

    This function distributes portfolio construction tasks across multiple CPUs
    using Ray, solving mean-variance optimization problems directly in active space
    for each date in parallel. Baseline fund constraints (long-only, dollar neutrality)
    are mathematically enforced under the hood.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataset containing at least the following columns:

        - ``date`` : datetime-like, the date of each observation.
        - ``barrid`` : str, unique identifier for each asset.
        - ``alpha`` : float, expected return (alpha) for each asset.
        - ``predicted_beta`` : float, optional, factor exposures for constraints.
        - ``benchmark_weight`` : float, optional, required if evaluating total weights.

    constraints : list[Constraint]
        List of portfolio constraints to enforce. Redundant total-space rules 
        are safely ignored by the optimizer to prevent solver conflicts.
    target_active_risk : float, default 0.05
        Automatically calibrate the implied risk aversion for each date to achieve 
        this target annualized active risk (e.g., 0.05 for 5%).
    n_cpus : int, optional
        Number of CPUs to allocate to Ray. If ``None``, defaults to
        ``os.cpu_count()`` but is capped at the number of unique dates.
    active_weights : bool, default True
        Flag indicating how to format output weights. True (default) evaluates the 
        pure active overlay. False returns the total fully invested portfolio.
    enforce_zero_beta : bool, default True
        Whether to strictly enforce zero active market beta. Set to False
        when researching factors with inherent systematic beta bias.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing optimized portfolio weights across all 
        backtest dates, with columns:

        - ``date`` : datetime, portfolio date.
        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.
        - ``gamma`` : float, calibrated risk aversion parameter for each date.
        - ``active_risk`` : float, achieved annualized active risk for each date.

    Notes
    -----
    - Benchmark data is required to map long-only bounds and to construct total 
      weights when ``active_weights`` is False.
    - Ray is initialized with ``ignore_reinit_error=True``, allowing safe
      re-invocation within the same process.

    See Also
    --------
    dynamic_mve_optimizer : Underlying active-space optimizer.

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
    ...     target_active_risk=0.05,
    ...     active_weights=True,
    ...     enforce_zero_beta=True
    ... )
    """
    dates = data["date"].unique().sort().to_list()

    n_cpus = n_cpus or os.cpu_count()
    n_cpus = min(len(dates), n_cpus)
    ray.init(ignore_reinit_error=True, num_cpus=n_cpus, runtime_env={})

    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    progress_bar = remote_tqdm.remote(
        total=len(dates), desc=f"Computing portfolios with {n_cpus} cpus"
    )

    portfolio_futures = [
        _construct_portfolio.remote(
            date_=date_,
            data=data,
            constraints=constraints,
            target_active_risk=target_active_risk,
            progress_bar=progress_bar,
            active_weights=active_weights,
            enforce_zero_beta=enforce_zero_beta
        )
        for date_ in dates
    ]

    portfolio_list = ray.get(portfolio_futures)

    progress_bar.close.remote()
    ray.shutdown()

    return pl.concat(portfolio_list)
