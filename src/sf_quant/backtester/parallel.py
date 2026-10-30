import polars as pl
import os
import tqdm
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.optimizers import mve_optimizer
from sf_quant.optimizer.constraints import Constraint
from sf_quant.schema.portfolio_schema import PortfolioSchema


def _construct_portfolio(
    date_: dt.date,
    data: pl.DataFrame,
    constraints: list[Constraint],
    gamma: float,
):
    """
    Construct a single optimized portfolio for a given date.

    This is a helper function for parallel portfolio construction, run once
    per date on a worker thread. It filters data to a specific date, constructs the factor model components,
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

    (
        factor_exposures,
        factor_covariance,
        specific_risk,
    ) = construct_factor_model_components(date_, barrids)

    portfolio = mve_optimizer(
        ids=barrids,
        alphas=alphas,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        gamma=gamma,
        constraints=constraints,
        betas=betas,
    )

    portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
        "date", "barrid", "weight"
    )

    return portfolio


def backtest_parallel(
    data: pl.DataFrame,
    constraints: list[Constraint],
    gamma: float = 2,
    n_cpus: int | None = None,
) -> pl.DataFrame:
    """
    Run a parallelized backtest of portfolio optimization using a thread pool.

    This function distributes portfolio construction tasks across
    worker threads, solving mean–variance optimization problems
    for each date in parallel. A progress bar tracks computation
    progress.

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
    gamma : float, optional
        Risk aversion parameter. Higher values penalize portfolio
        variance more strongly. Default is 2.
    n_cpus : int, optional
        Number of worker threads. If ``None``, defaults to
        ``os.cpu_count()`` but is capped at the number of unique dates.

    Returns
    -------
    pl.DataFrame
        A PortfolioSchema-validated Polars DataFrame containing optimized
        portfolio weights across all backtest dates, with columns:

        - ``date`` : datetime, portfolio date.
        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.

    Notes
    -----
    - Threads share the input data, so nothing is copied to the workers. The
      solvers release the GIL, so the solves run in parallel.
    - The number of threads is automatically limited to the number of unique dates
      to avoid unnecessary idle workers.

    See Also
    --------
    backtest_sequential : Sequential version for smaller datasets or debugging.
    dynamic_backtest_parallel : Parallel version with active risk calibration.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import sf_quant.backtester as sfb
    >>> import sf_quant.optimizer as sfo
    >>> import polars as pl
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 1, 10)
    >>> columns = [
    ...     'date',
    ...     'barrid',
    ... ]
    >>> data = (
    ...     sfd.load_assets(
    ...         start=start,
    ...         end=end,
    ...         in_universe=True,
    ...         columns=columns
    ...     )
    ...     .with_columns(
    ...         pl.lit(0).alias('alpha')
    ...     )
    ... )
    >>> constraints = [
    ...     sfo.FullInvestment()
    ... ]
    >>> weights = sfb.backtest_parallel(
    ...     data=data,
    ...     constraints=constraints,
    ...     gamma=2,
    ... )
    shape: (5, 3)
    ┌────────────┬─────────┬───────────┐
    │ date       ┆ barrid  ┆ weight    │
    │ ---        ┆ ---     ┆ ---       │
    │ date       ┆ str     ┆ f64       │
    ╞════════════╪═════════╪═══════════╡
    │ 2024-01-02 ┆ USA06Z1 ┆ -0.000639 │
    │ 2024-01-02 ┆ USA0771 ┆ -0.000083 │
    │ 2024-01-02 ┆ USA0C11 ┆ -0.003044 │
    │ 2024-01-02 ┆ USA0SY1 ┆ -0.002177 │
    │ 2024-01-02 ┆ USA11I1 ┆ 0.001475  │
    └────────────┴─────────┴───────────┘
    """
    # Get dates
    dates = data["date"].unique().sort().to_list()

    n_cpus = n_cpus or os.cpu_count()
    n_cpus = min(len(dates), n_cpus)

    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        portfolio_futures = executor.map(
            lambda date_: _construct_portfolio(
                date_=date_,
                data=data,
                constraints=constraints,
                gamma=gamma,
            ),
            dates,
        )
        portfolio_list = list(
            tqdm.tqdm(
                portfolio_futures,
                total=len(dates),
                desc=f"Computing portfolios with {n_cpus} threads",
            )
        )

    return PortfolioSchema.validate(pl.concat(portfolio_list))
