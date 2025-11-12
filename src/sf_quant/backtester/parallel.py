import polars as pl
import os
import ray
import datetime as dt
from ray.experimental import tqdm_ray
from sf_quant.data.covariance_matrix import construct_covariance_matrix
from sf_quant.optimizer.optimizers import mve_optimizer
from sf_quant.optimizer.constraints import Constraint
from sf_quant.schema.portfolio_schema import PortfolioSchema


@ray.remote
def _construct_portfolio(
    date_: dt.date,
    data: pl.DataFrame,
    constraints: list[Constraint],
    gamma: float,
    progress_bar: tqdm_ray.tqdm | None = None,
):
    subset = data.filter(pl.col("date").eq(date_)).sort("barrid")
    barrids = subset["barrid"].to_list()
    alphas = subset["alpha"].to_numpy()

    betas = (
        subset["predicted_beta"].to_numpy()
        if "predicted_beta" in subset.columns
        else None
    )

    covariance_matrix = (
        construct_covariance_matrix(date_, barrids).drop("barrid").to_numpy()
    )

    portfolio = mve_optimizer(
        ids=barrids,
        alphas=alphas,
        covariance_matrix=covariance_matrix,
        gamma=gamma,
        constraints=constraints,
        betas=betas,
    )

    portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
        "date", "barrid", "weight"
    )

    if progress_bar is not None:
        progress_bar.update.remote(1)

    return portfolio


def backtest_parallel(
    data: pl.DataFrame,
    constraints: list[Constraint],
    gamma: float = 2,
    n_cpus: int | None = None,
) -> pl.DataFrame:
    """
    Run a parallelized backtest of portfolio optimization using Ray.

    This function distributes portfolio construction tasks across
    multiple CPUs with Ray, solving mean–variance optimization problems
    for each date in parallel. A Ray-based progress bar tracks
    computation progress.

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
        Number of CPUs to allocate to Ray. If ``None``, defaults to
        ``os.cpu_count()`` but is capped at the number of unique dates.

    Returns
    -------
    pl.DataFrame
        A PortfolioSchema-validated Polars DataFrame containing optimized
        portfolio weights across all backtest dates, with columns:

        - ``date`` : datetime, portfolio date.
        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.

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

    # Set up ray
    n_cpus = n_cpus or os.cpu_count()
    n_cpus = min(len(dates), n_cpus)
    ray.init(ignore_reinit_error=True, num_cpus=n_cpus)

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
            gamma=gamma,
            progress_bar=progress_bar,
        )
        for date_ in dates
    ]

    portfolio_list = ray.get(portfolio_futures)

    progress_bar.close.remote()
    ray.shutdown()

    return PortfolioSchema.validate(pl.concat(portfolio_list))
