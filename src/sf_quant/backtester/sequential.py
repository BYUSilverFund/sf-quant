import polars as pl
import tqdm
from sf_quant.data.covariance_matrix import construct_covariance_matrix
from sf_quant.optimizer.optimizers import mve_optimizer
from sf_quant.optimizer.constraints import Constraint

def backtest_sequential(data: pl.DataFrame, constraints: list[Constraint], gamma: float = 2) -> pl.DataFrame:
    """
    Run a sequential backtest of portfolio optimization.

    This function iterates through unique dates in the input dataset,
    constructs the covariance matrix for each date, and solves a mean–
    variance optimization (MVE) problem subject to user-specified
    constraints. The optimized portfolios are concatenated into a
    single Polars DataFrame.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataset containing at least the following columns:
        
        - ``date`` : datetime-like, the date of each observation.
        - ``barrid`` : str, unique identifier for each asset.
        - ``alpha`` : float, expected return (alpha) for each asset.
        - ``predicted_beta`` : float, optional, factor exposures for constraints.

    constraints : list[Constraint]
        A list of constraints (e.g., :class:`~sf_quant.optimizer.constraints.FullInvestment`,
        :class:`~sf_quant.optimizer.constraints.LongOnly`) to be enforced in the optimization.

    gamma : float, optional
        Risk aversion parameter. Higher values penalize portfolio variance more strongly.
        Default is 2.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing optimized portfolio weights for each date, 
        with the following columns:
        
        - ``date`` : datetime, backtest date.
        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.

    Notes
    -----
    - The optimization is solved using :func:`~sf_quant.optimizer.optimizers.mve_optimizer`.
    - The covariance matrix is constructed using
      :func:`~sf_quant.data.covariance_matrix.construct_covariance_matrix`.

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
    >>> weights = sfb.backtest_sequential(
    ...     data=data,
    ...     constraints=constraints,
    ...     gamma=2,
    ... )
    >>> weights.head()
    shape: (5, 3)
    ┌────────────┬─────────┬───────────┐
    │ date       ┆ barrid  ┆ weight    │
    │ ---        ┆ ---     ┆ ---       │
    │ date       ┆ str     ┆ f64       │
    ╞════════════╪═════════╪═══════════╡
    │ 2024-01-02 ┆ USA06Z1 ┆ -0.000639 │
    │ 2024-01-03 ┆ USA06Z1 ┆ -0.000644 │
    │ 2024-01-04 ┆ USA06Z1 ┆ -0.000641 │
    │ 2024-01-05 ┆ USA06Z1 ┆ -0.000666 │
    │ 2024-01-08 ┆ USA06Z1 ┆ -0.000666 │
    └────────────┴─────────┴───────────┘
    """
    dates = data['date'].unique().sort().to_list()
    portfolio_list = []
    for date_ in tqdm.tqdm(dates, "Running backtest"):
        subset = data.filter(pl.col("date").eq(date_)).sort("barrid")

        barrids = subset["barrid"].to_list()
        alphas = subset["alpha"].to_numpy()

        betas = subset["predicted_beta"].to_numpy() if 'predicted_beta' in subset.columns else None

        covariance_matrix = (
            construct_covariance_matrix(date_, barrids).drop("barrid").to_numpy()
        )

        portfolio = mve_optimizer(
            ids=barrids,
            alphas=alphas,
            covariance_matrix=covariance_matrix,
            gamma=gamma,
            constraints=constraints,
            betas=betas
        )

        portfolio = (
            portfolio
            .with_columns(pl.lit(date_).alias('date'))
            .select('date', 'barrid', 'weight')
        )

        portfolio_list.append(portfolio)
    
    return pl.concat(portfolio_list).sort('barrid', 'date')