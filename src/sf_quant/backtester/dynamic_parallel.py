import polars as pl
import os
import tqdm
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.optimizers import dynamic_mve_optimizer
from sf_quant.optimizer.constraints import Constraint


def _construct_portfolio(
    date_: dt.date,
    subset: pl.DataFrame,
    constraints: list[Constraint],
    target_active_risk: float,
    active_weights: bool = True,
):
    """
    Construct a single optimized portfolio for a given date.

    Parameters
    ----------
    date_ : dt.date
        The date for which to construct the portfolio.
    subset : pl.DataFrame
        Single-date slice with columns ``date``, ``barrid``, ``alpha``, and
        optionally ``predicted_beta`` and ``benchmark_weight``.
    constraints : list[Constraint]
        Portfolio constraints to enforce during optimization.
    target_active_risk : float
        Target annualized active risk for dual-variable gamma calibration.
    active_weights : bool, default True
        True returns pure active weights; False adds benchmark weights back to
        return total portfolio weights.

    Returns
    -------
    pl.DataFrame
        Portfolio weights for the given date, with columns ``date``, ``barrid``,
        ``weight``, ``gamma``, and ``active_risk``.
    """
    subset = subset.sort("barrid")
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

    if benchmark_weights is None and not active_weights:
        raise ValueError(
            "Column 'benchmark_weight' must be present in data when active_weights is False."
        )

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
    )

    portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
        "date", "barrid", "weight", "gamma", "active_risk"
    )

    return portfolio


def dynamic_backtest_parallel(
    data: pl.DataFrame,
    constraints: list[Constraint],
    target_active_risk: float = 0.05,
    n_cpus: int | None = None,
    active_weights: bool = True,
) -> pl.DataFrame:
    """
    Run a parallelized backtest of portfolio optimization using a thread pool.

    Distributes single-date portfolio construction tasks across worker
    threads, solving mean-variance optimization problems in active space with
    explicitly supplied ``Constraint`` objects.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataset containing:

        - ``date`` : datetime-like, the date of each observation.
        - ``barrid`` : str, unique identifier for each asset.
        - ``alpha`` : float, expected return (alpha) for each asset.
        - ``predicted_beta`` : float, optional, required if ``ZeroBeta`` or
          ``UnitBeta`` is included in ``constraints``.
        - ``benchmark_weight`` : float, optional, required if total-space
          constraints (such as ``LongOnly``, ``NoBuyingOnMargin``, or
          ``GrossLeverage``) are used relative to a benchmark or if
          ``active_weights`` is False.

    constraints : list[Constraint]
        List of portfolio constraints to enforce (e.g.,
        ``[ZeroInvestment(), LongOnly(), ZeroBeta()]`` for the Silver Fund
        mandate, ``[ZeroInvestment(), GrossLeverage(1.6)]`` for 130/30, or
        ``[ZeroInvestment()]`` for unconstrained signal research).
    target_active_risk : float, default 0.05
        Target annualized active risk cap (e.g., 0.05 for 5%).
    n_cpus : int, optional
        Number of worker threads. If ``None``, defaults to
        ``os.cpu_count()`` capped at the number of unique dates.
    active_weights : bool, default True
        True (default) outputs active weights :math:`h`. False outputs total
        portfolio weights :math:`w = h + b`.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing optimized portfolio weights across all
        backtest dates, with columns ``date``, ``barrid``, ``weight``,
        ``gamma``, and ``active_risk``.
    """
    if not active_weights and "benchmark_weight" not in data.columns:
        raise ValueError(
            "Column 'benchmark_weight' must be present in data when active_weights is False."
        )

    partitions = {
        df["date"][0]: df for df in data.partition_by("date")
    }
    dates = sorted(partitions.keys())

    if not dates:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "barrid": pl.String,
                "weight": pl.Float64,
                "gamma": pl.Float64,
                "active_risk": pl.Float64,
            }
        )

    n_cpus = n_cpus or os.cpu_count() or 1
    n_cpus = min(len(dates), n_cpus)

    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        portfolio_futures = executor.map(
            lambda date_: _construct_portfolio(
                date_=date_,
                subset=partitions[date_],
                constraints=constraints,
                target_active_risk=target_active_risk,
                active_weights=active_weights,
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

    return pl.concat(portfolio_list)