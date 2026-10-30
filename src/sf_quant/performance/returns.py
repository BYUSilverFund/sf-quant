import polars as pl
from sf_quant.data._tables import assets_table
from sf_quant.data.benchmark import load_benchmark
from sf_quant.schema.portfolio_schema import PortfolioSchema
from sf_quant.schema.returns_schema import PortfolioRetSchema, MultiPortfolioRetSchema


def generate_multi_returns_from_weights(weights: PortfolioSchema) -> MultiPortfolioRetSchema:
    """
    Generate portfolio, benchmark, and active returns from given portfolio weights.

    This function calculates returns by joining provided portfolio weights with
    asset returns and benchmark weights. It derives total, benchmark, and
    active portfolio weights, then computes weighted returns for each
    portfolio type over time.

    Parameters
    ----------
    weights : PortfolioSchema
        Portfolio weights validated against PortfolioSchema.
        Must include the following columns:

        - ``date`` (date): The date for each weight.
        - ``barrid`` (str): The unique asset identifier.
        - ``weight`` (float): The portfolio weight assigned to the asset.

    Returns
    -------
    MultiPortfolioRetSchema
        A validated DataFrame containing portfolio returns with the
        following columns:

        - ``date`` (date): The observation date.
        - ``portfolio`` (str): Portfolio type; one of
          ``"total"``, ``"benchmark"``, or ``"active"``.
        - ``return`` (float): The weighted return for the portfolio
          on the given date.

    Notes
    -----
    - Asset returns are sourced via ``load_assets``.
    - Benchmark weights are sourced via ``load_benchmark``.
    - Returns are computed as the weighted sum of returns by portfolio.
    - Weights are matched with returns on the same date, so build them from
      signals lagged one day.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.performance as sfp
    >>> import datetime as dt
    >>> weights = pl.DataFrame(
    ...     {
    ...         'date': [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 3)],
    ...         'barrid': ['USA06Z1', 'USA0771', 'USA06Z1', 'USA0771'],
    ...         'weight': [0.5, 0.5, 0.3, 0.7]
    ...     }
    ... )
    >>> returns = sfp.generate_multi_returns_from_weights(weights)
    >>> returns
    shape: (6, 3)
    ┌────────────┬───────────┬────────────┐
    │ date       ┆ portfolio ┆ return     │
    │ ---        ┆ ---       ┆ ---        │
    │ date       ┆ str       ┆ f64        │
    ╞════════════╪═══════════╪════════════╡
    │ 2024-01-02 ┆ active    ┆ 0.015432   │
    │ 2024-01-02 ┆ benchmark ┆ 1.2047e-7  │
    │ 2024-01-02 ┆ total     ┆ 0.015432   │
    │ 2024-01-03 ┆ active    ┆ 0.020741   │
    │ 2024-01-03 ┆ benchmark ┆ 2.4094e-7  │
    │ 2024-01-03 ┆ total     ┆ 0.020741   │
    └────────────┴───────────┴────────────┘
    """
    start = weights["date"].min()
    end = weights["date"].max()

    returns = _held_returns(weights["barrid"], start, end)

    benchmark = load_benchmark(start=start, end=end)

    returns = returns.sort("barrid", "date")

    joined = (
        weights.join(returns, on=["date", "barrid"], how="left")
        .join(benchmark, on=["date", "barrid"], how="left", suffix="_bmk")
        .with_columns(
            pl.col("return").truediv(100),
        )
        .with_columns(pl.col("weight").sub("weight_bmk").alias("weight_act"))
        .rename({"weight": "total", "weight_bmk": "benchmark", "weight_act": "active"})
        .unpivot(
            index=["date", "barrid", "return"],
            variable_name="portfolio",
            value_name="weight",
        )
    )

    # Build aggregation expressions
    agg_exprs = []
    agg_exprs.append(pl.col("return").mul("weight").sum().alias("return"))

    result = (
        joined.group_by("date", "portfolio")
        .agg(agg_exprs)
        .sort("date", "portfolio")
    )
    return MultiPortfolioRetSchema.validate(result)


def generate_returns_from_weights(weights: PortfolioSchema) -> PortfolioRetSchema:
    """
    Generate portfolio returns from given portfolio weights.

    This function calculates returns by joining provided portfolio weights with
    asset returns, computing weighted returns over time.

    Parameters
    ----------
    weights : PortfolioSchema
        Portfolio weights validated against PortfolioSchema.
        Must include the following columns:

        - ``date`` (date): The date for each weight.
        - ``barrid`` (str): The unique asset identifier.
        - ``weight`` (float): The portfolio weight assigned to the asset.

    Returns
    -------
    PortfolioRetSchema
        A validated DataFrame containing portfolio returns with the
        following columns:

        - ``date`` (date): The observation date.
        - ``return`` (float): The weighted return for the portfolio on the given date.

    Notes
    -----
    - Asset returns are sourced via ``load_assets``.
    - Returns are computed as the weighted sum of returns by portfolio.
    - Weights are matched with returns on the same date, so build them from
      signals lagged one day.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.performance as sfp
    >>> import datetime as dt
    >>> weights = pl.DataFrame(
    ...     {
    ...         'date': [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 3)],
    ...         'barrid': ['USA06Z1', 'USA0771', 'USA06Z1', 'USA0771'],
    ...         'weight': [0.5, 0.5, 0.3, 0.7]
    ...     }
    ... )
    >>> returns = sfp.generate_returns_from_weights(weights)
    >>> returns
    shape: (2, 2)
    ┌────────────┬────────────┐
    │ date       ┆ return     │
    │ ---        ┆ ---        │
    │ date       ┆ f64        │
    ╞════════════╪════════════╡
    │ 2024-01-02 ┆ 0.015432   │
    │ 2024-01-03 ┆ 0.020741   │
    └────────────┴────────────┘
    """
    start = weights["date"].min()
    end = weights["date"].max()

    returns = _held_returns(weights["barrid"], start, end)

    returns = returns.sort("barrid", "date")

    joined = (
        weights.join(returns, on=["date", "barrid"], how="left")
        .with_columns(
            pl.col("return").truediv(100),
        )
    )

    result = (
        joined.group_by("date")
        .agg([
            pl.col("return").mul("weight").sum().alias("return"),
        ])
        .sort("date")
    )
    return PortfolioRetSchema.validate(result)


def _held_returns(barrids: pl.Series, start, end) -> pl.DataFrame:
    """
    Daily returns of the given names, whether or not they're in the universe.

    A held name earns its return even on a day it isn't flagged in the
    universe, e.g. after leaving the Russell indexes. Only the given names are
    read, filtered while the files are scanned.
    """
    return (
        assets_table.scan()
        .filter(
            pl.col("date").is_between(start, end),
            pl.col("barrid").is_in(barrids.unique().implode()),
        )
        .select("date", "barrid", "return")
        .collect()
    )
