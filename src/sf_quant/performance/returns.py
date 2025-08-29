import polars as pl
from sf_quant.data.assets import load_assets
from sf_quant.data.benchmark import load_benchmark


def generate_returns_from_weights(weights: pl.DataFrame) -> pl.DataFrame:
    """
    Generate portfolio, benchmark, and active returns from given portfolio weights.

    This function calculates returns by joining provided portfolio weights with
    asset forward returns and benchmark weights. It derives total, benchmark, and
    active portfolio weights, then computes weighted forward returns for each
    portfolio type over time.

    Parameters
    ----------
        weights (pl.DataFrame): A Polars DataFrame containing portfolio weights.
            Must include the following columns:
            - ``date`` (date): The date for each weight.
            - ``barrid`` (str): The unique asset identifier.
            - ``weight`` (float): The portfolio weight assigned to the asset.

    Returns
    -------
        pl.DataFrame: A Polars DataFrame containing portfolio returns with the
        following columns:
            - ``date`` (date): The observation date.
            - ``portfolio`` (str): Portfolio type; one of
              ``"total"``, ``"benchmark"``, or ``"active"``.
            - ``return`` (float): The weighted forward return for the portfolio
              on the given date.

    Notes
    -----
        - Asset returns are sourced via ``load_assets`` with ``fwd_return`` column.
        - Benchmark weights are sourced via ``load_benchmark``.
        - Returns are computed as the weighted sum of forward returns by portfolio.

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
    shape: (6, 3)
    ┌────────────┬───────────┬────────────┐
    │ date       ┆ portfolio ┆ return     │
    │ ---        ┆ ---       ┆ ---        │
    │ date       ┆ str       ┆ f64        │
    ╞════════════╪═══════════╪════════════╡
    │ 2024-01-02 ┆ active    ┆ 0.020741   │
    │ 2024-01-02 ┆ benchmark ┆ 2.4094e-7  │
    │ 2024-01-02 ┆ total     ┆ 0.020741   │
    │ 2024-01-03 ┆ active    ┆ -0.03616   │
    │ 2024-01-03 ┆ benchmark ┆ -5.0834e-7 │
    │ 2024-01-03 ┆ total     ┆ -0.03616   │
    └────────────┴───────────┴────────────┘
    """
    start = weights["date"].min()
    end = weights["date"].max()

    columns = ["date", "barrid", "fwd_return"]

    returns = load_assets(start=start, end=end, in_universe=True, columns=columns)

    benchmark = load_benchmark(start=start, end=end)

    return (
        weights.join(returns, on=["date", "barrid"], how="left")
        .join(benchmark, on=["date", "barrid"], how="left", suffix="_bmk")
        .with_columns(pl.col("fwd_return").truediv(100))
        .with_columns(pl.col("weight").sub("weight_bmk").alias("weight_act"))
        .rename({"weight": "total", "weight_bmk": "benchmark", "weight_act": "active"})
        .unpivot(
            index=["date", "barrid", "fwd_return"],
            variable_name="portfolio",
            value_name="weight",
        )
        .group_by("date", "portfolio")
        .agg(pl.col("fwd_return").mul("weight").sum().alias("return"))
        .sort("date", "portfolio")
    )
