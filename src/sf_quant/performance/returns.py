import polars as pl
from sf_quant.data.assets import load_assets
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

        - ``date`` (date): The formation date that opened the period.
        - ``portfolio`` (str): Portfolio type; one of
          ``"total"``, ``"benchmark"``, or ``"active"``.
        - ``return`` (float): The portfolio's return over the period.

    Notes
    -----
    - Asset returns are sourced via ``load_assets``.
    - Benchmark weights are sourced via ``load_benchmark``, on each formation
      date.
    - Each row is the return earned after its formation date, never on it:
      a portfolio formed at :math:`t_k` earns each asset's return compounded
      over :math:`(t_k, t_{k+1}]`. A K-date grid gives K-1 rows per portfolio.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> returns = sfp.generate_multi_returns_from_weights(weights)   # doctest: +SKIP
    """
    start = weights["date"].min()
    end = weights["date"].max()

    benchmark = load_benchmark(start=start, end=end)

    stacked = (
        weights.join(benchmark, on=["date", "barrid"], how="left", suffix="_bmk")
        .with_columns(pl.col("weight").sub("weight_bmk").alias("weight_act"))
        .rename({"weight": "total", "weight_bmk": "benchmark", "weight_act": "active"})
        .unpivot(
            index=["date", "barrid"],
            variable_name="portfolio",
            value_name="weight",
        )
    )

    result = _earn(stacked, _period_returns(weights), "weight", by=("portfolio",))
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

        - ``date`` (date): The formation date that opened the period.
        - ``return`` (float): The portfolio's return over the period.

    Notes
    -----
    - Asset returns are sourced via ``load_assets``.
    - Each row is the return earned after its formation date, never on it:
      a portfolio formed at :math:`t_k` earns each asset's return compounded
      over :math:`(t_k, t_{k+1}]`. A K-date grid gives K-1 rows.

    Examples
    --------
    >>> import sf_quant.performance as sfp
    >>> returns = sfp.generate_returns_from_weights(weights)   # doctest: +SKIP
    """
    result = _earn(weights, _period_returns(weights), "weight")
    return PortfolioRetSchema.validate(result)


def _period_returns(weights: pl.DataFrame) -> pl.DataFrame:
    """Each asset's compounded return over every holding period on the grid.

    Keyed by ``form_date``, the formation date that opened the period. A period
    runs from the day after its formation date through the next formation date,
    so a portfolio never earns the return of the day it was built: that return
    is already known when the weights are formed. A name with no observations
    in a period contributes nothing.
    """
    formation_dates = weights["date"].unique().sort()
    if formation_dates.len() < 2:
        raise ValueError(
            "need at least two rebalance dates to measure one holding period, got "
            f"{formation_dates.len()}"
        )

    daily = (
        load_assets(
            start=formation_dates.min(),
            end=formation_dates.max(),
            columns=["date", "barrid", "return"],
            in_universe=True,
        )
        .with_columns(pl.col("return").truediv(100))
        .drop_nulls("return")
    )

    calendar = pl.DataFrame({"date": daily["date"].unique().sort()}).with_row_index("t")

    # Each trading day belongs to the period opened by the most recent formation
    # date *strictly* before it. Matching against the day after each formation
    # date is what makes the backward join strict: a formation date's own return
    # closes the previous period rather than opening its own.
    opens = calendar.filter(pl.col("date").is_in(formation_dates.implode())).select(
        (pl.col("t") + 1).alias("t_open"), pl.col("date").alias("form_date")
    )

    buckets = (
        calendar.join_asof(opens, left_on="t", right_on="t_open", strategy="backward")
        .drop_nulls("form_date")
        .select("date", "form_date")
    )

    return (
        daily.join(buckets, on="date", how="inner")
        .group_by("form_date", "barrid")
        .agg(pl.col("return").add(1).product().sub(1).alias("return"))
    )


def _earn(
    weights: pl.DataFrame, period_returns: pl.DataFrame, weight: str, by: tuple[str, ...] = ()
) -> pl.DataFrame:
    """Sum of weight times period return per formation date (and per ``by``)."""
    return (
        weights.filter(
            pl.col("date").is_in(period_returns["form_date"].unique().implode())
        )
        .join(
            period_returns,
            left_on=["date", "barrid"],
            right_on=["form_date", "barrid"],
            how="left",
        )
        .group_by("date", *by)
        .agg(
            pl.col(weight)
            .mul(pl.col("return").fill_null(0.0))
            .sum()
            .alias("return")
        )
        .sort("date", *by)
    )
