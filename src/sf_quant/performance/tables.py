import polars as pl
import polars.selectors as cs


def generate_summary_table(returns: pl.DataFrame, wide: bool = True) -> pl.DataFrame:
    """
    Generate a summary statistics table for portfolio returns.

    This function calculates performance metrics for each portfolio, including
    mean return, volatility, total return, and Sharpe ratio. Results can be
    returned either in wide format (one row per portfolio) or in long format
    (statistics transposed into rows).

    Parameters
    ----
        returns (pl.DataFrame): A Polars DataFrame containing portfolio returns.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``portfolio`` (str): Portfolio name or identifier.
            - ``return`` (float): Daily portfolio return.
        wide (bool, optional): If ``True`` (default), return the summary in wide
            format with one row per portfolio. If ``False``, return the summary
            transposed with statistics as rows.

    Returns
    -------
        pl.DataFrame: A Polars DataFrame containing portfolio summary statistics.
        The exact structure depends on the ``wide`` argument:

        - **Wide format (default):**
            - ``Portfolio`` (str): Portfolio name.
            - ``Count`` (int): Number of days in the sample.
            - ``Mean Return (%)`` (float): Annualized mean return (in percent).
            - ``Volatility (%)`` (float): Annualized volatility (in percent).
            - ``Total Return (%)`` (float): Total return over the period (in percent).
            - ``Sharpe`` (float): Sharpe ratio.

        - **Long format (wide=False):**
            - ``statistics`` (str): Statistic name.
            - One column per portfolio containing the respective values.

    Notes
    -----
        - Annualization assumes 252 trading days per year.
        - Returns are converted to percentages for readability.
        - Sharpe ratio is calculated as ``mean_return / volatility``.

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
    >>> summary = sfp.generate_summary_table(returns)
    >>> summary
    shape: (3, 6)
    ┌───────────┬───────┬─────────────────┬────────────────┬──────────────────┬────────┐
    │ Portfolio ┆ Count ┆ Mean Return (%) ┆ Volatility (%) ┆ Total Return (%) ┆ Sharpe │
    │ ---       ┆ ---   ┆ ---             ┆ ---            ┆ ---              ┆ ---    │
    │ str       ┆ u32   ┆ f64             ┆ f64            ┆ f64              ┆ f64    │
    ╞═══════════╪═══════╪═════════════════╪════════════════╪══════════════════╪════════╡
    │ Active    ┆ 2     ┆ -194.28         ┆ 63.87          ┆ -1.62            ┆ -3.04  │
    │ Benchmark ┆ 2     ┆ -0.0            ┆ 0.0            ┆ -0.0             ┆ -4.01  │
    │ Total     ┆ 2     ┆ -194.28         ┆ 63.87          ┆ -1.62            ┆ -3.04  │
    └───────────┴───────┴─────────────────┴────────────────┴──────────────────┴────────┘
    """
    summary_wide = (
        returns.group_by("portfolio")
        .agg(
            pl.col("date").n_unique().alias("n_days"),
            pl.col("return").mean().alias("mean_return"),
            pl.col("return").std().alias("volatility"),
            pl.col("return").add(1).product().sub(1).alias("total_return"),
        )
        .with_columns(
            pl.col("mean_return").mul(252),
            pl.col("volatility").mul(pl.lit(252).sqrt()),
        )
        .with_columns(pl.col("mean_return").truediv("volatility").alias("sharpe"))
        .with_columns(pl.col("mean_return", "volatility", "total_return").mul(100))
        .sort("portfolio")
        .with_columns(pl.col("portfolio").str.to_titlecase())
        .rename(
            {
                "portfolio": "Portfolio",
                "n_days": "Count",
                "mean_return": "Mean Return (%)",
                "volatility": "Volatility (%)",
                "total_return": "Total Return (%)",
                "sharpe": "Sharpe",
            }
        )
        .with_columns(cs.float().round(2))
    )

    if wide:
        return summary_wide

    else:
        return (
            summary_wide.drop("Portfolio")
            .transpose(
                include_header=True,
                header_name="statistics",
                column_names=summary_wide["Portfolio"],
            )
            .with_columns(cs.float().round(2))
        )
