import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt


def _compute_turnover(weights: pl.DataFrame) -> pl.DataFrame:
    return (
        weights.sort("date", "barrid")
        .with_columns(
            pl.col("weight")
            .sub(pl.col("weight").shift(1))
            .over("barrid")
            .alias("diff")
        )
        .group_by("date")
        .agg(pl.col("diff").abs().sum().alias("two_sided_turnover"))
        .sort("date")
        .with_columns(pl.col("two_sided_turnover").rolling_mean(252))
    )


def get_turnover_stats(weights: pl.DataFrame) -> pl.DataFrame:
    """
    Compute summary statistics for two-sided portfolio turnover.

    Calculates the rolling 252-day mean two-sided turnover and summarises
    it with mean, min, and max statistics.

    Parameters
    ----------
    weights : pl.DataFrame
        Portfolio weights containing:

        - ``date`` (date): The observation date.
        - ``barrid`` (str): Security identifier.
        - ``weight`` (float): Portfolio weight.

    Returns
    -------
    pl.DataFrame
        Summary statistics with columns:

        - ``Mean Turnover`` (float): Mean of the rolling two-sided turnover.
        - ``Min Turnover`` (float): Minimum of the rolling two-sided turnover.
        - ``Max Turnover`` (float): Maximum of the rolling two-sided turnover.

    Notes
    -----
    - Two-sided turnover is the sum of absolute weight changes per date.
    - Rolling window is 252 trading days.
    - Null values (warm-up period) are excluded from summary statistics.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.performance as sfp
    >>> import datetime as dt
    >>> weights = pl.DataFrame({
    ...     'date': [dt.date(2024, 1, i) for i in range(2, 6)],
    ...     'barrid': ['A'] * 4,
    ...     'weight': [0.5, 0.6, 0.4, 0.5],
    ... })
    >>> sfp.get_turnover_stats(weights)
    shape: (1, 3)
    ┌───────────────┬───────────────┬───────────────┐
    │ Mean Turnover ┆ Min Turnover  ┆ Max Turnover  │
    │ ---           ┆ ---           ┆ ---           │
    │ f64           ┆ f64           ┆ f64           │
    ╞═══════════════╪═══════════════╪═══════════════╡
    │ 0.15          ┆ 0.1           ┆ 0.2           │
    └───────────────┴───────────────┴───────────────┘
    """
    turnover = _compute_turnover(weights)

    return (
        turnover.drop_nulls("two_sided_turnover")
        .select(
            pl.col("two_sided_turnover").mean().alias("Mean Turnover"),
            pl.col("two_sided_turnover").min().alias("Min Turnover"),
            pl.col("two_sided_turnover").max().alias("Max Turnover"),
        )
        .with_columns(cs.float().round(4))
    )


def plot_turnover(
    weights: pl.DataFrame,
    title: str,
    subtitle: str | None = None,
    file_name: str | None = None,
) -> None:
    """
    Plot rolling two-sided turnover over time.

    Parameters
    ----------
    weights : pl.DataFrame
        Portfolio weights containing:

        - ``date`` (date): The observation date.
        - ``barrid`` (str): Security identifier.
        - ``weight`` (float): Portfolio weight.
    title : str
        The chart's main title.
    subtitle : str or None, optional
        The chart's subtitle. Defaults to ``None``.
    file_name : str or None, optional
        If not ``None``, saves the chart to the given file path.
        Defaults to displaying the chart interactively.

    Returns
    -------
    None

    Notes
    -----
    - Two-sided turnover is the sum of absolute weight changes per date.
    - Rolling window is 252 trading days.
    """
    turnover = _compute_turnover(weights)

    plt.figure(figsize=(10, 6))
    plt.plot(turnover["date"], turnover["two_sided_turnover"])

    plt.suptitle(title)
    plt.title(subtitle)

    plt.xlabel(None)
    plt.ylabel("Two-Sided Turnover (Rolling 252-Day Mean)")

    plt.grid()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
