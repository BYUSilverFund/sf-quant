import datetime as dt
import polars as pl

from ._tables import optimal_weights_table

def load_optimal_weights(
    start: dt.date, end: dt.date, columns: list[str]
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of optimal weights data between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.
    columns : list of str
        List of column names to include in the result.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing optimal weights data between the specified dates,
        with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> columns = ["barrid", "date", "weight"]
    >>> df = sfd.load_optimal_weights(
    ...     start=start,
    ...     end=end,
    ...     columns=columns,
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌─────────┬────────────┬──────────┐
    │ barrid  ┆ date       ┆ weight   │
    │ ---     ┆ ---        ┆ ---      │
    │ str     ┆ date       ┆ f64      │
    ╞═════════╪════════════╪══════════╡
    │ USA06Z1 ┆ 2024-01-03 ┆ 0.012431 │
    │ USA0A21 ┆ 2024-01-03 ┆ -0.00482 │
    │ USA1BC1 ┆ 2024-01-03 ┆ 0.008117 │
    │ USA2DF1 ┆ 2024-01-03 ┆ 0.000000 │
    │ USA3GH1 ┆ 2024-01-03 ┆ -0.01564 │
    └─────────┴────────────┴──────────┘
    """
    return (
        optimal_weights_table.scan()
        .filter(pl.col("date").is_between(start, end))
        .sort(["barrid", "date"])
        .select(columns)
        .collect()
    )


def load_optimal_weights_by_date(
    date_: dt.date, columns: list[str]
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of optimal weights data for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the data frame.
    columns : list of str
        List of column names to include in the result.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing optimal weights data on the specified date,
        with only the selected columns.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> columns = ["barrid", "date", "weight"]
    >>> df = sf.data.load_optimal_weights_by_date(
    ...     date_=date_,
    ...     columns=columns,
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌─────────┬────────────┬──────────┐
    │ barrid  ┆ date       ┆ weight   │
    │ ---     ┆ ---        ┆ ---      │
    │ str     ┆ date       ┆ f64      │
    ╞═════════╪════════════╪══════════╡
    │ USA06Z1 ┆ 2024-01-03 ┆ 0.012431 │
    │ USA0A21 ┆ 2024-01-03 ┆ -0.00482 │
    │ USA1BC1 ┆ 2024-01-03 ┆ 0.008117 │
    │ USA2DF1 ┆ 2024-01-03 ┆ 0.000000 │
    │ USA3GH1 ┆ 2024-01-03 ┆ -0.01564 │
    └─────────┴────────────┴──────────┘
    """
    return (
        optimal_weights_table.scan(date_.year)
        .filter(pl.col("date").eq(date_))
        .sort(["barrid", "date"])
        .select(columns)
        .collect()
    )