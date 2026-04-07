import datetime as dt
import polars as pl

from ._tables import composite_alphas_table


def load_composite_alphas(
    start: dt.date, end: dt.date
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of composite alpha data between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing alpha data between the specified dates.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> df = sfd.load_composite_alphas(
    ...     start=start,
    ...     end=end,
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌────────────┬─────────┬─────────────┐
    │ date       ┆ barrid  ┆ alpha       │
    │ ---        ┆ ---     ┆ ---         │
    │ date       ┆ str     ┆ f64         │
    ╞════════════╪═════════╪═════════════╡
    │ 2023-01-03 ┆ USA06Z1 ┆ -17.611814  │
    │ 2023-01-03 ┆ USA06Z2 ┆ 1.55668     │
    │ 2023-01-04 ┆ USA06Z1 ┆ -18.676612  │
    │ 2023-01-04 ┆ USA06Z2 ┆ 3.775271    │
    │ 2023-01-05 ┆ USA06Z1 ┆ -17.06099   │
    └────────────┴─────────┴─────────────┘
    """
    return (
        composite_alphas_table.scan()
        .filter(pl.col("date").is_between(start, end))
        .sort(["barrid", "date"])
        .collect()
    )


def load_composite_alphas_by_date(
    date_: dt.date,
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of alphas data for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the data frame.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing alphas data on the specified date.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> df = sf.data.load_composite_alphas_by_date(
    ...     date_=date_,
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌────────────┬─────────┬─────────────┐
    │ date       ┆ barrid  ┆ alpha       │
    │ ---        ┆ ---     ┆ ---         │
    │ date       ┆ str     ┆ f64         │
    ╞════════════╪═════════╪═════════════╡
    │ 2023-01-03 ┆ USA06Z1 ┆ -17.611814  │
    │ 2023-01-03 ┆ USA06Z2 ┆ 1.55668     │
    │ 2023-01-03 ┆ USA06Z3 ┆ -18.676612  │
    │ 2023-01-03 ┆ USA06Z4 ┆ 3.775271    │
    │ 2023-01-03 ┆ USA06Z5 ┆ -17.06099   │
    └────────────┴─────────┴─────────────┘
    """
    return (
        composite_alphas_table.scan(date_.year)
        .filter(pl.col("date").eq(date_))
        .sort(["barrid", "date"])
        .collect()
    )
