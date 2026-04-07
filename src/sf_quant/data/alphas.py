import datetime as dt
import polars as pl

from ._tables import alphas_table


def load_alphas(
    start: dt.date, end: dt.date, names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of alpha data between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.
    names : list of str, optional
        List of signal names to filter the data frame by.
        If None (default), all alpha names are included.

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
    >>> df = sfd.load_alphas(
    ...     start=start,
    ...     end=end,
    ...     names=["momentum", "reversal"]
    ... )
    >>> df.head()
    shape: (5, 4)
    ┌────────────┬─────────┬─────────────┬─────────────┐
    │ date       ┆ barrid  ┆ signal_name ┆ alpha_value │
    │ ---        ┆ ---     ┆ ---         ┆ ---         │
    │ date       ┆ str     ┆ str         ┆ f64         │
    ╞════════════╪═════════╪═════════════╪═════════════╡
    │ 2023-01-03 ┆ USA06Z1 ┆ momentum    ┆ -17.611814  │
    │ 2023-01-03 ┆ USA06Z1 ┆ reversal    ┆ 1.55668     │
    │ 2023-01-04 ┆ USA06Z1 ┆ momentum    ┆ -18.676612  │
    │ 2023-01-04 ┆ USA06Z1 ┆ reversal    ┆ 3.775271    │
    │ 2023-01-05 ┆ USA06Z1 ┆ momentum    ┆ -17.06099   │
    └────────────┴─────────┴─────────────┴─────────────┘
    """
    if names is not None:
        return (
            alphas_table.scan()
            .filter(
                pl.col("date").is_between(start, end),
                pl.col("signal_name").is_in(names),
            )
            .sort(["barrid", "date"])
            .collect()
        )

    else:
        return (
            alphas_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .collect()
        )


def load_alphas_by_date(
    date_: dt.date, names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of alphas data for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the data frame.
    names : list of str, optional
        List of signal names to filter the data frame by.
        If None (default), all alpha names are included.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing alphas data on the specified date.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> df = sf.data.load_alphas_by_date(
    ...     date_=date_,
    ...     names=["momentum", "reversal"]
    ... )
    >>> df.head()
    shape: (5, 4)
    ┌────────────┬─────────┬─────────────┬─────────────┐
    │ date       ┆ barrid  ┆ signal_name ┆ alpha_value │
    │ ---        ┆ ---     ┆ ---         ┆ ---         │
    │ date       ┆ str     ┆ str         ┆ f64         │
    ╞════════════╪═════════╪═════════════╪═════════════╡
    │ 2023-01-03 ┆ USA06Z1 ┆ momentum    ┆ -17.611814  │
    │ 2023-01-03 ┆ USA06Z1 ┆ reversal    ┆ 1.55668     │
    │ 2023-01-03 ┆ USA06Z1 ┆ momentum    ┆ -18.676612  │
    │ 2023-01-03 ┆ USA06Z1 ┆ reversal    ┆ 3.775271    │
    │ 2023-01-03 ┆ USA06Z1 ┆ momentum    ┆ -17.06099   │
    └────────────┴─────────┴─────────────┴─────────────┘
    """
    if names is not None:
        return (
            alphas_table.scan()
            .filter(
                pl.col("date").eq(date_),
                pl.col("signal_name").is_in(names),
            )
            .sort(["barrid", "date"])
            .collect()
        )
    else:
        return (
            alphas_table.scan(date_.year)
            .filter(pl.col("date").eq(date_))
            .sort(["barrid", "date"])
            .collect()
        )


def get_alpha_names() -> list[str]:
    """
    Return the list of available alpha (signal) names.

    Returns
    -------
    list of str
        A list of unique signal names in the alphas dataset.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_alpha_names()
    ["momentum", "reversal", ...]
    """
    return (
        alphas_table.scan()
        .select("signal_name")
        .unique()
        .collect()["signal_name"]
        .to_list()
    )
