import datetime as dt
import polars as pl

from ._tables import alphas_table, combined_alphas_table


def load_alphas(
    start: dt.date, end: dt.date, columns: list[str], signal_names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of alpha data between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.
    columns : list of str
        List of column names to include in the result.
    signal_names : list of str
        List of signal names to filter the data frame by.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing alpha data between the specified dates,
        with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> columns = ["barrid", "date", "signal_name", "alpha_value"]
    >>> df = sfd.load_alphas(
    ...     start=start,
    ...     end=end,
    ...     columns=columns,
    ...     signal_names=["momentum", "reversal"]
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
    if signal_names is not None:
        return (
            alphas_table.scan()
            .filter(
                pl.col("date").is_between(start, end),
                pl.col("signal_name").is_in(signal_names),
            )
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )

    else:
        return (
            alphas_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )


def load_alphas_by_date(
    date_: dt.date, columns: list[str], signal_names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of alphas data for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the data frame.
    columns : list of str
        List of column names to include in the result.
    signal_names : list of str
        List of signal names to filter the data frame by.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing alphas data on the specified date,
        with only the selected columns.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> columns = ["barrid", "date", "signal_name", "alpha_value"]
    >>> df = sf.data.load_alphas_by_date(
    ...     date_=date_,
    ...     columns=columns,
    ...     signal_names=["momentum", "reversal"]
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
    if signal_names is not None:
        return (
            alphas_table.scan()
            .filter(
                pl.col("date").eq(date_),
                pl.col("signal_name").is_in(signal_names),
            )
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )
    else:
        return (
            alphas_table.scan(date_.year)
            .filter(pl.col("date").eq(date_))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )


def load_combined_alphas(
    start: dt.date, end: dt.date, columns: list[str]
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of combined alpha data between two dates.

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
        A DataFrame containing combined alpha data between the specified dates,
        with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> columns = ["barrid", "date", "combined_alpha_value"]
    >>> df = sfd.load_combined_alphas(
    ...     start=start,
    ...     end=end,
    ...     columns=columns
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌─────────┬────────────┬──────────────────────┐
    │ barrid  ┆ date       ┆ combined_alpha_value │
    │ ---     ┆ ---        ┆ ---                  │
    │ str     ┆ date       ┆ f64                  │
    ╞═════════╪════════════╪══════════════════════╡
    │ USA06Z1 ┆ 2024-01-03 ┆ -0.187421            │
    │ USA0A21 ┆ 2024-01-03 ┆ 0.052188             │
    │ USA1BC1 ┆ 2024-01-03 ┆ 0.391552             │
    │ USA2DF1 ┆ 2024-01-03 ┆ -0.024119            │
    │ USA3GH1 ┆ 2024-01-03 ┆ 0.145993             │
    └─────────┴────────────┴──────────────────────┘
    """
    return (
        combined_alphas_table.scan()
        .filter(pl.col("date").is_between(start, end))
        .sort(["barrid", "date"])
        .select(columns)
        .collect()
    )


def load_combined_alphas_by_date(
    date_: dt.date, columns: list[str]
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of combined alphas data for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the data frame.
    columns : list of str
        List of column names to include in the result.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing combined alphas data on the specified date,
        with only the selected columns.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> columns = ["barrid", "date", "combined_alpha_value"]
    >>> df = sf.data.load_combined_alphas_by_date(
    ...     date_=date_,
    ...     columns=columns
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌─────────┬────────────┬──────────────────────┐
    │ barrid  ┆ date       ┆ combined_alpha_value │
    │ ---     ┆ ---        ┆ ---                  │
    │ str     ┆ date       ┆ f64                  │
    ╞═════════╪════════════╪══════════════════════╡
    │ USA06Z1 ┆ 2024-01-03 ┆ -0.187421            │
    │ USA0A21 ┆ 2024-01-03 ┆ 0.052188             │
    │ USA1BC1 ┆ 2024-01-03 ┆ 0.391552             │
    │ USA2DF1 ┆ 2024-01-03 ┆ -0.024119            │
    │ USA3GH1 ┆ 2024-01-03 ┆ 0.145993             │
    └─────────┴────────────┴──────────────────────┘
    """
    return (
        combined_alphas_table.scan(date_.year)
        .filter(pl.col("date").eq(date_))
        .sort(["barrid", "date"])
        .select(columns)
        .collect()
    )