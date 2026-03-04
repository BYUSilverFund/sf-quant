import datetime as dt
import polars as pl

from ._tables import signals_table


def load_signals(
    start: dt.date, end: dt.date, columns: list[str], signal_names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of signals data between two dates.

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
        A DataFrame containing signals data between the specified dates,
        with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> columns = ["barrid", "date", "signal_name", "signal_value"]
    >>> df = sfd.load_signals(
    ...     start=start,
    ...     end=end,
    ...     columns=columns,
    ...     signal_names=["momentum"]
    ... )
    >>> df.head()
    shape: (5, 4)
    ┌────────────┬─────────┬─────────────┬──────────────┐
    │ date       ┆ barrid  ┆ signal_name ┆ signal_value │
    │ ---        ┆ ---     ┆ ---         ┆ ---          │
    │ date       ┆ str     ┆ str         ┆ f64          │
    ╞════════════╪═════════╪═════════════╪══════════════╡
    │ 2022-01-03 ┆ USA3871 ┆ momentum    ┆ 0.226603     │
    │ 2022-01-03 ┆ USBDIJ1 ┆ momentum    ┆ 0.63502      │
    │ 2022-01-03 ┆ USA91R1 ┆ momentum    ┆ 0.073258     │
    │ 2022-01-03 ┆ USBFCZ1 ┆ momentum    ┆ 0.161975     │
    │ 2022-01-03 ┆ USAA181 ┆ momentum    ┆ 0.183556     │
    └────────────┴─────────┴─────────────┴──────────────┘
    """
    if signal_names is not None:
        return (
            signals_table.scan()
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
            signals_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )


def load_signals_by_date(
    date_: dt.date, columns: list[str], signal_names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of signal data for a single date.

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
        A DataFrame containing signal data on the specified date,
        with only the selected columns.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> columns = ["barrid", "date", "signal_name", "signal_value"]
    >>> df = sf.data.load_signals_by_date(
    ...     date_=date_,
    ...     columns=columns,
    ...     signal_names=["momentum"]
    ... )
    >>> df.head()
    shape: (5, 4)
    ┌────────────┬─────────┬─────────────┬──────────────┐
    │ date       ┆ barrid  ┆ signal_name ┆ signal_value │
    │ ---        ┆ ---     ┆ ---         ┆ ---          │
    │ date       ┆ str     ┆ str         ┆ f64          │
    ╞════════════╪═════════╪═════════════╪══════════════╡
    │ 2022-01-03 ┆ USA3871 ┆ momentum    ┆ 0.226603     │
    │ 2022-01-03 ┆ USBDIJ1 ┆ momentum    ┆ 0.63502      │
    │ 2022-01-03 ┆ USA91R1 ┆ momentum    ┆ 0.073258     │
    │ 2022-01-03 ┆ USBFCZ1 ┆ momentum    ┆ 0.161975     │
    │ 2022-01-03 ┆ USAA181 ┆ momentum    ┆ 0.183556     │
    └────────────┴─────────┴─────────────┴──────────────┘
    """
    if signal_names is not None:
        return (
            signals_table.scan()
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
            signals_table.scan(date_.year)
            .filter(pl.col("date").eq(date_))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )
