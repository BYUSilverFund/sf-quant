import datetime as dt
import polars as pl

from ._tables import signals_table


def load_signals(
    start: dt.date, end: dt.date, names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of signals data between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.
    names : list of str, optional
        List of signal names to filter the data frame by.
        If None (default), all signal names are included.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing signals data between the specified dates.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> df = sfd.load_signals(
    ...     start=start,
    ...     end=end,
    ...     names=["momentum"]
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
    if names is not None:
        return (
            signals_table.scan()
            .filter(
                pl.col("date").is_between(start, end),
                pl.col("signal_name").is_in(names),
            )
            .sort(["barrid", "date"])
            .collect()
        )

    else:
        return (
            signals_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .collect()
        )


def load_signals_by_date(
    date_: dt.date, names: list[str] | None = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of signal data for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the data frame.
    names : list of str, optional
        List of signal names to filter the data frame by.
        If None (default), all signal names are included.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing signal data on the specified date.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> df = sf.data.load_signals_by_date(
    ...     date_=date_,
    ...     names=["momentum"]
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
    if names is not None:
        return (
            signals_table.scan()
            .filter(
                pl.col("date").eq(date_),
                pl.col("signal_name").is_in(names),
            )
            .sort(["barrid", "date"])
            .collect()
        )
    else:
        return (
            signals_table.scan(date_.year)
            .filter(pl.col("date").eq(date_))
            .sort(["barrid", "date"])
            .collect()
        )


def get_signal_names() -> list[str]:
    """
    Return the list of available signal names.

    Returns
    -------
    list of str
        A list of unique signal names in the signals dataset.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_signal_names()
    ["momentum", "reversal", ...]
    """
    return (
        signals_table.scan()
        .select("signal_name")
        .unique()
        .collect()["signal_name"]
        .to_list()
    )
