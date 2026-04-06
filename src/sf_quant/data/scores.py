import datetime as dt
import polars as pl

from ._tables import scores_table


def load_scores(
    start: dt.date, end: dt.date, names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of scores data between two dates.

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
        A DataFrame containing scores data between the specified dates.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> df = sfd.load_scores(
    ...     start=start,
    ...     end=end,
    ...     names=["momentum"]
    ... )
    >>> df.head()
    shape: (5, 4)
    ┌────────────┬─────────┬─────────────┬───────────┐
    │ date       ┆ barrid  ┆ signal_name ┆ score     │
    │ ---        ┆ ---     ┆ ---         ┆ ---       │
    │ date       ┆ str     ┆ str         ┆ f64       │
    ╞════════════╪═════════╪═════════════╪═══════════╡
    │ 2022-01-03 ┆ USA3871 ┆ momentum    ┆ 0.324434  │
    │ 2022-01-03 ┆ USBDIJ1 ┆ momentum    ┆ 1.239903  │
    │ 2022-01-03 ┆ USA91R1 ┆ momentum    ┆ -0.019291 │
    │ 2022-01-03 ┆ USBFCZ1 ┆ momentum    ┆ 0.17957   │
    │ 2022-01-03 ┆ USAA181 ┆ momentum    ┆ 0.227943  │
    └────────────┴─────────┴─────────────┴───────────┘
    """
    if names is not None:
        return (
            scores_table.scan()
            .filter(
                pl.col("date").is_between(start, end),
                pl.col("signal_name").is_in(names),
            )
            .sort(["barrid", "date"])
            .collect()
        )

    else:
        return (
            scores_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .collect()
        )


def load_scores_by_date(
    date_: dt.date, names: list[str] = None
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of scores data for a single date.

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
        A DataFrame containing scores data on the specified date.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> df = sf.data.load_scores_by_date(
    ...     date_=date_,
    ...     names=["momentum"]
    ... )
    >>> df.head()
    shape: (5, 4)
    ┌────────────┬─────────┬─────────────┬───────────┐
    │ date       ┆ barrid  ┆ signal_name ┆ score     │
    │ ---        ┆ ---     ┆ ---         ┆ ---       │
    │ date       ┆ str     ┆ str         ┆ f64       │
    ╞════════════╪═════════╪═════════════╪═══════════╡
    │ 2022-01-03 ┆ USA3871 ┆ momentum    ┆ 0.324434  │
    │ 2022-01-03 ┆ USBDIJ1 ┆ momentum    ┆ 1.239903  │
    │ 2022-01-03 ┆ USA91R1 ┆ momentum    ┆ -0.019291 │
    │ 2022-01-03 ┆ USBFCZ1 ┆ momentum    ┆ 0.17957   │
    │ 2022-01-03 ┆ USAA181 ┆ momentum    ┆ 0.227943  │
    └────────────┴─────────┴─────────────┴───────────┘
    """
    if names is not None:
        return (
            scores_table.scan()
            .filter(
                pl.col("date").eq(date_),
                pl.col("signal_name").is_in(names),
            )
            .sort(["barrid", "date"])
            .collect()
        )
    else:
        return (
            scores_table.scan(date_.year)
            .filter(pl.col("date").eq(date_))
            .sort(["barrid", "date"])
            .collect()
        )


def get_score_names() -> list[str]:
    """
    Return the list of available score (signal) names.

    Returns
    -------
    list of str
        A list of unique signal names in the scores dataset.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_score_names()
    ["momentum", "reversal", ...]
    """
    return (
        scores_table.scan()
        .select("signal_name")
        .unique()
        .collect()["signal_name"]
        .to_list()
    )
