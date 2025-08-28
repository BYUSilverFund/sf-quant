import datetime as dt
import polars as pl

from ._views import crsp_daily_clean, crsp_daily_table


def load_crsp_daily(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    """
    Load a Polars DataFrame of CRSP daily data between two dates.

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
        A DataFrame containing CRSP daily data between the specified dates,
        sorted by ``permno`` and ``date``, with only the selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 1, 31)
    >>> columns = ["permno", "date", "ret"]
    >>> df = sfd.load_crsp_daily(
    ...     start=start,
    ...     end=end,
    ...     columns=columns
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌────────────┬────────┬───────┐
    │ date       ┆ permno ┆ ret   │
    │ ---        ┆ ---    ┆ ---   │
    │ date       ┆ i64    ┆ f64   │
    ╞════════════╪════════╪═══════╡
    │ 2024-01-02 ┆ 10001  ┆ 0.002 │
    │ 2024-01-03 ┆ 10001  ┆ -0.001│
    │ 2024-01-04 ┆ 10001  ┆ 0.003 │
    │ 2024-01-05 ┆ 10001  ┆ 0.000 │
    │ 2024-01-08 ┆ 10001  ┆ -0.004│
    └────────────┴────────┴───────┘
    """
    return (
        crsp_daily_clean.filter(pl.col("date").is_between(start, end))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def get_crsp_daily_columns() -> str:
    """
    Return the available columns in the CRSP daily dataset.

    This function provides a schema of all CRSP daily fields that can be
    retrieved with :func:`load_crsp_daily`. The output is a string representationPolars DataFrame
    listing each column name along with its corresponding data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the assets table.

    Examples
    --------
    >>> import sf_quant as sf
    >>> sf.data.get_assets_columns()
    shape: (30, 2)
    ┌──────────────────────┬─────────┐
    │ column               ┆ dtype   │
    │ ---                  ┆ ---     │
    │ str                  ┆ str     │
    ╞══════════════════════╪═════════╡
    │ date                 ┆ Date    │
    │ rootid               ┆ String  │
    │ barrid               ┆ String  │
    │ issuerid             ┆ String  │
    │ instrument           ┆ String  │
    │ ...                  ┆ ...     │
    └──────────────────────┴─────────┘
    """
    return crsp_daily_table.columns()
