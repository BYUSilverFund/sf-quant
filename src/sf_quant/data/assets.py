import datetime as dt
import polars as pl

from ._tables import assets_table
from ._views import in_universe_assets


def load_assets(
    start: dt.date, end: dt.date, in_universe: bool, columns: list[str]
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of assets data between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.
    in_universe : bool
        If ``True``, restrict to assets that are in the universe.
        If ``False``, include all assets.
    columns : list of str
        List of column names to include in the result.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing asset data between the specified dates,
        filtered by universe membership if requested, with only the
        selected columns.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> columns = ["barrid", "date", "price"]
    >>> df = sfd.load_assets(
    ...     start=start,
    ...     end=end,
    ...     in_universe=True,
    ...     columns=columns
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌────────────┬─────────┬───────┐
    │ date       ┆ barrid  ┆ price │
    │ ---        ┆ ---     ┆ ---   │
    │ date       ┆ str     ┆ f64   │
    ╞════════════╪═════════╪═══════╡
    │ 2024-01-02 ┆ USA06Z1 ┆ 7.87  │
    │ 2024-01-03 ┆ USA06Z1 ┆ 7.775 │
    │ 2024-01-04 ┆ USA06Z1 ┆ 7.76  │
    │ 2024-01-05 ┆ USA06Z1 ┆ 7.8   │
    │ 2024-01-08 ┆ USA06Z1 ┆ 8.22  │
    └────────────┴─────────┴───────┘
    """
    if in_universe:
        return (
            in_universe_assets.filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )
    else:
        return (
            assets_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )


def load_assets_by_date(
    date_: dt.date, in_universe: bool, columns: list[str]
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of assets data for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the data frame.
    in_universe : bool
        If ``True``, restrict to assets that are in the universe.
        If ``False``, include all assets.
    columns : list of str
        List of column names to include in the result.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing asset data on the specified date,
        filtered by universe membership if requested, with only the
        selected columns.

    Examples
    --------
    >>> import sf_quant as sf
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> columns = ["barrid", "date", "price"]
    >>> df = sf.data.load_assets_by_date(
    ...     date_=date_,
    ...     in_universe=True,
    ...     columns=columns
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌────────────┬─────────┬───────┐
    │ date       ┆ barrid  ┆ price │
    │ ---        ┆ ---     ┆ ---   │
    │ date       ┆ str     ┆ f64   │
    ╞════════════╪═════════╪═══════╡
    │ 2024-01-03 ┆ USA06Z1 ┆ 7.775 │
    │ 2024-01-03 ┆ USA0771 ┆ 10.23 │
    │ 2024-01-03 ┆ USA0C11 ┆ 74.15 │
    │ 2024-01-03 ┆ USA0SY1 ┆ 130.1 │
    │ 2024-01-03 ┆ USA11I1 ┆ 43.55 │
    └────────────┴─────────┴───────┘
    """
    if in_universe:
        return (
            in_universe_assets.filter(pl.col("date").eq(date_))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )
    else:
        return (
            assets_table.scan(date_.year)
            .filter(pl.col("date").eq(date_))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )


def get_assets_columns() -> str:
    """
    Return the available columns in the assets dataset.

    This function provides a schema of all asset-level fields that can be
    retrieved with :func:`load_assets`. The output is a table listing each
    column name along with its corresponding data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the assets table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_assets_columns()
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
    return assets_table.columns()
