import datetime as dt
import polars as pl

from ._tables import exposures_table
from ._views import in_universe_assets

def load_exposures(start: dt.date, end: dt.date, in_universe: bool, columns: list) -> pl.DataFrame:
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
    >>> columns = [
    ...     "date",
    ...     "barrid",
    ...     "USSLOWL_MOMENTUM",
    ...     "USSLOWL_VALUE"
    ... ]
    >>> df = sfd.load_exposures(
    ...     start=start, 
    ...     end=end, 
    ...     in_universe=True,
    ...     columns=columns
    ... )
    >>> df.head()
    shape: (5, 4)
    ┌────────────┬─────────┬──────────────────┬───────────────┐
    │ date       ┆ barrid  ┆ USSLOWL_MOMENTUM ┆ USSLOWL_VALUE │
    │ ---        ┆ ---     ┆ ---              ┆ ---           │
    │ date       ┆ str     ┆ f64              ┆ f64           │
    ╞════════════╪═════════╪══════════════════╪═══════════════╡
    │ 2024-01-02 ┆ USA06Z1 ┆ 1.39             ┆ -0.724        │
    │ 2024-01-03 ┆ USA06Z1 ┆ 1.417            ┆ -0.718        │
    │ 2024-01-04 ┆ USA06Z1 ┆ 1.308            ┆ -0.709        │
    │ 2024-01-05 ┆ USA06Z1 ┆ 1.359            ┆ -0.71         │
    │ 2024-01-08 ┆ USA06Z1 ┆ 1.36             ┆ -0.706        │
    └────────────┴─────────┴──────────────────┴───────────────┘
    """
    if in_universe:
        return (
            in_universe_assets
            .select('date', 'barrid')
            .join(
                other=exposures_table.scan(),
                on=['date', 'barrid'],
                how='left'
            )
            .filter(pl.col("date").is_between(start, end))
            .select(columns)
            .sort("barrid", "date")
            .collect()       
        )

    else:
        return (
            exposures_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .select('date', 'barrid', *columns)
            .sort("barrid", "date")
            .collect()
        )


def load_exposures_by_date(date_: dt.date) -> pl.DataFrame:
    return (
        exposures_table.scan(date_.year)
        .filter(pl.col("date").eq(date_))
        .sort("barrid", "date")
        .collect()
    )


def get_exposures_columns() -> str:
    """
    Return the available columns in the exposures dataset.

    This function provides a schema of all barra factor exposures that can be
    retrieved with :func:`load_exposures`. The output is a table listing each
    column name along with its corresponding data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the assets table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_exposures_columns()
    shape: (79, 2)
    ┌──────────────────┬─────────┐
    │ column           ┆ dtype   │
    │ ---              ┆ ---     │
    │ str              ┆ str     │
    ╞══════════════════╪═════════╡
    │ date             ┆ Date    │
    │ barrid           ┆ String  │
    │ USSLOWL_AERODEF  ┆ Float64 │
    │ USSLOWL_AIRLINES ┆ Float64 │
    │ USSLOWL_ALUMSTEL ┆ Float64 │
    │ USSLOWL_APPAREL  ┆ Float64 │
    │ USSLOWL_AUTO     ┆ Float64 │
    │ ...              ┆ ...     │
    └──────────────────┴─────────┘
    """
    return exposures_table.columns()
