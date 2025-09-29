import datetime as dt
import polars as pl

from ._tables import factors_table


def load_factors(
    start: dt.date, end: dt.date
) -> pl.DataFrame:
    """
    Load a Polars DataFrame of factor returns(%) between two dates.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing factor returns(%) between the specified dates.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> df = sfd.load_factors(
    ...     start=start, 
    ...     end=end
    ... )
    >>> df.head()
    shape: (5, 78)
    ┌────────────┬─────────────────┬──────────────────┬──────────────────┬───┐
    │ date       ┆ USSLOWL_AERODEF ┆ USSLOWL_AIRLINES ┆ USSLOWL_ALUMSTEL ┆ … ┆
    │ ---        ┆ ---             ┆ ---              ┆ ---              ┆   ┆
    │ date       ┆ f64             ┆ f64              ┆ f64              ┆   ┆
    ╞════════════╪═════════════════╪══════════════════╪══════════════════╪═══╡
    │ 2024-01-02 ┆ -0.265309       ┆ -0.791378        ┆ 0.156503         ┆ … ┆
    │ 2024-01-03 ┆ -0.498421       ┆ -2.233705        ┆ 0.8307382        ┆ … ┆
    │ 2024-01-04 ┆ 0.086316        ┆ 2.365157         ┆ -0.328195        ┆ … ┆
    │ 2024-01-05 ┆ -0.209432       ┆ 2.549703         ┆ -0.18945         ┆ … ┆
    │ 2024-01-08 ┆ -1.414716       ┆ 0.6914087        ┆ -0.973735        ┆ … ┆
    └────────────┴─────────────────┴──────────────────┴──────────────────┴───┘
    """
    return (
        factors_table.scan()
        .filter(
            pl.col('date').is_between(start, end)
        )
        .sort('date')
        .collect()
    )

def get_factors_columns() -> str:
    """
    Return the available columns in the factors dataset.

    This function provides a schema of all factors that can be
    retrieved with :func:`load_factors`. The output is a table listing each
    column name along with its corresponding data type.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the factors table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_factors_columns()
    shape: (78, 2)
    ┌──────────────────┬─────────┐
    │ column           ┆ dtype   │
    │ ---              ┆ ---     │
    │ str              ┆ str     │
    ╞══════════════════╪═════════╡
    │ date             ┆ Date    │
    │ USSLOWL_AERODEF  ┆ Float64 │
    │ USSLOWL_AIRLINES ┆ Float64 │
    │ USSLOWL_ALUMSTEL ┆ Float64 │
    │ USSLOWL_APPAREL  ┆ Float64 │
    │ USSLOWL_AUTO     ┆ Float64 │
    │ ...              ┆ ...     │
    └──────────────────┴─────────┘
    """
    return factors_table.columns()
