import datetime as dt
import polars as pl

from ._tables import covariances_table


def load_covariances_by_date(date_: dt.date) -> pl.DataFrame:
    """
    Load the Barra factor covariance matrix for a single date.

    Parameters
    ----------
    date_ : datetime.date
        Date of the covariance matrix.

    Returns
    -------
    polars.DataFrame
        A DataFrame with one row per factor and the factor covariances as
        columns. Use :func:`construct_covariance_matrix` to build an
        asset-level covariance matrix from it.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> df = sfd.load_covariances_by_date(dt.date(2024, 1, 3))
    """
    return (
        covariances_table.scan(date_.year)
        .filter(pl.col("date").eq(date_))
        .sort("date")
        .collect()
    )


def get_covariances_columns() -> str:
    """
    Return the available columns in the covariances dataset.

    Returns
    -------
    str
        A string representation of a polars data frame containing the
        column names and types for the covariances table.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_covariances_columns()
    """
    return covariances_table.columns()
