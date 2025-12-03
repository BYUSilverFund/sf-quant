import datetime as dt

import polars as pl

from ._views import fama_french
from ..schema import FamaFrenchSchema


def load_fama_french(start: dt.date, end: dt.date) -> pl.DataFrame:
    """
    Load Fama-French 5-factor data between two dates from the database.

    This function loads the Fama-French factors (mkt_rf, smb, hml, rmw, cma, rf)
    from the pre-configured database table.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive) of the data frame.
    end : datetime.date
        End date (inclusive) of the data frame.

    Returns
    -------
    polars.DataFrame
        A DataFrame containing Fama-French factor data between the specified dates,
        with columns: date, mkt_rf, smb, hml, rmw, cma, rf.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> ff_data = sfd.load_fama_french(start=start, end=end)
    >>> ff_data.head()
    shape: (5, 7)
    ┌────────────┬────────┬────────┬────────┬────────┬────────┬────────┐
    │ date       ┆ mkt_rf ┆ smb    ┆ hml    ┆ rmw    ┆ cma    ┆ rf     │
    │ ---        ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    │
    │ date       ┆ f64    ┆ f64    ┆ f64    ┆ f64    ┆ f64    ┆ f64    │
    ╞════════════╪════════╪════════╪════════╪════════╪════════╪════════╡
    │ 2024-01-02 ┆ 0.0123 ┆ 0.0045 ┆ -0.002 ┆ 0.0010 ┆ 0.0008 ┆ 0.0001 │
    └────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
    """
    result = fama_french.filter(pl.col("date").is_between(start, end)).collect()
    return FamaFrenchSchema.validate(result)


def get_fama_french_columns() -> str:
    """
    Return the available columns in the Fama-French dataset.

    Returns
    -------
    str
        A string representation of the column names and types.

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> sfd.get_fama_french_columns()
    shape: (7, 2)
    ┌────────┬─────────┐
    │ column ┆ dtype   │
    │ ---    ┆ ---     │
    │ date   ┆ Date    │
    │ Mkt-RF ┆ Float64 │
    │ SMB    ┆ Float64 │
    │ HML    ┆ Float64 │
    │ RMW    ┆ Float64 │
    │ CMA    ┆ Float64 │
    │ RF     ┆ Float64 │
    └────────┴─────────┘
    """
    pl.Config.set_tbl_rows(-1)
    schema = fama_french.collect_schema()
    df_str = str(
        pl.DataFrame(
            {
                "column": list(schema.keys()),
                "dtype": [str(t) for t in schema.values()],
            }
        )
    )
    pl.Config.set_tbl_rows(10)
    return df_str
