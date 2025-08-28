import datetime as dt
import polars as pl

from ._views import benchmark


def load_benchmark(
    start: dt.date, end: dt.date
) -> pl.DataFrame:
    """
    Load benchmark data between two dates.

    This function queries the benchmark view for data within the given
    date range, sorts the result by ``barrid`` and ``date``, and returns
    it as a Polars DataFrame.

    Parameters
    ----------
    start : datetime.date
        Start date (inclusive).
    end : datetime.date
        End date (inclusive).

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing benchmark data with at least
        the following columns:

        - ``barrid`` : str, asset identifier for the benchmark.
        - ``date`` : datetime, observation date.
        - ``weight``: float, benchmark weight (market cap weighted).

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> start = dt.date(2024, 1, 1)
    >>> end = dt.date(2024, 12, 31)
    >>> df = sfd.load_benchmark(
    ...     start=start,
    ...     end=end
    ... )
    >>> df.head()
    shape: (5, 3)
    ┌────────────┬─────────┬──────────┐
    │ date       ┆ barrid  ┆ weight   │
    │ ---        ┆ ---     ┆ ---      │
    │ date       ┆ str     ┆ f64      │
    ╞════════════╪═════════╪══════════╡
    │ 2024-01-02 ┆ USA06Z1 ┆ 0.000019 │
    │ 2024-01-03 ┆ USA06Z1 ┆ 0.000019 │
    │ 2024-01-04 ┆ USA06Z1 ┆ 0.000019 │
    │ 2024-01-05 ┆ USA06Z1 ┆ 0.000019 │
    │ 2024-01-08 ┆ USA06Z1 ┆ 0.00002  │
    └────────────┴─────────┴──────────┘
    """
    return (
        benchmark.filter(pl.col("date").is_between(start, end))
        .sort(["barrid", "date"])
        .collect()
    )