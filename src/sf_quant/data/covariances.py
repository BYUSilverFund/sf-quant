import datetime as dt
import polars as pl

from ._tables import covariances_table


def load_covariances_by_date(date_: dt.date) -> pl.DataFrame:
    return (
        covariances_table.scan(date_.year)
        .filter(pl.col("date").eq(date_))
        .sort("date")
        .collect()
    )


def get_covariances_columns() -> str:
    return covariances_table.columns()
