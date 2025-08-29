import datetime as dt
import polars as pl

from ._tables import exposures_table


def load_exposures_by_date(date_: dt.date) -> pl.DataFrame:
    return (
        exposures_table.scan(date_.year)
        .filter(pl.col("date").eq(date_))
        .sort("barrid", "date")
        .collect()
    )


def get_exposures_columns() -> str:
    return exposures_table.columns()
