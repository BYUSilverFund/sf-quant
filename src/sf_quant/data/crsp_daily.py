import datetime as dt
import polars as pl

from ._views import crsp_daily_clean, crsp_daily_table


def load(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    return (
        crsp_daily_clean.filter(pl.col("date").is_between(start, end))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def get_columns() -> str:
    return crsp_daily_table.columns()


__all__ = ["load", "get_columns"]
