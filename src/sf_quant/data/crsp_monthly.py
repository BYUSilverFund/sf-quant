import datetime as dt
import polars as pl

from ._views import crsp_monthly_clean, crsp_monthly_table


def load(start: dt.date, end: dt.date, columns: list[str]) -> pl.DataFrame:
    return (
        crsp_monthly_clean.filter(pl.col("date").is_between(start, end))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def get_columns() -> str:
    return crsp_monthly_table.columns()


__all__ = ["load", "get_columns"]
