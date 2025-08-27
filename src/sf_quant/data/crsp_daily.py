from datetime import date

import polars as pl

from .views import crsp_daily_clean, crsp_daily_table


def load(start: date, end: date, columns: list[str]) -> pl.DataFrame:
    return (
        crsp_daily_clean.filter(pl.col("date").is_between(start, end))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def get_columns() -> str:
    return crsp_daily_table.columns()
