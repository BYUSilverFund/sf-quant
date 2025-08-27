from datetime import date

import polars as pl

from .views import crsp_monthly_clean, crsp_monthly_table


def load(start: date, end: date, columns: list[str]) -> pl.DataFrame:
    return (
        crsp_monthly_clean.filter(pl.col("date").is_between(start, end))
        .sort(["permno", "date"])
        .select(columns)
        .collect()
    )


def get_columns() -> str:
    return crsp_monthly_table.columns()
