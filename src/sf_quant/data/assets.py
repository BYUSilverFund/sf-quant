import datetime as dt
import polars as pl

from ._tables import assets_table
from ._views import in_universe_assets


def load(
    start: dt.date, end: dt.date, in_universe: bool, columns: list[str]
) -> pl.DataFrame:
    if in_universe:
        return (
            in_universe_assets.filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )
    else:
        return (
            assets_table.scan()
            .filter(pl.col("date").is_between(start, end))
            .sort(["barrid", "date"])
            .select(columns)
            .collect()
        )


def get_columns() -> str:
    return assets_table.columns()


__all__ = ["load", "get_columns"]
