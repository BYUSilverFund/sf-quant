from datetime import date

import polars as pl

from .tables import assets_table
from .views import in_universe_assets

def load(start: date, end: date, in_universe: bool, columns: list[str]) -> pl.DataFrame:
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
