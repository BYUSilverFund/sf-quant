import polars as pl

from ._tables import (
    assets_table,
    crsp_daily_table,
    crsp_events_table,
    crsp_monthly_table,
)

crsp_events_monthly = (
    crsp_events_table.scan()
    .select(
        pl.col("date").dt.strftime("%Y-%m").alias("month_date"),
        "permno",
        "ticker",
        "shrcd",
        "exchcd",
    )
    .group_by(["month_date", "permno"])
    .agg(pl.col("ticker").last(), pl.col("shrcd").last(), pl.col("exchcd").last())
)

crsp_monthly_clean = (
    crsp_monthly_table.scan()
    .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month_date"))
    .join(crsp_events_monthly, on=["month_date", "permno"], how="left")
    .sort(["permno", "date"])
    .with_columns(
        pl.col("ticker").fill_null(strategy="forward").over("permno"),
        pl.col("shrcd").fill_null(strategy="forward").over("permno"),
        pl.col("exchcd").fill_null(strategy="forward").over("permno"),
    )
    .filter(
        pl.col("shrcd").is_in([10, 11, None]), pl.col("exchcd").is_in([1, 2, 3, None])
    )
    .with_columns(pl.col("prc").abs())
    .filter(~pl.col("ret").is_in([-66.0, -77.0, -88.0, -99.0]))
    .sort(["permno", "date"])
)

crsp_daily_clean = (
    crsp_daily_table.scan()
    .join(crsp_events_table.scan(), on=["date", "permno"], how="left")
    .sort(["permno", "date"])
    .with_columns(
        pl.col("ticker").fill_null(strategy="forward").over("permno"),
        pl.col("shrcd").fill_null(strategy="forward").over("permno"),
        pl.col("exchcd").fill_null(strategy="forward").over("permno"),
    )
    .filter(pl.col("shrcd").is_in([10, 11]), pl.col("exchcd").is_in([1, 2, 3]))
    .with_columns(pl.col("prc").abs())
    .filter(~pl.col("ret").is_in([-66.0, -77.0, -88.0, -99.0]))
    .sort(["permno", "date"])
)


benchmark = (
    assets_table.scan()
    .filter(pl.col('in_universe'))
    .select(
        "date",
        "barrid",
        pl.col("market_cap")
        .truediv(pl.col("market_cap").sum())
        .over("date")
        .alias("weight"),
    )
)