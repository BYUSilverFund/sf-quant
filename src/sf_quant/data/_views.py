import polars as pl

from ._tables import (
    assets_table,
    crsp_daily_table,
    crsp_events_table,
    crsp_monthly_table,
)

russell_rebalance_dates = (
    assets_table.scan()
    # Standard filters
    .filter(pl.col("barrid").eq(pl.col("rootid")))
    .filter(pl.col("iso_country_code").eq("USA"))
    # Russell constituency filter
    .filter(pl.col("russell_1000") | pl.col("russell_2000"))
    # Create rebalance column
    .select("date", pl.lit(True).alias("russell_rebalance"))
    .unique()
)

in_universe_assets = (
    assets_table.scan()
    # Standard filters
    .filter(pl.col("barrid").eq(pl.col("rootid")))
    .filter(pl.col("iso_country_code").eq("USA"))
    # Join rebalance dates
    .join(russell_rebalance_dates, on="date", how="left")
    # Fill nulls with false on rebalance dates
    .with_columns(
        pl.when(pl.col("russell_rebalance")).then(
            pl.col("russell_1000", "russell_2000").fill_null(False)
        )
    )
    # Sort before forward fill
    .sort(["barrid", "date"])
    # Forward fill
    .with_columns(
        pl.col("ticker", "russell_1000", "russell_2000")
        .fill_null(strategy="forward")
        .over("barrid")
    )
    # Russell constituency filter
    .filter(pl.col("russell_1000") | pl.col("russell_2000"))
    # Drop russell_rebalance column
    .drop("russell_rebalance")
    .sort("barrid", "date")
    .with_columns(pl.col("return").shift(-1).over("barrid").alias("fwd_return"))
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


benchmark = in_universe_assets.select(
    "date",
    "barrid",
    pl.col("market_cap")
    .truediv(pl.col("market_cap").sum())
    .over("date")
    .alias("weight"),
)
