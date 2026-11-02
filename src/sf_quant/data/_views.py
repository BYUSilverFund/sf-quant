import datetime as dt
import polars as pl

from ._tables import (
    assets_table,
    crsp_daily_table,
    crsp_events_table,
    crsp_monthly_table,
    crsp_v2_monthly_table,
    crsp_v2_daily_table,
    ff_table,
)


def crsp_events_monthly():
    return (
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


def crsp_monthly_clean():
    return (
        crsp_monthly_table.scan()
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month_date"))
        .join(crsp_events_monthly(), on=["month_date", "permno"], how="left")
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


def crsp_v2_monthly_clean():
    return (
        crsp_v2_monthly_table.scan()
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month_date"))
        .join(crsp_events_monthly(), on=["month_date", "permno"], how="left")
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


def crsp_daily_clean():
    return (
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


def crsp_v2_daily_clean():
    return (
        crsp_v2_daily_table.scan()
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


def benchmark():
    return (
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


def benchmark_returns(start=None, end=None):
    """Each stock's benchmark weight at the previous close, with the day's return.

    A day's benchmark return weights each stock by its market cap at the previous
    close, over the stocks in the universe at that close. The day's own closing cap
    already includes the day's return, so weighting by it would overweight that
    day's winners after the fact and overstate the benchmark.

    ``start`` and ``end`` bound the scan; a two-week margin before ``start`` supplies
    each stock's previous close.
    """
    frame = assets_table.scan()
    if start is not None:
        frame = frame.filter(pl.col("date") >= start - dt.timedelta(days=14))
    if end is not None:
        frame = frame.filter(pl.col("date") <= end)
    return (
        frame.select("date", "barrid", "market_cap", "in_universe", "return")
        .sort("barrid", "date")
        .with_columns(
            pl.col("market_cap").shift(1).over("barrid").alias("previous_cap"),
            pl.col("in_universe").shift(1).over("barrid").alias("previous_in_universe"),
        )
        .filter(pl.col("previous_in_universe").fill_null(False), pl.col("previous_cap") > 0)
        .select(
            "date",
            "barrid",
            pl.col("previous_cap")
            .truediv(pl.col("previous_cap").sum())
            .over("date")
            .alias("weight"),
            "return",
        )
    )


def fama_french():
    return ff_table.scan()
