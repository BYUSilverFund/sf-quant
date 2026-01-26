import polars as pl
from ._schemas import SignalSchema
from sf_quant.data.benchmark import load_benchmark

def generate_quantile_ports(signal: SignalSchema, num_bins: int = 10) -> pl.DataFrame:
    return (
        signal
            .with_columns(
                pl.col('signal').qcut(num_bins, labels=[f"p_{i}" for i in range(1, num_bins + 1)]).alias('bin').over('date')
            )
            .group_by(["date", "bin"])
            .agg(pl.col("fwd_return").mean().alias("ew_return"))
            .sort(["date", "bin"])
            .pivot(
                index="date",
                on="bin", 
                values="ew_return"
            ).with_columns(
                (pl.col(f"p_{num_bins}") - pl.col("p_1")).alias("spread")
            )
    )

def vol_scale_ports(quantile_ports: pl.DataFrame, target_vol: float = 0.05, window: int = 22) -> pl.DataFrame:
    port_cols = [col for col in quantile_ports.columns if col.startswith("p_") or col == "spread"]

    return quantile_ports.with_columns([
        (pl.col(col) * target_vol / (pl.col(col).rolling_std(window) * (252 ** 0.5))).alias(f"{col}_vol_scaled")
        for col in port_cols
    ])

def beta_scale_ports(quantile_ports: pl.DataFrame, lookback: int = 60) -> pl.DataFrame:
    """
    Calculates a rolling beta and the resulting scale factor 
    to hit a target beta of 1.0.
    """
    port_cols = [col for col in quantile_ports.columns if col.startswith("p_") or col == "spread"]
    return quantile_ports.with_columns([
        (
            pl.rolling_cov(pl.col("portfolio_return"), pl.col("market_return"), window_size=lookback) / 
            pl.col("market_return").rolling_var(window_size=lookback)
        ).alias("current_beta")
    ]).with_columns([
        (1.0 / pl.col("current_beta")).fill_nan(None).alias("beta_scale_factor")
    ])


def generate_quantile_ports(
    signal: pl.DataFrame, 
    num_bins: int = 10
) -> pl.DataFrame:
    
    # Base Quantile Logic
    df = (
        signal
        .with_columns(
            pl.col('signal').qcut(num_bins, labels=[f"p_{i}" for i in range(1, num_bins + 1)]).alias('bin').over('date')
        )
        .group_by(["date", "bin"])
        .agg(pl.col("return").mean().alias("ew_return"))
        .sort(["date", "bin"])
        .pivot(index="date", on="bin", values="ew_return")
        .with_columns((pl.col(f"p_{num_bins}") - pl.col("p_1")).alias("spread"))
    )

    # load market returns
    market_returns = load_benchmark
    df = df.join(market_returns, on="date", how="left")

    # Conditional Scaling
    if vol_scale:
        df = vol_scale_ports(df)
        
    if beta_scale:
        # Assumes market_returns DF has a column named 'market_return'
        df = beta_scale_ports(df, market_col="market_return")

    return df

import polars as pl

def vol_scale_ports(df: pl.DataFrame, target_vol: float = 0.05, window: int = 22) -> pl.DataFrame:
    # Identify which columns to scale (p_1, p_2... and spread)
    port_cols = [col for col in df.columns if col.startswith("p_") or col == "spread"]
    
    return df.with_columns([
        (
            pl.col(col) * (target_vol / (pl.col(col).rolling_std(window) * (252 ** 0.5)))
        ).alias(f"{col}_vol_scaled")
        for col in port_cols
    ])

def beta_scale_ports(df: pl.DataFrame, market_col: str = "market_return", target_beta: float = 1.0, lookback: int = 60) -> pl.DataFrame:
    """
    Scales portfolio columns to a target beta relative to a market column.
    """
    port_cols = [col for col in df.columns if col.startswith("p_") or col == "spread"]
    
    return df.with_columns([
        # Current Beta = Cov(port, market) / Var(market)
        (
            pl.rolling_cov(pl.col(col), pl.col(market_col), window_size=lookback) / 
            pl.col(market_col).rolling_var(window_size=lookback)
        ).alias(f"{col}_beta")
        for col in port_cols
    ]).with_columns([
        # Scale = Target / Current
        (pl.col(col) * (target_beta / pl.col(f"{col}_beta"))).alias(f"{col}_beta_scaled")
        for col in port_cols
    ])