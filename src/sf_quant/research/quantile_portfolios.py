import logging
import polars as pl
from ._schemas import SignalSchema
from sf_quant.data.benchmark import load_benchmark_returns

logger = logging.getLogger(__name__)

def check_nulls_in_data(signal: pl.DataFrame, check_cols: list[str] | None = None) -> dict:
    """
    Check for nulls in signal data columns.

    Parameters
    ----------
    signal : pl.DataFrame
        Data to check for nulls.
    check_cols : list[str], optional
        Specific columns to check. If None, checks 'signal' and 'return' columns.

    Returns
    -------
    dict
        Dictionary with null counts for each checked column.
    """
    if check_cols is None:
        check_cols = ['signal', 'return']

    null_counts = signal.select([
        pl.col(col).null_count().alias(f'{col}_nulls')
        for col in check_cols if col in signal.columns
    ]).row(0, named=True)

    # Log warnings for any nulls found
    for col, count in null_counts.items():
        if count > 0:
            logger.warning(f"Found {count} null values in {col}")

    return null_counts

def generate_quantile_ports(
    signal: SignalSchema,
    num_bins: int = 10
) -> pl.DataFrame:
    """
    Generate quantile portfolios from signal data with vol and beta scaling.

    This function bins assets into quantiles based on signal values, calculates
    equal-weighted returns for each quantile, computes long-short spread, and
    applies volatility and beta scaling transformations.

    Parameters
    ----------
    signal : SignalSchema
        Signal data containing at least:

        - ``date`` (date): The observation date.
        - ``signal`` (float): The signal value used for quantile binning.
        - ``return`` (float): Forward returns for each asset.

    num_bins : int, optional
        Number of quantile bins to create (default: 10).
        Assets are sorted into equal-sized bins based on signal values
        within each date.

    Returns
    -------
    pl.DataFrame
        A DataFrame with quantile portfolio returns (vol and beta scaled):

        - ``date`` (date): The observation date.
        - ``p_1``, ``p_2``, ..., ``p_{num_bins}`` (float): Equal-weighted
          returns for each quantile portfolio, vol-scaled and beta-scaled.
        - ``spread`` (float): Long-short spread (highest - lowest quantile),
          vol-scaled and beta-scaled.
        - ``bmk_return`` (float): Benchmark market returns.

    Notes
    -----
    - Quantiles are computed within each date using ``qcut``.
    - Benchmark returns are joined from ``load_benchmark_returns``.
    - Volatility scaling uses a 22-day rolling window with 5% target vol.
    - Beta scaling uses a 60-day rolling window with target beta of 1.0.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.research as sfr
    >>> import datetime as dt
    >>> signal_df = pl.DataFrame({
    ...     'date': [dt.date(2024, 1, 2)] * 100,
    ...     'signal': list(range(100)),
    ...     'return': [0.01] * 100
    ... })
    >>> quantile_ports = sfr.generate_quantile_ports(signal_df, num_bins=5)
    >>> quantile_ports.columns
    ['date', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'spread', 'bmk_return', ...]
    """

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

    market_returns = load_benchmark_returns(df['date'].min(), df['date'].max())
    df = df.join(market_returns, on="date", how="left")
    df = beta_scale_ports(df, market_col="bmk_return")
    df = vol_scale_ports(df)

    return df

def vol_scale_ports(df: pl.DataFrame, target_vol: float = 0.05, window: int = 22) -> pl.DataFrame:
    """
    Scale portfolio returns to a target volatility level in place.

    This function standardizes portfolio returns by scaling them to achieve
    a specified target volatility based on rolling historical volatility.
    The scaling factor is computed using a rolling window and annualized
    assuming 252 trading days per year. Original return columns are replaced
    with their scaled versions.

    Parameters
    ----------
    df : pl.DataFrame
        Portfolio returns data containing columns starting with ``p_``
        or named ``spread``. Additional columns are preserved.
    target_vol : float, optional
        Target annualized volatility level (default: 0.05 for 5%).
        Returns are scaled to achieve this volatility.
    window : int, optional
        Rolling window size in days for volatility calculation (default: 22).
        Typically set to one month of trading days.

    Returns
    -------
    pl.DataFrame
        The DataFrame with volatility-scaled return columns:

        - ``p_1``, ``p_2``, ..., ``p_{num_bins}`` (float): Vol-scaled returns.
        - ``spread`` (float): Vol-scaled long-short spread.

    Notes
    -----
    - Scaling factor = target_vol / (rolling_std * sqrt(252))
    - Only columns starting with ``p_`` or named ``spread`` are scaled.
    - Original return values are replaced with scaled versions.
    - Early rows (< window size) will have null scaled values.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.research as sfr
    >>> import datetime as dt
    >>> ports = pl.DataFrame({
    ...     'date': [dt.date(2024, 1, i) for i in range(1, 30)],
    ...     'p_1': [0.01] * 29,
    ...     'p_10': [0.02] * 29,
    ...     'spread': [0.01] * 29
    ... })
    >>> scaled = sfr.vol_scale_ports(ports, target_vol=0.10, window=20)
    >>> scaled.columns
    ['date', 'p_1', 'p_10', 'spread']
    """
    port_cols = [col for col in df.columns if col.startswith("p_") or col == "spread"]

    return df.with_columns([
        (
            pl.col(col) * (target_vol / (pl.col(col).rolling_std(window) * (252 ** 0.5)))
        ).alias(col)
        for col in port_cols
    ])

def beta_scale_ports(df: pl.DataFrame, market_col: str = "bmk_return", target_beta: float = 1.0, lookback: int = 60) -> pl.DataFrame:
    """
    Scale portfolio returns to a target beta relative to a market benchmark in place.

    This function computes rolling beta estimates for each portfolio relative
    to a market benchmark, then scales the portfolios to achieve a specified
    target beta. Beta is calculated using rolling covariance and variance over
    a specified lookback window. Original return columns are replaced with
    their beta-scaled versions.

    Parameters
    ----------
    df : pl.DataFrame
        Portfolio returns data containing:

        - Portfolio columns starting with ``p_`` or named ``spread``.
        - A market benchmark column specified by ``market_col``.

    market_col : str, optional
        Name of the column containing market benchmark returns
        (default: "bmk_return").
    target_beta : float, optional
        Target beta level for scaled portfolios (default: 1.0).
        A value of 1.0 means the portfolio moves in line with the market.
    lookback : int, optional
        Rolling window size in days for beta calculation (default: 60).
        Typically set to 2-3 months of trading days.

    Returns
    -------
    pl.DataFrame
        The DataFrame with beta-scaled return columns:

        - ``p_1``, ``p_2``, ..., ``p_{num_bins}`` (float): Beta-scaled returns.
        - ``spread`` (float): Beta-scaled long-short spread.

    Notes
    -----
    - Beta = rolling_cov(portfolio, market) / rolling_var(market)
    - Scaling factor = target_beta / estimated_beta
    - Only columns starting with ``p_`` or named ``spread`` are scaled.
    - Original return values are replaced with scaled versions.
    - Early rows (< lookback window) will have null values.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.research as sfr
    >>> import datetime as dt
    >>> ports = pl.DataFrame({
    ...     'date': [dt.date(2024, 1, i) for i in range(1, 100)],
    ...     'p_1': [0.01] * 99,
    ...     'p_10': [0.02] * 99,
    ...     'spread': [0.01] * 99,
    ...     'bmk_return': [0.015] * 99
    ... })
    >>> beta_scaled = sfr.beta_scale_ports(ports, target_beta=0.5, lookback=30)
    >>> beta_scaled.columns
    ['date', 'p_1', 'p_10', 'spread', 'bmk_return']
    """

    port_cols = [col for col in df.columns if col.startswith("p_")]
    bmk_var = pl.col(market_col).rolling_var(window_size=lookback)
    return df.with_columns([
        (
            pl.col(col) * (
                target_beta / (
                    pl.rolling_cov(pl.col(col), pl.col(market_col), window_size=lookback) /
                    bmk_var
                ).clip(0, 5.0).fill_nan(1.0)
            )
        ).alias(col)
        for col in port_cols
    ])