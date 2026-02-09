import polars as pl
import numpy as np
import matplotlib.pyplot as plt


def signal_stats(signal: pl.DataFrame, column: str = "signal") -> pl.DataFrame:
    """
    Compute statistical measures of a signal column.

    Parameters
    ----------
    signal : pl.DataFrame
        DataFrame containing the signal column
    column : str, default "signal"
        Name of the column to analyze

    Returns
    -------
    pl.DataFrame
        Single-row DataFrame with statistics: mean, std, min, max, q25, q50, q75

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.research as sfr
    >>> signal_df = pl.DataFrame({
    ...     'signal': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ... })
    >>> stats = sfr.signal_stats(signal_df, column='signal')
    >>> stats
    shape: (1, 7)
    ┌──────┬──────────┬─────┬─────┬──────┬──────┬──────┐
    │ mean ┆ std      ┆ min ┆ max ┆ q25  ┆ q50  ┆ q75  │
    │ ---  ┆ ---      ┆ --- ┆ --- ┆ ---  ┆ ---  ┆ ---  │
    │ f64  ┆ f64      ┆ f64 ┆ f64 ┆ f64  ┆ f64  ┆ f64  │
    ╞══════╪══════════╪═════╪═════╪══════╪══════╪══════╡
    │ 0.55 ┆ 0.302765 ┆ 0.1 ┆ 1.0 ┆ 0.325 ┆ 0.55 ┆ 0.775 │
    └──────┴──────────┴─────┴─────┴──────┴──────┴──────┘
    """
    return signal.select([
        pl.col(column).mean().alias("mean"),
        pl.col(column).std().alias("std"),
        pl.col(column).min().alias("min"),
        pl.col(column).max().alias("max"),
        pl.col(column).quantile(0.25).alias("q25"),
        pl.col(column).quantile(0.50).alias("q50"),
        pl.col(column).quantile(0.75).alias("q75"),
    ])


def signal_distribution(signal: pl.DataFrame, column: str = "signal") -> None:
    """
    Plot the distribution of a signal column as a histogram.

    Parameters
    ----------
    signal : pl.DataFrame
        DataFrame containing the signal column
    column : str, default "signal"
        Name of the column to plot

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.research as sfr
    >>> signal_df = pl.DataFrame({
    ...     'signal': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ... })
    >>> sfr.signal_distribution(signal_df, column='signal')
    # Displays a histogram plot of the signal values
    """
    signal_values = signal.select(column).to_numpy().flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(signal_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title("Signal Distribution")
    plt.xlabel("Signal Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
