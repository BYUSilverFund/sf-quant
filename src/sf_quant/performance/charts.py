import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns


def generate_returns_chart(
    returns: pl.DataFrame,
    title: str,
    subtitle: str | None = None,
    log_scale: bool = False,
    file_name: str | None = None,
) -> None:
    """
    Plot cumulative portfolio returns over time.

    This function generates a line chart of cumulative returns for multiple
    portfolios. Returns are compounded over time and plotted in percentage terms.
    Optionally, returns can be displayed on a logarithmic scale.

    Parameters
    ----------
        returns (pl.DataFrame): A Polars DataFrame containing portfolio returns.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``portfolio`` (str): Portfolio name or identifier.
            - ``return`` (float): Daily portfolio return.
        title (str): The chart's main title.
        subtitle (str | None, optional): The chart's subtitle, shown beneath the
            main title. Defaults to ``None``.
        log_scale (bool, optional): If ``True``, plot cumulative log returns instead
            of cumulative returns. Defaults to ``False``.
        file_name (str, optional): If not ``None``, the plot is saved to the given
            file path. Defaults to just displaying the chart.

    Returns
    -------
        None: Displays the cumulative returns chart using Matplotlib and Seaborn.

    Notes
    -----
        - Cumulative returns are computed as the compounded product of daily
          returns for each portfolio.
        - Returns are expressed in percentages for visualization.
        - If ``log_scale=True``, both daily and cumulative returns are transformed
          using the natural log (``log1p``).
    """
    returns_wide = (
        returns.sort("date", "portfolio")
        .with_columns(
            pl.col("return")
            .add(1)
            .cum_prod()
            .sub(1)
            .over("portfolio")
            .alias("cumulative_return")
        )
        .with_columns(pl.col("portfolio").str.to_titlecase().alias("label"))
    )

    if log_scale:
        returns_wide = returns_wide.with_columns(
            pl.col("return", "cumulative_return").log1p()
        )

    # Put into percent space
    returns_wide = returns_wide.with_columns(
        pl.col("return", "cumulative_return").mul(100)
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(returns_wide, x="date", y="cumulative_return", hue="label")

    plt.suptitle(title)
    plt.title(subtitle)

    plt.xlabel(None)

    if log_scale:
        plt.ylabel("Cumulative Log Returns (%)")
    else:
        plt.ylabel("Cumulative Returns (%)")

    plt.grid()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
