import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns
import dataframely as dy
from sf_quant.schema import ICSchema


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


def generate_ic_chart(ics: dy.DataFrame[ICSchema],
    title: str | None = None,
    ic_type: str | None = None,
    file_name: str | None = None,
    ) -> None:
    """
    Plot Information Coefficient (IC) over time.

    This function generates a line chart of IC values across dates. The IC measures
    the correlation between a predictive signal (alpha) and subsequent realized returns,
    either in raw (Pearson) or rank (Spearman) form. A horizontal line at zero is added
    to indicate the baseline.

    Parameters
    ----------
        ics (dy.DataFrame[ICSchema]): A DataFrame containing IC values.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``ic`` (float): The IC value for that date.
        title (str | None, optional): The chart's main title. Defaults to
            ``'Information Coefficient Over Time'`` if not provided.
        ic_type (str | None, optional): Type of IC to display (e.g., 'Pearson' or 'Rank').
            If not provided, defaults to 'Rank IC'.
        file_name (str | None, optional): If not ``None``, saves the chart to the given
            file path. Otherwise, the chart is displayed interactively.

    Returns
    -------
        None: Displays the IC time series chart using Matplotlib and Seaborn,
        or saves it to a file if ``file_name`` is specified.

    Notes
    -----
        - The chart includes a dashed horizontal line at y=0 to indicate the
          neutral correlation level.
        - IC values are plotted directly; no cumulative or compounding calculation
          is applied.
        - Useful for visualizing the predictive power of alphas over time.
    """
    if title is None:
        title = 'Information Coefficient Over Time'

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=ics, x='date', y='ic')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title(title)
    plt.xlabel(None)

    if ic_type is not None:
        plt.ylabel(f'{ic_type} IC')
    else:
        plt.ylabel('Rank IC')

    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()