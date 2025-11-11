import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sf_quant.schema.leverage_schema import LeverageSchema
from sf_quant.schema.drawdown_schema import DrawdownSchema
from sf_quant.schema.returns_schema import PortfolioRetSchema, MultiPortfolioRetSchema


def generate_returns_chart(
    returns: PortfolioRetSchema,
    title: str,
    subtitle: str | None = None,
    log_scale: bool = False,
    file_name: str | None = None,
) -> None:
    """
    Plot cumulative portfolio returns over time.

    This function generates a line chart of cumulative returns.
    Returns are compounded over time and plotted in percentage terms.
    Optionally, returns can be displayed on a logarithmic scale.

    Parameters
    ----------
        returns (PortfolioRetSchema): Portfolio returns validated against PortfolioRetSchema.
            Must include the following columns:
            - ``date`` (date): The observation date.
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
        - Cumulative returns are computed as the compounded product of daily returns.
        - Returns are expressed in percentages for visualization.
        - If ``log_scale=True``, both daily and cumulative returns are transformed
          using the natural log (``log1p``).
    """
    returns_wide = (
        returns.sort("date")
        .with_columns(
            pl.col("return")
            .add(1)
            .cum_prod()
            .sub(1)
            .alias("cumulative_return")
        )
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

    sns.lineplot(returns_wide, x="date", y="cumulative_return")

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

def generate_multi_returns_chart(
    returns: MultiPortfolioRetSchema,
    title: str,
    subtitle: str | None = None,
    log_scale: bool = False,
    file_name: str | None = None,
) -> None:
    """
    Plot cumulative portfolio returns over time for multiple portfolios.

    This function generates a line chart of cumulative returns for multiple
    portfolios (total, benchmark, active). Returns are compounded over time
    and plotted in percentage terms. Optionally, returns can be displayed
    on a logarithmic scale.

    Parameters
    ----------
        returns (MultiPortfolioRetSchema): Portfolio returns validated against MultiPortfolioRetSchema.
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

def generate_leverage_chart(
    leverage: LeverageSchema,
    title: str,
    subtitle: str | None = None,
    file_name: str | None = None,
) -> None:
    """
    Plot portfolio leverage over time.

    This function generates a line chart of leverage.
    Leverage is expressed as the sum of absolute weights (e.g., 1.0 = fully invested).

    Parameters
    ----------
        leverage (LeverageSchema): Portfolio leverage validated against LeverageSchema.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``leverage`` (float): Daily portfolio leverage.
        title (str): The chart's main title.
        subtitle (str | None, optional): The chart's subtitle, shown beneath the
            main title. Defaults to ``None``.
        file_name (str, optional): If not ``None``, the plot is saved to the given
            file path. Defaults to just displaying the chart.

    Returns
    -------
        None: Displays the leverage chart using Matplotlib and Seaborn.

    Notes
    -----
        - Leverage = 1.0 indicates fully invested with no leverage or shorting.
        - Leverage > 1.0 indicates use of margin or shorting.
        - Leverage is expressed as a ratio (not percentage).
    """
    leverage_wide = leverage.sort("date")

    plt.figure(figsize=(10, 6))

    sns.lineplot(leverage_wide, x="date", y="leverage")

    plt.suptitle(title)
    plt.title(subtitle)

    plt.xlabel(None)
    plt.ylabel("Leverage")

    plt.grid()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

def generate_drawdown_chart(
    drawdowns: DrawdownSchema,
    title: str,
    subtitle: str | None = None,
    file_name: str | None = None,
) -> None:
    """
    Plot portfolio drawdowns over time.

    This function generates a line chart of drawdowns.
    Drawdowns are expressed as negative percentages from the peak value.

    Parameters
    ----------
        drawdowns (DrawdownSchema): Portfolio drawdowns validated against DrawdownSchema.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``drawdown`` (float): Daily portfolio drawdown.
        title (str): The chart's main title.
        subtitle (str | None, optional): The chart's subtitle, shown beneath the
            main title. Defaults to ``None``.
        file_name (str, optional): If not ``None``, the plot is saved to the given
            file path. Defaults to just displaying the chart.

    Returns
    -------
        None: Displays the drawdown chart using Matplotlib and Seaborn.

    Notes
    -----
        - Drawdowns are expressed in percentages for visualization.
    """
    drawdowns_wide = (
        drawdowns.sort("date")
        .with_columns(pl.col("drawdown").mul(100))
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(drawdowns_wide, x="date", y="drawdown")

    plt.suptitle(title)
    plt.title(subtitle)

    plt.xlabel(None)
    plt.ylabel("Drawdown (%)")

    plt.grid()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()