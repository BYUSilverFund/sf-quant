import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns
import dataframely as dy
from sf_quant.schema.ic_schema import ICSchema
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


def generate_ic_chart(
    ics: ICSchema,
    title: str | None = None,
    ic_type: str | None = None,
    file_name: str | None = None,
    ) -> None:
    """
    Plot cumulative Information Coefficient (IC) over time.

    This function generates a line chart of the cumulative sum of IC values
    across dates. The cumulative IC provides insight into the persistent
    predictive power of alphas by showing whether ICs compound positively or
    negatively over time.

    Parameters
    ----------
        ics (dy.DataFrame[ICSchema]): A DataFrame containing IC values.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``ic`` (float): The IC value for that date.
        title (str | None, optional): The chart's main title. Defaults to
            ``'Cumulative Information Coefficient'`` if not provided.
        ic_type (str | None, optional): Type of IC to display (e.g., 'Pearson' or 'Rank').
            If not provided, defaults to 'Rank IC'.
        file_name (str | None, optional): If not ``None``, saves the chart to the given
            file path. Otherwise, the chart is displayed interactively.

    Returns
    -------
        None: Displays the cumulative IC time series chart using Matplotlib and Seaborn,
        or saves it to a file if ``file_name`` is specified.

    Notes
    -----
        - The cumulative IC is computed using a simple cumulative sum (no compounding).
        - A rising line indicates consistent positive ICs, while a declining line
          indicates persistent negative ICs.
        - Useful for assessing whether a signal’s predictive strength holds over time.
    """
    if title is None:
        title = 'Cumulative Information Coefficient'

    if not isinstance(ics, pl.DataFrame):
        ics = pl.from_pandas(ics.to_pandas())

    ics = (
        ics
        .sort("date")
        .with_columns(
            pl.col("ic").fill_null(0).cum_sum().alias("cumulative_ic")
        )
    )

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=ics, x='date', y='cumulative_ic')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title(title)
    plt.xlabel(None)

    if ic_type is not None:
        plt.ylabel(f'Cumulative {ic_type} IC')
    else:
        plt.ylabel('Cumulative Rank IC')

    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
