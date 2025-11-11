import polars as pl
from sf_quant.schema.drawdown_schema import DrawdownSchema
from sf_quant.schema.returns_schema import PortfolioRetSchema

def generate_drawdown_from_returns(returns: PortfolioRetSchema) -> DrawdownSchema:
    """
    Calculate drawdowns for each portfolio over time.

    Parameters
    ----------
        returns (PortfolioRetSchema): Portfolio returns validated against PortfolioReturnsSchema.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``return`` (float): Daily portfolio return.

    Returns
    -------
        DrawdownSchema: A validated DataFrame with columns:
            - ``date`` (date): The observation date.
            - ``drawdown`` (float): Drawdown from peak (≤ 0).
    
    Notes
    -----
        - Drawdown is calculated as (current_value / peak_value) - 1
        - Values are ≤ 0, where 0 indicates a new peak
        - Calculated per portfolio using grouped operations

        
    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.performance as sfp
    >>> import datetime as dt
    >>> weights = pl.DataFrame(
    ...     {
    ...         'date': [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 3)],
    ...         'barrid': ['USA06Z1', 'USA0771', 'USA06Z1', 'USA0771'],
    ...         'weight': [0.5, 0.5, 0.3, 0.7]
    ...     }
    ... )
    >>> returns = sfp.generate_returns_from_weights(weights)
    >>> drawdowns = sfp.generate_drawdown_from_returns(returns)
    >>> drawdowns
    shape: (2, 2)
    ┌────────────┬───────────┐
    │ date       ┆ drawdown  │
    │ ---        ┆ ---       │
    │ date       ┆ f64       │
    ╞════════════╪═══════════╡
    │ 2024-01-02 ┆ 0.0       │
    │ 2024-01-03 ┆ -0.016154 │
    └────────────┴───────────┘
    """
    result = (
        returns.sort("date")
        .with_columns(
            pl.col("return")
            .add(1)
            .cum_prod()
            .alias("portfolio_value")
        )
        .with_columns(
            pl.col("portfolio_value")
            .cum_max()
            .alias("peak")
        )
        .with_columns(
            (pl.col("portfolio_value") / pl.col("peak") - 1)
            .alias("drawdown")
        )
        .select("date", "drawdown")
    )
    return DrawdownSchema.validate(result)