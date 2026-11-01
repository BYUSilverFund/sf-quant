import polars as pl
from sf_quant.schema.drawdown_schema import DrawdownSchema
from sf_quant.schema.returns_schema import PortfolioRetSchema

def generate_drawdown_from_returns(returns: PortfolioRetSchema) -> DrawdownSchema:
    """
    Calculate drawdowns for each portfolio over time.

    Parameters
    ----------
        returns (PortfolioRetSchema): Portfolio returns validated against PortfolioRetSchema.
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
        - Calculated for a single portfolio time series

        
    Examples
    --------
    >>> import datetime as dt
    >>> import polars as pl
    >>> import sf_quant.performance as sfp
    >>> from sf_quant.simulation import simulate_portfolio
    >>> dates = [dt.date(2024, 1, d) for d in (2, 3, 4)]
    >>> targets = pl.DataFrame({
    ...     "date": [dates[0], dates[0]], "barrid": ["A", "B"],
    ...     "weight": [0.5, 0.5],
    ... })
    >>> asset_returns = pl.DataFrame({
    ...     "date": [d for d in dates for _ in range(2)],
    ...     "barrid": ["A", "B"] * 3,
    ...     "return": [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],
    ... })
    >>> result = simulate_portfolio(targets, asset_returns)
    >>> returns = result.returns("total")
    >>> drawdowns = sfp.generate_drawdown_from_returns(returns)
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