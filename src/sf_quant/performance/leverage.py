import polars as pl
from sf_quant.data.assets import load_assets
from sf_quant.data.benchmark import load_benchmark
from sf_quant.schema.leverage_schema import LeverageSchema
from sf_quant.schema.portfolio_schema import PortfolioSchema

def generate_leverage_from_weights(weights: PortfolioSchema) -> LeverageSchema:
    """
    Calculate leverage from portfolio weights.

    Parameters
    ----------
        weights (PortfolioSchema): Portfolio weights validated against PortfolioSchema.
            Must include the following columns:
            - ``date`` (date): The observation date.
            - ``barrid`` (str): Asset identifier.
            - ``weight`` (float): Position weight.

    Returns
    -------
        LeverageSchema: A validated DataFrame with columns:
            - ``date`` (date): The observation date.
            - ``leverage`` (float): Sum of absolute weights.
    
    Notes
    -----
        - Leverage is calculated as the sum of absolute weights per date.
        - Leverage = 1.0 means fully invested with no shorting.
        - Leverage > 1.0 indicates use of margin or shorting.

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
    >>> leverage = sfp.generate_leverage_from_weights(weights)
    >>> leverage
    shape: (2, 2)
    ┌────────────┬──────────┐
    │ date       ┆ leverage │
    │ ---        ┆ ---      │
    │ date       ┆ f64      │
    ╞════════════╪══════════╡
    │ 2024-01-02 ┆ 1.0      │
    │ 2024-01-03 ┆ 1.0      │
    └────────────┴──────────┘
    """
    result = (
        weights.group_by("date")
        .agg(
            pl.col("weight").abs().sum().round(2).alias("leverage")
        )
        .sort("date")
        .select("date", "leverage")
    )
    return LeverageSchema.validate(result)