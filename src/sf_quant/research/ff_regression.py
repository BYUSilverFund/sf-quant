import polars as pl
import polars_ols
from sf_quant.data import load_fama_french


def run_ff_regression(
    portfolio_returns: pl.DataFrame,
) -> pl.DataFrame:
    """
    Run Fama-French 5-factor regressions on portfolio returns.

    This function performs ordinary least squares regression of portfolio excess
    returns against the Fama-French 5 factors (market, size, value, profitability,
    and investment). It computes alphas, betas, t-statistics, and other regression
    statistics for each portfolio.

    Parameters
    ----------
    portfolio_returns : pl.DataFrame
        Portfolio returns data containing:

        - ``date`` (date): The observation date.
        - Portfolio return columns (e.g., ``p_1``, ``p_2``, ..., ``spread``).

    Returns
    -------
    pl.DataFrame
        Regression results with rows as statistics and columns as portfolios:

        - ``feature_names`` (str): Statistic name (alpha, alpha_t, beta_mkt,
          mkt_t, beta_smb, smb_t, beta_hml, hml_t, beta_rmw, rmw_t,
          beta_cma, cma_t).
        - Portfolio columns (e.g., ``p_1``, ``p_2``, ..., ``spread``): Values
          for each coefficient or t-statistic.

    Notes
    -----
    - Fama-French factors are automatically loaded based on date range in portfolio_returns.
    - Portfolio excess returns are computed as portfolio return minus risk-free rate.
    - Regressions use OLS with intercept.
    - Results are transposed so rows are statistics and columns are portfolios.

    Examples
    --------
    >>> import polars as pl
    >>> import sf_quant.research as sfr
    >>> import datetime as dt
    >>> ports = pl.DataFrame({
    ...     'date': [dt.date(2024, 1, i) for i in range(1, 100)],
    ...     'p_1': [0.01] * 99,
    ...     'p_10': [0.02] * 99,
    ...     'spread': [0.01] * 99
    ... })
    >>> results = sfr.run_ff_regression(ports)
    >>> results
    shape: (12, 4)
    ┌───────────────┬──────────┬──────────┬──────────┐
    │ feature_names ┆ p_1      ┆ p_10     ┆ spread   │
    │ ---           ┆ ---      ┆ ---      ┆ ---      │
    │ str           ┆ f64      ┆ f64      ┆ f64      │
    ╞═══════════════╪══════════╪══════════╪══════════╡
    │ alpha         ┆ 0.0023   ┆ 0.0045   ┆ 0.0078   │
    │ alpha_t       ┆ 2.134    ┆ 3.421    ┆ 4.532    │
    │ beta_mkt      ┆ 1.123    ┆ 0.987    ┆ 0.854    │
    │ mkt_t         ┆ 15.234   ┆ 14.123   ┆ 12.456   │
    │ beta_smb      ┆ 0.234    ┆ -0.123   ┆ -0.345   │
    │ smb_t         ┆ 3.456    ┆ -2.134   ┆ -4.567   │
    │ beta_hml      ┆ -0.145   ┆ 0.234    ┆ 0.456    │
    │ hml_t         ┆ -2.345   ┆ 3.456    ┆ 5.678    │
    │ beta_rmw      ┆ 0.123    ┆ 0.234    ┆ 0.345    │
    │ rmw_t         ┆ 2.345    ┆ 3.456    ┆ 4.567    │
    │ beta_cma      ┆ -0.234   ┆ -0.345   ┆ -0.456   │
    │ cma_t         ┆ -3.456   ┆ -4.567   ┆ -5.678   │
    └───────────────┴──────────┴──────────┴──────────┘
    """

    portfolio_names = [col for col in portfolio_returns.columns if col.startswith("p_") or col == "spread"]
    
    # Drop nulls from scaling operations (rolling window warm-up period)
    df = portfolio_returns.drop_nulls(subset=portfolio_names)

    ff_factors = load_fama_french(
        start=df['date'].min(), 
        end=df['date'].max()
    )

    port = (
        df
            .join(ff_factors, on="date", how="inner")
            .sort("date")
            .with_columns([
                (pl.col(name) - pl.col("rf")).alias(name) for name in portfolio_names
            ])
    )

    name_map = {
        "const": "alpha", "mkt_rf": "beta_mkt", "smb": "beta_smb",
        "hml": "beta_hml", "rmw": "beta_rmw", "cma": "beta_cma"
    }

    results = []

    for p in portfolio_names:
        res = (
            port.select(
                pl.col(p).least_squares.ols(
                    pl.col('mkt_rf', 'smb', 'hml', 'rmw', 'cma'),
                    mode='statistics',
                    add_intercept=True
                ).alias("stats")
            )
            .unnest("stats")
            .explode("feature_names", "coefficients", "t_values") 
            .with_columns(
                pl.col("feature_names").replace(name_map)
            )
            .select([
                pl.col("feature_names"),
                pl.format("{} ({}){}", 
                    pl.col("coefficients").round(4), 
                    pl.col("t_values").round(2),
                    pl.when(pl.col("t_values").abs() > 2)
                    .then(pl.lit("*"))
                    .otherwise(pl.lit(""))
                ).alias(p)
            ])
        )
        results.append(res)

    final_df = results[0]
    for other_df in results[1:]:
        final_df = final_df.join(other_df, on="feature_names", how="left")

    return final_df