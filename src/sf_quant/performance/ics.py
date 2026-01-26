import polars as pl
from sf_quant.schema import AlphaSchema, SecurityRetSchema
import dataframely as dy

def generate_alpha_ics(
    alphas: dy.DataFrame[AlphaSchema],
    rets:   dy.DataFrame[SecurityRetSchema],
    method: str = "rank",      # "pearson" or "rank"
    window: int = 22
    ) -> pl.DataFrame:
    """
    Compute Information Coefficients (ICs) between previous-day alpha and realized returns.

    This function calculates the IC for each date by aligning the alpha signal
    with the realized return at the end of the given window for each
    asset. IC can be computed either as a Pearson correlation (raw values) or as a 
    Spearman correlation (ranked values).

    Parameters
    ----------
    alphas : pl.DataFrame
        A Polars DataFrame containing alpha signals.
        Must include the following columns:

        - ``date`` (date): The observation date of the alpha.
        - ``barrid`` (str): Unique asset identifier.
        - ``alpha`` (float): Alpha value for the asset on the given date.

        Must be validated against the ``AlphaSchema`` before calling this function.
    rets : pl.DataFrame
        A Polars DataFrame containing realized returns. Should
        include returns for at least [window] days after the end of the alpha period.
        Must include the following columns:

        - ``date`` (date): The date of the return.
        - ``barrid`` (str): Unique asset identifier.
        - ``return`` (float): Realized return for the asset on the given date.
    method : str, optional
        Method to compute IC. Either ``"pearson"`` for raw
        correlation or ``"rank"`` for Spearman correlation on ranks. Defaults to
        ``"rank"``.
    window : int, optional
        Number of days to compute rolling returns over. Defaults to 22.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing ICs by date with the following columns:

        - ``date`` (date): The observation date.
        - ``ic`` (float): Information coefficient for that date.
        - ``n`` (int): Number of observations used to compute the IC.

    Notes
    -----
    - The function first shifts alpha values by one day per asset to align with
      the next day's realized return.
    - Observations with null or non-finite alpha or return values are excluded.
    - Spearman IC is computed by ranking alpha and return values within each date
      and then calculating the Pearson correlation of ranks.
    - Users should validate that input DataFrames conform to the expected schema
      to avoid runtime errors.

    Examples
    --------
    >>> import polars as pl
    >>> import datetime as dt
    >>> alphas = pl.DataFrame(
    ...     {
    ...         'date': [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
    ...         'barrid': ['USA06Z1', 'USA06Z1'],
    ...         'alpha': [0.02, -0.01]
    ...     }
    ... )
    >>> rets = pl.DataFrame(
    ...     {
    ...         'date': [dt.date(2024, 1, 3), dt.date(2024, 1, 4)],
    ...         'barrid': ['USA06Z1', 'USA06Z1'],
    ...         'return': [0.01, -0.02]
    ...     }
    ... )

    >>> alphas = dy.validate(alphas, AlphaSchema)
    >>> rets = dy.validate(rets, SecurityRetSchema)
    >>> ic_df = generate_alpha_ics(alphas, rets, method="rank")
    >>> ic_df
    shape: (2, 3)
    ┌────────────┬──────────┬─────┐
    │ date       ┆ ic       ┆ n   │
    │ ---        ┆ ---      ┆ --- │
    │ date       ┆ f64      ┆ i64 │
    ╞════════════╪══════════╪═════╡
    │ 2024-01-03 ┆ 1.0      ┆ 1   │
    │ 2024-01-04 ┆ -1.0     ┆ 1   │
    └────────────┴──────────┴─────┘
    """
    # Lag alpha by security
    alphas = (
        alphas
        .sort(["barrid", "date"])
        .select("date", "barrid", "alpha")
    )

    rets = (
        rets
        .select("date", "barrid", "return")
        .sort(["barrid", "date"])
        .with_columns(
            pl.col("return").log1p()
            .rolling_sum(window).over("barrid")
            .exp().sub(1).shift(-window + 1)
            .alias("window_return")
            )
    )

    # Join with returns
    df = (
        alphas.join(rets.select("date", "barrid", "window_return"), on=["date", "barrid"], how="inner")
             .filter(pl.col("alpha").is_not_null()
                     & pl.col("alpha").is_finite()
                     & pl.col("window_return").is_not_null()
                     & pl.col("window_return").is_finite())
    )

    m = method.lower()
    if m not in {"pearson", "rank"}:
        raise ValueError("method must be 'pearson' or 'rank'")

    if m == "pearson":
        ic = (
            df.group_by("date")
              .agg(
                  pl.len().alias("n"),
                  pl.corr("alpha", "window_return").alias("ic"),
              )
        )
    else:  # rank
        ranked = df.with_columns(
            pl.col("alpha").rank(method="average").over("date").alias("alpha_r"),
            pl.col("window_return").rank(method="average").over("date").alias("ret_r"),
        )
        ic = (
            ranked.group_by("date")
                .agg(
                    pl.len().alias("n"),
                    pl.corr("alpha_r", "ret_r").alias("ic"),
                )
        )

    return ic.select("date", "ic", "n")