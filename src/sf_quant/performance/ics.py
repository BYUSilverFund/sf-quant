import polars as pl
from sf_quant.schema import AlphaSchema, SecurityRetSchema
import dataframely as dy

def get_alpha_ics(
    alphas: dy.DataFrame[AlphaSchema],
    rets:   dy.DataFrame[SecurityRetSchema],
    method: str = "rank",      # "pearson" or "rank"
) -> pl.DataFrame:
    """
    IC between previous-day alpha and today's realized return, by date.
    Aligns by taking alpha_{t-1} per (barrid), then joining with return_t on (date, barrid).
    Returns: [date, ic, n, ic_type]
    """
    # 1) Build previous-observation alpha per security
    a_lag = (
        alphas
        .sort(["barrid", "date"])
        .with_columns(
            pl.col("alpha").shift(1).over("barrid").alias("alpha_lag")
        )
        .select("date", "barrid", "alpha_lag")
    )

    # 2) Join with realized returns on the same date
    df = (
        a_lag.join(rets.select("date", "barrid", "return"), on=["date", "barrid"], how="inner")
             .filter(pl.col("alpha_lag").is_not_null()
                     & pl.col("alpha_lag").is_finite()
                     & pl.col("return").is_finite())
    )

    m = method.lower()
    if m not in {"pearson", "rank"}:
        raise ValueError("method must be 'pearson' or 'rank'")

    if m == "pearson":
        ic = (
            df.group_by("date")
              .agg(
                  pl.len().alias("n"),
                  pl.pearson_corr("alpha_lag", "return").alias("ic"),
              )
              .with_columns(pl.lit("Pearson IC").alias("ic_type"))
        )
    else:  # rank (Spearman) = Pearson on ranks within each date
        ranked = df.with_columns(
            pl.col("alpha_lag").rank(method="average").over("date").alias("alpha_r"),
            pl.col("return").rank(method="average").over("date").alias("ret_r"),
        )
        ic = (
            ranked.group_by("date")
                  .agg(
                      pl.len().alias("n"),
                      pl.pearson_corr("alpha_r", "ret_r").alias("ic"),
                  )
                  .with_columns(pl.lit("Rank IC").alias("ic_type"))
        )

    return ic.select("date", "ic", "n", "ic_type")