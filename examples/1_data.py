import polars as pl
import datetime as dt
import sf_quant.data as sfd

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = 'momentum'
price_filter = 5
IC = .05

benchmark_weights = sfd.load_benchmark(start, end).rename({'weight': 'benchmark_weight'})
data = (
    sfd.load_assets(
        start=start,
        end=end,
        columns=[
            'date', 
            'barrid', 
            'price',
            'return',
            'specific_risk',
            'predicted_beta',
        ],
        in_universe=True
    )
    .with_columns(
        pl.col('return', 'specific_risk').truediv(100)
    )
    .join(
        other=benchmark_weights,
        on=['date', 'barrid'],
        how='left'
    )
    .sort('date', 'barrid')
)

signals = (
    data
    .sort('date', 'barrid')
    .with_columns(
        pl.col('return')
        .log1p()
        .rolling_sum(230)
        .shift(21)
        .over('barrid')
        .alias(signal_name)
    )
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

# Compute scores
scores = filtered.with_columns(
    "date",
    "barrid",
    "predicted_beta",
    "specific_risk",
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean())
    .truediv(pl.col(signal_name).std())
    .over("date")
    .alias("score"),
)

# Compute alphas
alphas = (
    scores
    .with_columns(
        pl.col("score")
        .mul(IC)
        .mul("specific_risk")
    .alias("alpha")
    )
    .select("date", "barrid", "alpha", "predicted_beta", "benchmark_weight")
    .sort("date", "barrid")
)

print(alphas)

alphas.write_parquet(f"{signal_name}_alphas.parquet")