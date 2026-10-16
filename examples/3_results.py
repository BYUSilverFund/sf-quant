import polars as pl
import sf_quant.data as sfd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)

weights = pl.read_parquet("weights/momentum/0.05/*.parquet")
print(weights)

returns = (
    sfd.load_assets(
        start=start,
        end=end,
        columns=['date', 'barrid', 'return'],
        in_universe=True
    )
    .sort('date', 'barrid')
    .with_columns(
        pl.col('return')
        .truediv(100)
        .shift(-1)
        .over('barrid')
    )
    .drop_nulls()
    .sort('date', 'barrid')
)

results = (
    weights
    .join(
        other=returns,
        on=['date', 'barrid'],
        how='inner',
    )
    .group_by('date')
    .agg(
        pl.col('return')
        .mul(pl.col('weight'))
        .sum()
    )
    .sort('date')
    .with_columns(
        pl.col('return')
        .log1p()
        .cum_sum()
        .mul(100)
        .alias('cumulative_return')
    )
)

table = (
    results
    .select(
        pl.col('return').mean().mul(252 * 100).alias('mean'),
        pl.col('return').std().mul(pl.lit(252).sqrt() * 100).alias('volatility')
    )
    .with_columns(
        pl.col('mean').truediv(pl.col('volatility')).alias('sharpe')
    )
)

sharpe = table['sharpe'].item()

sns.lineplot(results, x='date', y='cumulative_return')
plt.title(f"Momentum Backtest ({sharpe:.2f})")
plt.xlabel(None)
plt.ylabel("Cumulative Log Returns (%)")

plt.savefig("results.png")