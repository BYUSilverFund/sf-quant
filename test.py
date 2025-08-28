import sf_quant.data as sfd
import sf_quant.backtester as sfb
import sf_quant.optimizer as sfo
import polars as pl
import datetime as dt
start = dt.date(2024, 1, 1)
end = dt.date(2024, 1, 10)
columns = [
    'date',
    'barrid',
]
data = (
    sfd.load_assets(
        start=start,
        end=end,
        in_universe=True,
        columns=columns
    )
    .with_columns(
        pl.lit(0).alias('alpha')
    )
)
constraints = [
    sfo.FullInvestment()
]
weights = sfb.backtest_parallel(
    data=data,
    constraints=constraints,
    gamma=2,
)
print(weights.head())