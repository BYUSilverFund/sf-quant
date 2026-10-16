import polars as pl
import sf_quant.performance as sfp

weights = pl.read_parquet("weights/momentum/0.05/*.parquet")
print(weights)

returns = sfp.generate_returns_from_weights(weights)
print(sfp.generate_returns_summary_table(returns))

sfp.generate_returns_chart(
    returns,
    title="Momentum Backtest",
    log_scale=True,
    file_name="results.png",
)
