import polars as pl
import sf_quant.performance as sfp
import datetime as dt
weights = pl.DataFrame(
    {
        'date': [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 3)],
        'barrid': ['USA06Z1', 'USA0771', 'USA06Z1', 'USA0771'],
        'weight': [0.5, 0.5, 0.3, 0.7]
    }
)
returns = sfp.generate_returns_from_weights(weights)
summary = sfp.generate_summary_table(returns)
print(summary)