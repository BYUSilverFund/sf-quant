# Drift-aware simulation

`sf_quant.simulation` is independent of Ray, SLURM and database access:

```python
from sf_quant.simulation import SimulationConfig, simulate_portfolio

result = simulate_portfolio(
    targets,          # date: Date, barrid: String, weight: Float64; complete snapshots
    asset_returns,    # date, barrid, return: decimal TOTAL return ending on date
    benchmark_weights=benchmark,  # full daily date/barrid/weight compositions
    config=SimulationConfig(
        frequency="weekly",      # daily / weekly / monthly
        weight_type="active",    # active targets execute h + decision-date b
        execution_lag_sessions=1,
    ),
)
result.save("new_result_directory")  # refuses replacement; atomic publication
returns = result.returns("total")   # date, return, directly usable by performance APIs
active = result.returns("active")   # total minus benchmark, no RF subtraction
weights = result.weights()          # realized post-trade weights for leverage/charts
```

Returns are never shifted. At session t, old holdings earn the return ending t,
then trades execute at close t. Default lag one means close-observed decisions
execute at the NEXT session close, earning returns only thereafter. Lag zero
requires targets actually known before the close, not an assumption that close
market data can be traded at that same close. Intraday/open execution is unsupported.

Weekly/monthly default to the first available session of each calendar period;
`rebalance_day="last"` selects the last supplied session and requires full-period
calendar coverage. Initial investment is at the first eligible close, even when
it is not a period boundary. A target is executed once. Empty/missing decision
dates never force an old target to be reapplied. Supply `sessions=[...]` from a
trusted trading calendar to detect entirely missing market sessions; otherwise
sessions are inferred from the return panel. No synthetic weekday sessions are added.

Holdings are fixed total-return-adjusted units between trades. Reinvested
distributions and splits must already be reflected in total returns. Missing
held or benchmark returns fail; they are never interpreted as zero. An explicit
-100% terminal return writes a position off and prevents repurchasing its ID.
Other delisting payouts need caller-supplied adjusted returns. Raw prices,
separate cash dividends, external flows, tax lots and settlement lags are unsupported.

Cash equals NAV minus signed position values. `allow_short=True` explicitly opts
into a simplified financing model with unrestricted cash and shorts. Optional
`cash_lending_return`, `cash_borrow_return`, `short_fee_per_session` are fixed
PER-SESSION decimal rates; defaults are zero. No margin, collateral, borrow
availability, recalls, spread or market impact is modelled. Nonpositive NAV fails.
Without that opt-in, meaningful negative weights or borrowing fail. Numerical
optimizer deviations within 1e-7 are retained as explicit residual positions/cash.

Costs: `transaction_cost_bps * sum(abs(security_trade_dollars)) / 10000`, solved
against post-cost NAV. Daily two-sided turnover divides those absolute dollars
by pre-trade NAV, one-way turnover is half that. Inception trades count; cash
trades do not. No-trade drift has exactly zero turnover.
`sf_quant.performance.get_trade_turnover_stats(result.daily)` uses this definition;
legacy `get_turnover_stats(weights)` retains its previous weight-difference metric.

Outputs: daily beginning/pre/end NAV, investment/financing P&L, costs, cash,
returns, benchmark and active returns, executed decisions and turnover; holdings
contain pre/post weights, adjusted units, index levels, values and current
benchmark/active weights; trades contain executed dollars, units, costs and NAV.
The benchmark uses yesterday's complete composition, never today's market-cap
weights to explain a return already realized. A complete benchmark composition
is required on every session when benchmark reporting is requested.

The public names `generate_returns_from_weights` and
`generate_multi_returns_from_weights` are removed. Imports and calls using them
must migrate to `sf_quant.simulation.simulate_portfolio`; there are no callable
aliases or automatic substitutions. Simulation requires explicit decimal,
end-dated asset returns and an execution configuration, so renaming a call alone
is insufficient. Use `result.returns("total")` for one series and
`result.multi_returns()` for total/benchmark/active analytics.

`legacy_generate_returns_from_weights` and
`legacy_generate_multi_returns_from_weights` archive the old arithmetic for
historical reproducibility only. Each call emits a RuntimeWarning that it ignores
portfolio drift. Do not use either legacy function for portfolio backtests.
For FF5 use `run_ff_regression(result.returns("active"),
return_timing="realization", active_returns=True)`; defaults retain formation-date
legacy factor alignment. Ex-ante risk in targets is not realized tracking error,
and neither beta neutrality nor target active risk is reimposed between trades.

Tests: `python -m pytest tests -q`. Scale sample:
`python benchmarks/benchmark_simulation.py --assets 1000 --sessions 252`.
