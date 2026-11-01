Portfolio simulation
====================

Portfolio backtests must use ``simulate_portfolio`` instead of the removed
``generate_returns_from_weights`` or ``generate_multi_returns_from_weights``.
Simulation executes targets, carries adjusted units between trades, and derives
returns from NAV. Legacy weight multiplication does not model portfolio drift.

.. code-block:: python

   from sf_quant.simulation import SimulationConfig, simulate_portfolio

   result = simulate_portfolio(
       targets=weights,
       asset_returns=asset_returns,  # decimal TOTAL returns, dated at interval end
       benchmark_weights=benchmark,  # complete daily compositions for active inputs
       config=SimulationConfig(frequency="weekly", weight_type="active"),
   )
   returns = result.returns("total")
   active_returns = result.returns("active")
   multiple_returns = result.multi_returns()

The example assumes active targets. Use ``weight_type="total"`` for physical
total portfolio targets. Supply all held-asset return observations, including
assets that leave the signal universe. Missing held returns fail explicitly.

Default execution lag is one session. A close-observed decision executes at the
next session close and first earns the interval ending one session later. Never
shift these inputs forward. Lag zero requires information available before close.

.. autoclass:: sf_quant.simulation.SimulationConfig
   :members:

.. autoclass:: sf_quant.simulation.SimulationResult
   :members:

.. autofunction:: sf_quant.simulation.simulate_portfolio
