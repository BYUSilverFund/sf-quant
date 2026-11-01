Performance
===========

.. automodule:: sf_quant.performance
   :no-members:
   :no-undoc-members:
   :no-imported-members:

.. warning::

   ``generate_returns_from_weights`` and ``generate_multi_returns_from_weights``
   are retired because they ignore portfolio drift. Use
   :func:`sf_quant.simulation.simulate_portfolio`, then ``result.returns("total")``
   or ``result.multi_returns()``. Only explicitly named ``legacy_`` implementations
   remain for reproducing old arithmetic, and every call emits a warning.

See :doc:`simulation` for the execution API and migration example.

.. currentmodule:: sf_quant.performance

.. autosummary::
   :toctree: performance
   :nosignatures:

   simulate_portfolio
   get_trade_turnover_stats
   generate_leverage_from_weights
   generate_drawdown_from_returns
   generate_returns_summary_table
   generate_multi_returns_summary_table
   generate_leverage_summary_table
   generate_drawdown_summary_table
   generate_returns_chart
   generate_multi_returns_chart
   generate_leverage_chart
   generate_drawdown_chart
   generate_alpha_ics
   generate_ic_chart
   get_turnover_stats
   plot_turnover