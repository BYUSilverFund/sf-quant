:orphan:

generate_multi_returns_from_weights (removed)
=============================================

.. warning::

   NOTE this does not calculate correct returns, as it ignores portfolio drift.
   Do not use it. Use ``simulate_portfolio`` in ``sf_quant.simulation`` instead.

The old public name is no longer callable. See :doc:`../simulation` for the new
API and its explicit return-data and execution-timing requirements. Use
``result.returns("total")`` for one portfolio or ``result.multi_returns()`` for
benchmark-relative reporting.
