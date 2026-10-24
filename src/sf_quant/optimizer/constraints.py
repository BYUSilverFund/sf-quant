from functools import partial
from numbers import Real
from typing import Protocol

import cvxpy as cp
import numpy as np


class Constraint(Protocol):
    """
    Protocol for portfolio optimization constraints.

    Any class implementing this protocol must define a ``__call__`` method
    that accepts a ``cvxpy.Variable`` representing portfolio weights (active
    or total) and returns a ``cvxpy.Constraint``.
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """
        Apply the constraint to the portfolio weights.

        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights variable of shape (n_assets,).
        **kwargs
            Additional keyword arguments for constraint-specific parameters
            (e.g., ``betas``, ``benchmark_weights``).

        Returns
        -------
        cp.Constraint
            A CVXPY constraint object.
        """
        ...


class FullInvestment(Constraint):
    """
    Enforces a full-investment constraint on total portfolio weights.

    Ensures that the sum of total portfolio weights equals 1:
    - Total space (no benchmark): :math:`\\mathbf{1}^T w = 1`
    - Active space (with benchmark): :math:`\\mathbf{1}^T (h + b) = 1`
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the full-investment constraint."""
        benchmark_weights: np.ndarray | None = kwargs.get("benchmark_weights")
        total_weights = (
            weights if benchmark_weights is None else weights + benchmark_weights
        )
        return cp.sum(total_weights) == 1


class ZeroInvestment(Constraint):
    """
    Enforces a zero-investment (self-financing / dollar-neutral) constraint.

    Ensures that the sum of active (or long-short) weights equals 0:
    :math:`\\mathbf{1}^T h = 0`.
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the zero-investment constraint."""
        return cp.sum(weights) == 0


class LongOnly(Constraint):
    """
    Enforces a long-only constraint on total portfolio weights.

    Prohibits outright short positions in the total portfolio:
    - Total space (no benchmark): :math:`w \\ge 0`
    - Active space (with benchmark): :math:`h + b \\ge 0 \\iff h \\ge -b`
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the long-only constraint."""
        benchmark_weights: np.ndarray | None = kwargs.get("benchmark_weights")
        total_weights = (
            weights if benchmark_weights is None else weights + benchmark_weights
        )
        return total_weights >= 0


class NoBuyingOnMargin(Constraint):
    """
    Enforces a strict no-buying-on-margin constraint on total portfolio weights.

    Ensures that total long exposure does not exceed 100% of equity,
    prohibiting the use of borrowed money or short-sale proceeds to fund
    additional long positions:
    - Total space (no benchmark): :math:`\\sum_i \\max(w_i, 0) \\le 1`
    - Active space (with benchmark): :math:`\\sum_i \\max(h_i + b_i, 0) \\le 1`
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the no-buying-on-margin constraint."""
        benchmark_weights: np.ndarray | None = kwargs.get("benchmark_weights")
        total_weights = (
            weights if benchmark_weights is None else weights + benchmark_weights
        )
        return cp.sum(cp.pos(total_weights)) <= 1.0


class GrossLeverage(Constraint):
    """
    Constrains total portfolio gross exposure (L1 norm).

    Parameters
    ----------
    max_leverage : float, default 1.0
        The maximum allowed gross exposure (:math:`\\|w\\|_1 = \\sum_i |w_i|`).
        Must be finite and strictly positive; boolean values are not accepted.
        - ``1.0`` for an unleveraged portfolio.
        - ``1.6`` permits up to 130% long and 30% short when net exposure is 100%.
    """

    def __init__(self, max_leverage: float = 1.0):
        if isinstance(max_leverage, (bool, np.bool_)) or not isinstance(
            max_leverage, Real
        ):
            raise ValueError(
                "max_leverage must be a finite, strictly positive real number."
            )
        try:
            limit = float(max_leverage)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "max_leverage must be a finite, strictly positive real number."
            ) from exc
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(
                "max_leverage must be a finite, strictly positive real number."
            )
        self.max_leverage = limit

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the gross-leverage constraint."""
        benchmark_weights: np.ndarray | None = kwargs.get("benchmark_weights")
        total_weights = (
            weights if benchmark_weights is None else weights + benchmark_weights
        )
        return cp.norm1(total_weights) <= self.max_leverage


class UnitBeta(Constraint):
    """
    Enforces a unit-beta constraint on total portfolio weights.

    Requires the total portfolio's exposure to ``betas`` to equal 1:
    - Total space (no benchmark): :math:`\\beta^T w = 1`
    - Active space (with benchmark): :math:`\\beta^T (h + b) = 1`
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the unit-beta constraint."""
        betas: np.ndarray | None = kwargs.get("betas")
        if betas is None:
            raise ValueError("UnitBeta requires betas")

        benchmark_weights: np.ndarray | None = kwargs.get("benchmark_weights")
        total_weights = (
            weights if benchmark_weights is None else weights + benchmark_weights
        )
        return betas @ total_weights == 1


class ZeroBeta(Constraint):
    """
    Enforces a zero-beta constraint on active (or market-neutral) weights.

    Requires the active portfolio's exposure to ``betas`` to equal 0:
    :math:`\\beta^T h = 0`.
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the zero-beta constraint."""
        betas: np.ndarray | None = kwargs.get("betas")
        if betas is None:
            raise ValueError("ZeroBeta requires betas")
        return betas @ weights == 0


def _construct_constraints(
    constraints: list[Constraint],
    betas: np.ndarray | None = None,
    benchmark_weights: np.ndarray | None = None,
) -> list:
    """
    Construct a list of partial constraint functions.

    Wraps constraint objects with their required context keyword arguments
    (``betas`` and ``benchmark_weights``) so they can be evaluated against
    a ``cvxpy.Variable``.

    Parameters
    ----------
    constraints : list[Constraint]
        List of constraint objects implementing the ``Constraint`` protocol.
    betas : np.ndarray, optional
        Asset beta exposures required by ``UnitBeta`` or ``ZeroBeta``.
    benchmark_weights : np.ndarray, optional
        Benchmark weights used by total-space constraints (``LongOnly``,
        ``FullInvestment``, ``NoBuyingOnMargin``, ``GrossLeverage``, ``UnitBeta``)
        to map active weights ``h`` to total weights ``w = h + b``.

    Returns
    -------
    list
        List of partial constraint functions ready to be called with a
        ``cvxpy.Variable``.
    """
    return [
        partial(
            constraint,
            betas=betas,
            benchmark_weights=benchmark_weights,
        )
        for constraint in constraints
    ]
