from typing import Protocol
import cvxpy as cp
import numpy as np
from functools import partial


class Constraint(Protocol):
    """
    Protocol for portfolio optimization constraints.

    Any class implementing this protocol must define a ``__call__`` method
    that accepts a ``cvxpy.Variable`` representing portfolio weights and
    returns a ``cvxpy.Constraint``.

    Methods
    -------
    __call__(weights, **kwargs) -> cp.Constraint
        Apply the constraint to the portfolio weights.
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint: ...


class FullInvestment(Constraint):
    """
    Enforces a full-investment constraint.

    This constraint ensures that the sum of all portfolio weights equals 1.

    Examples
    --------
    >>> weights = cp.Variable(3)
    >>> constraint = FullInvestment()(weights)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        return cp.sum(weights) == 1


class LongOnly:
    """
    Enforces a long-only constraint.

    This constraint ensures that all portfolio weights are non-negative.

    Examples
    --------
    >>> weights = cp.Variable(3)
    >>> constraint = LongOnly()(weights)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        return weights >= 0


class NoBuyingOnMargin:
    """
    Enforces a no-buying-on-margin constraint.

    This constraint ensures that the total invested capital does not exceed 1,
    i.e., it prohibits leveraged long positions.

    Examples
    --------
    >>> weights = cp.Variable(3)
    >>> constraint = NoBuyingOnMargin()(weights)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        return cp.sum(weights) <= 1


class UnitBeta:
    """
    Enforces a unit-beta constraint.

    This constraint requires the portfolio's exposure to a given beta vector
    to equal 1. A ``betas`` array must be provided as a keyword argument.

    Raises
    ------
    ValueError
        If ``betas`` is not provided in the keyword arguments.

    Examples
    --------
    >>> weights = cp.Variable(3)
    >>> betas = np.array([0.5, 1.2, 0.8])
    >>> constraint = UnitBeta()(weights, betas=betas)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        betas: np.ndarray | None = kwargs.get("betas")
        if betas is None:
            raise ValueError("UnitBeta requires betas")
        return betas @ weights == 1


def _construct_constraints(
    constraints: list[Constraint], betas: np.ndarray | None = None
) -> list[cp.Constraint]:
    """
    Construct a list of cvxpy constraints.

    Parameters
    ----------
    constraints : list of Constraint
        List of constraint objects implementing the ``Constraint`` protocol.
    betas : np.ndarray, optional
        Beta exposures required by constraints such as ``UnitBeta``.

    Returns
    -------
    list of cp.Constraint
        List of instantiated ``cvxpy.Constraint`` objects.

    Examples
    --------
    >>> weights = cp.Variable(3)
    >>> betas = np.array([0.5, 1.2, 0.8])
    >>> constraints = _construct_constraints([FullInvestment(), UnitBeta()], betas=betas)
    >>> constraints = [c(weights) for c in constraints]  # apply constraints to weights
    """
    return [partial(constraint, betas=betas) for constraint in constraints]
