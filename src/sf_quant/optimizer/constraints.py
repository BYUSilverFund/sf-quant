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
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """
        Apply the constraint to the portfolio weights.

        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights variable of shape (n_assets,).
        **kwargs
            Additional keyword arguments for constraint-specific parameters.

        Returns
        -------
        cp.Constraint
            A CVXPY constraint object.
        """
        ...


class FullInvestment(Constraint):
    """
    Enforces a full-investment constraint.

    This constraint ensures that the sum of all portfolio weights equals 1.

    Returns
    -------
    cp.Constraint
        A CVXPY constraint enforcing full investment.

    Examples
    --------
    >>> import cvxpy as cp
    >>> from sf_quant.optimizer.constraints import FullInvestment
    >>> weights = cp.Variable(3)
    >>> constraint = FullInvestment()(weights)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the full-investment constraint."""
        return cp.sum(weights) == 1


class ZeroInvestment(Constraint):
    """
    Enforces a zero-investment constraint.

    This constraint ensures that the sum of all portfolio weights equals 0,
    representing a market-neutral position.

    Returns
    -------
    cp.Constraint
        A CVXPY constraint enforcing zero investment.

    Examples
    --------
    >>> import cvxpy as cp
    >>> from sf_quant.optimizer.constraints import ZeroInvestment
    >>> weights = cp.Variable(3)
    >>> constraint = ZeroInvestment()(weights)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the zero-investment constraint."""
        return cp.sum(weights) == 0
    

class LongOnly(Constraint):
    """
    Enforces a long-only constraint.

    This constraint ensures that all portfolio weights are non-negative,
    prohibiting short positions.

    Returns
    -------
    cp.Constraint
        A CVXPY constraint enforcing non-negative weights.

    Examples
    --------
    >>> import cvxpy as cp
    >>> from sf_quant.optimizer.constraints import LongOnly
    >>> weights = cp.Variable(3)
    >>> constraint = LongOnly()(weights)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the long-only constraint."""
        return weights >= 0


class NoBuyingOnMargin(Constraint):
    """
    Enforces a no-buying-on-margin constraint.

    This constraint ensures that the total invested capital does not exceed 1,
    prohibiting leveraged long positions.

    Returns
    -------
    cp.Constraint
        A CVXPY constraint limiting total investment to 1.

    Examples
    --------
    >>> import cvxpy as cp
    >>> from sf_quant.optimizer.constraints import NoBuyingOnMargin
    >>> weights = cp.Variable(3)
    >>> constraint = NoBuyingOnMargin()(weights)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """Apply the no-buying-on-margin constraint."""
        return cp.sum(weights) <= 1


class UnitBeta(Constraint):
    """
    Enforces a unit-beta constraint.

    This constraint requires the portfolio's exposure to a given beta vector
    to equal 1. A ``betas`` array must be provided as a keyword argument.

    Returns
    -------
    cp.Constraint
        A CVXPY constraint enforcing unit beta exposure.

    Raises
    ------
    ValueError
        If ``betas`` is not provided in the keyword arguments.

    Examples
    --------
    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from sf_quant.optimizer.constraints import UnitBeta
    >>> weights = cp.Variable(3)
    >>> betas = np.array([0.5, 1.2, 0.8])
    >>> constraint = UnitBeta()(weights, betas=betas)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """
        Apply the unit-beta constraint.

        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights variable.
        **kwargs
            Must include ``betas`` : np.ndarray
                Beta exposures for each asset.

        Returns
        -------
        cp.Constraint
            A CVXPY constraint.
        """
        betas: np.ndarray | None = kwargs.get("betas")
        if betas is None:
            raise ValueError("UnitBeta requires betas")
        return betas @ weights == 1
    
class ZeroBeta(Constraint):
    """
    Enforces a zero-beta constraint.

    This constraint requires the portfolio's exposure to a given beta vector
    to equal 0, creating a market-neutral position. A ``betas`` array must be
    provided as a keyword argument.

    Returns
    -------
    cp.Constraint
        A CVXPY constraint enforcing zero beta exposure.

    Raises
    ------
    ValueError
        If ``betas`` is not provided in the keyword arguments.

    Examples
    --------
    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from sf_quant.optimizer.constraints import ZeroBeta
    >>> weights = cp.Variable(3)
    >>> betas = np.array([0.5, 1.2, 0.8])
    >>> constraint = ZeroBeta()(weights, betas=betas)
    """

    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        """
        Apply the zero-beta constraint.

        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights variable.
        **kwargs
            Must include ``betas`` : np.ndarray
                Beta exposures for each asset.

        Returns
        -------
        cp.Constraint
            A CVXPY constraint.
        """
        betas: np.ndarray | None = kwargs.get("betas")
        if betas is None:
            raise ValueError("ZeroBeta requires betas")
        return betas @ weights == 0


def _construct_constraints(
    constraints: list[Constraint], betas: np.ndarray | None = None
) -> list:
    """
    Construct a list of partial constraint functions.

    Wraps constraint objects with their required keyword arguments
    (e.g., ``betas``) to prepare them for application to weight variables.

    Parameters
    ----------
    constraints : list[Constraint]
        List of constraint objects implementing the ``Constraint`` protocol.
    betas : np.ndarray, optional
        Beta exposures required by constraints such as ``UnitBeta`` and ``ZeroBeta``.

    Returns
    -------
    list
        List of partial constraint functions, each ready to be called with
        a ``cvxpy.Variable`` to produce a ``cvxpy.Constraint``.

    Examples
    --------
    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from sf_quant.optimizer.constraints import _construct_constraints, FullInvestment, UnitBeta
    >>> betas = np.array([0.5, 1.2, 0.8])
    >>> constraint_partials = _construct_constraints([FullInvestment(), UnitBeta()], betas=betas)
    >>> weights = cp.Variable(3)
    >>> cvxpy_constraints = [c(weights) for c in constraint_partials]
    """
    return [partial(constraint, betas=betas) for constraint in constraints]
