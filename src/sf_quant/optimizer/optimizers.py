import numpy as np
import cvxpy as cp
import polars as pl

from .constraints import Constraint, _construct_constraints


def _quadratic_program(
    alphas: np.ndarray,
    covariance_matrix: np.ndarray,
    gamma: float,
    constraints: list[cp.Constraint],
) -> np.ndarray:
    n_assets = len(alphas)

    weights = cp.Variable(n_assets)

    constraints = [constraint(weights) for constraint in constraints]

    portfolio_return = weights.T @ alphas
    portfolio_variance = weights.T @ covariance_matrix @ weights

    objective = cp.Maximize(portfolio_return - 0.5 * gamma * portfolio_variance)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver="OSQP")

    return weights.value


def mve_optimizer(
    ids: list[str],
    alphas: np.ndarray,
    covariance_matrix: np.ndarray,
    constraints: list[Constraint],
    gamma: float = 2,
    betas: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Mean-variance optimizer with constraints.

    This function solves the constrained mean-variance optimization
    problem and returns optimal portfolio weights aligned with the
    provided ids.

    Parameters
    ----------
    ids : list of str
        Identifiers for the assets, used to label the output.
    alphas : np.ndarray
        Expected returns for each asset, shape (n_assets,).
    covariance_matrix : np.ndarray
        Covariance matrix of asset returns, shape (n_assets, n_assets).
    constraints : list of Constraint
        List of constraint objects implementing the ``Constraint`` protocol.
    gamma : float
        Risk-aversion parameter. Higher values penalize variance more strongly. Defaults to 2.
    betas : np.ndarray, optional
        Predicted betas, required for certain constraints (e.g., :class:`UnitBeta`).

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with barrid and weight columns.

    Examples
    --------
    >>> import sf_quant.optimizer as sfo
    >>> import numpy as np
    >>> ids = ['AAPL', 'IBM']
    >>> alphas = np.array([1.1, 1.2])
    >>> covariance_matrix = np.array([
    ...     [.5, .1],
    ...     [.1, .2]
    ... ])
    >>> constraints = [sfo.FullInvestment()]
    >>> weights = sfo.mve_optimizer(
    ...     ids=ids,
    ...     alphas=alphas,
    ...     covariance_matrix=covariance_matrix,
    ...     constraints=constraints
    ... )
    >>> weights
    shape: (2, 2)
    ┌────────┬────────┐
    │ barrid ┆ weight │
    │ ---    ┆ ---    │
    │ str    ┆ f64    │
    ╞════════╪════════╡
    │ AAPL   ┆ 0.1    │
    │ IBM    ┆ 0.9    │
    └────────┴────────┘
    """
    constraints = _construct_constraints(constraints, betas=betas)

    optimal_weights = _quadratic_program(
        alphas=alphas,
        covariance_matrix=covariance_matrix,
        gamma=gamma,
        constraints=constraints,
    )

    weights = pl.DataFrame({"barrid": ids, "weight": optimal_weights})

    return weights
