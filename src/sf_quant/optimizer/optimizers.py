import numpy as np
import cvxpy as cp
import polars as pl

from .constraints import Constraint, _construct_constraints


def _quadratic_program(
    alphas: np.ndarray,
    factor_exposures: np.ndarray,  # B: (N x K)
    factor_covariance: np.ndarray, # F: (K x K)
    specific_risk: np.ndarray,     # D: (N,) vector of idiosyncratic variance
    gamma: float,
    constraints: list[cp.Constraint],
) -> np.ndarray:
    n_assets = len(alphas)

    weights = cp.Variable(n_assets)

    constraints = [constraint(weights) for constraint in constraints]

    portfolio_return = weights.T @ alphas
    # alternative, faster calculation of portfolio variance
    factor_loadings = factor_exposures.T @ weights
    factor_variance = cp.quad_form(factor_loadings, factor_covariance)
    specific_variance = cp.sum(cp.multiply(specific_risk, cp.square(weights)))
    portfolio_variance = factor_variance + specific_variance
    #portfolio_variance = weights.T @ covariance_matrix @ weights

    objective = cp.Maximize(portfolio_return - 0.5 * gamma * portfolio_variance)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver="OSQP")

    return weights.value

def mve_optimizer(
    ids: list[str],
    alphas: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_risk: np.ndarray,
    constraints: list[Constraint],
    gamma: float = 2,
    betas: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Mean-variance optimizer using a Factor Risk Model decomposition.

    Parameters
    ----------
    ids : list of str
        Asset identifiers.
    alphas : np.ndarray
        Expected returns, shape (n_assets,).
    factor_exposures : np.ndarray
        B matrix, shape (n_assets, n_factors).
    factor_covariance : np.ndarray
        F matrix, shape (n_factors, n_factors).
    specific_risk : np.ndarray
        D vector, shape (n_assets,).
    constraints : list of Constraint
        Standard sf-quant constraint objects.
    gamma : float
        Risk aversion (default 2).
    betas : np.ndarray, optional
        Predicted betas for specific constraints.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with 'barrid' and 'weight'.
        
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
        gamma=gamma,
        constraints=constraints,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
    )

    weights = pl.DataFrame({"barrid": ids, "weight": optimal_weights})
    return weights