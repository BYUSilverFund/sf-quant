import numpy as np
import cvxpy as cp
import polars as pl

from .constraints import Constraint, _construct_constraints


def _quadratic_program(
    alphas: np.ndarray,
    gamma: float,
    constraints: list[cp.Constraint],
    factor_exposures: np.ndarray,  # B: shape (N, K)
    factor_covariance: np.ndarray, # F: shape (K, K)
    specific_risk: np.ndarray,     # D: shape (N,)
) -> np.ndarray:
    n_assets = len(alphas)

    weights = cp.Variable(n_assets)

    constraints = [constraint(weights) for constraint in constraints]

    portfolio_return = weights.T @ alphas
    # portfolio_variance = weights.T @ covariance_matrix @ weights
    factor_loadings = factor_exposures.T @ weights
    factor_variance = cp.quad_form(factor_loadings, factor_covariance)
    specific_variance = cp.sum(cp.multiply(specific_risk, cp.square(weights)))
    portfolio_variance = factor_variance + specific_variance

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
    Mean-variance optimizer with constraints, utilizing a factor risk model.

    This function solves the constrained mean-variance optimization
    problem using a structured factor decomposition for high-dimensional
    efficiency, and returns optimal portfolio weights.

    Parameters
    ----------
    ids : list of str
        Identifiers for the assets, used to label the output.
    alphas : np.ndarray
        Expected returns for each asset, shape (n_assets,).
    factor_exposures : np.ndarray
        Asset exposures to risk factors, shape (n_assets, n_factors).
    factor_covariance : np.ndarray
        Covariance matrix of the risk factors, shape (n_factors, n_factors).
    specific_risk : np.ndarray
        Idiosyncratic variance for each asset, shape (n_assets,).
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
    >>> # 1 Factor, 2 Assets
    >>> factor_exposures = np.array([[1.0], [0.8]]) 
    >>> factor_covariance = np.array([[0.1]])
    >>> specific_risk = np.array([0.05, 0.04])
    >>> constraints = [sfo.FullInvestment()]
    >>> weights = sfo.mve_optimizer(
    ...     ids=ids,
    ...     alphas=alphas,
    ...     factor_exposures=factor_exposures,
    ...     factor_covariance=factor_covariance,
    ...     specific_risk=specific_risk,
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
