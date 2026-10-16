import numpy as np
import cvxpy as cp
import polars as pl

from .constraints import Constraint, _construct_constraints

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
    Mean-variance optimizer using a factor risk model decomposition.

    Solves a mean-variance optimization problem where portfolio variance
    is computed using a factor model with Fama-French style factor decomposition:

    .. math::

        \\Sigma = B F B^T + \\text{diag}(D)

    Parameters
    ----------
    ids : list[str]
        Asset identifiers (e.g., ticker symbols or BARRAs).
    alphas : np.ndarray
        Expected returns for each asset, shape (n_assets,).
    factor_exposures : np.ndarray
        B matrix of shape (n_assets, n_factors), containing asset factor exposures.
    factor_covariance : np.ndarray
        F matrix of shape (n_factors, n_factors), factor covariance matrix.
    specific_risk : np.ndarray
        D vector of shape (n_assets,), idiosyncratic (specific) variance per asset.
    constraints : list[Constraint]
        List of constraint objects implementing the ``Constraint`` protocol
        (e.g., ``FullInvestment()``, ``LongOnly()``).
    gamma : float, optional
        Risk aversion parameter. Higher values penalize variance more strongly.
        Default is 2.
    betas : np.ndarray, optional
        Predicted betas or other asset-level values required by certain constraints
        such as ``UnitBeta`` or ``ZeroBeta``.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns:

        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.

    Examples
    --------
    >>> import sf_quant.optimizer as sfo
    >>> import numpy as np
    >>> ids = ['AAPL', 'IBM']
    >>> alphas = np.array([1.1, 1.2])
    >>> factor_exposures = np.array([[0.8, 0.5], [1.2, 0.3]])
    >>> factor_covariance = np.array([[0.5, 0.1], [0.1, 0.2]])
    >>> specific_risk = np.array([0.1, 0.15])
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
    │ AAPL   ┆ 0.45   │
    │ IBM    ┆ 0.55   │
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

def dynamic_mve_optimizer(
    ids: list[str],
    alphas: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_risk: np.ndarray,
    constraints: list[Constraint],
    target_active_risk: float,
    betas: np.ndarray | None = None,
    benchmark_weights: np.ndarray | None = None,
    active_weights: bool = True,
    enforce_zero_beta: bool = True
) -> pl.DataFrame:
    """
    Mean-variance optimizer with active risk calibration solved directly in active weight space.

    Extends standard mean-variance optimization by automatically enforcing the core Silver Fund
    active mandate: dollar neutrality, benchmark-relative long-only bounds, and conditional
    beta neutrality. It evaluates custom constraints while safely intercepting and ignoring
    redundant total-space constraints to prevent solver conflicts.

    Parameters
    ----------
    ids : list[str]
        Asset identifiers (e.g., ticker symbols or BARRAs).
    alphas : np.ndarray
        Expected returns for each asset, shape (n_assets,).
    factor_exposures : np.ndarray
        B matrix of shape (n_assets, n_factors), containing asset factor exposures.
    factor_covariance : np.ndarray
        F matrix of shape (n_factors, n_factors), factor covariance matrix.
    specific_risk : np.ndarray
        D vector of shape (n_assets,), idiosyncratic variance per asset.
    constraints : list[Constraint]
        List of constraint objects implementing the Constraint protocol.
        Redundant baseline constraints ("FullInvestment", "ZeroInvestment", "LongOnly",
        "UnitBeta", "ZeroBeta", "NoBuyingOnMargin") are automatically intercepted and ignored.
        Custom constraints are evaluated on the total weight vector.
    target_active_risk : float
        Automatically calibrate the implied risk aversion (gamma) to achieve this target
        annualized active risk (e.g., 0.05).
    betas : np.ndarray, optional
        Predicted asset betas. Required if enforce_zero_beta is True.
    benchmark_weights : np.ndarray, optional
        Benchmark portfolio weights of shape (n_assets,). Required to enforce the long-only
        bound and to calculate total weights.
    active_weights : bool, default True
        Flag indicating the output format. True (default) returns pure active weights.
        False adds benchmark weights back to return fully invested total weights.
    enforce_zero_beta : bool, default True
        If True and betas are provided, enforces a strict zero active beta constraint.
        Set to False when testing factor signals with systematic beta exposure.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns:

        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.
        - ``gamma`` : float, calibrated risk aversion parameter.
        - ``active_risk`` : float, achieved annualized active risk.
    """
    if benchmark_weights is None:
        raise ValueError(
            "benchmark_weights must be provided to enforce the long-only boundary (h >= -b)."
        )

    if enforce_zero_beta and betas is None:
        raise ValueError(
            "betas must be provided when enforce_zero_beta is True."
        )
    
    n_assets = len(alphas)
    
    h = cp.Variable(n_assets)
    w = benchmark_weights + h  # Simplified

    factor_loadings = factor_exposures.T @ h
    factor_variance = cp.quad_form(factor_loadings, factor_covariance)
    specific_variance = cp.sum(cp.multiply(specific_risk, cp.square(h)))
    active_variance = factor_variance + specific_variance
    te_constraint = active_variance <= (target_active_risk ** 2)

    cvxpy_constraints = [
        cp.sum(h) == 0,
        h >= -benchmark_weights,  # Simplified unconditionally
        te_constraint
    ]

    if enforce_zero_beta:
        cvxpy_constraints.append(betas @ h == 0)  # Simplified

    if constraints:
        redundant_types = {
            "FullInvestment",
            "ZeroInvestment",
            "LongOnly",
            "UnitBeta",
            "ZeroBeta",
            "NoBuyingOnMargin"
        }
        
        custom_constraints = [c for c in constraints if type(c).__name__ not in redundant_types]
        if custom_constraints:
            constructed_custom = _construct_constraints(custom_constraints, betas=betas)
            cvxpy_constraints.extend([c(w) for c in constructed_custom])

    objective = cp.Maximize(alphas @ h)
    
    problem = cp.Problem(objective, cvxpy_constraints)
    problem.solve(solver='CLARABEL')

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed. Status: {problem.status}")

    optimal_h = h.value

    act_fac_load = factor_exposures.T @ optimal_h
    act_fac_var = act_fac_load.T @ factor_covariance @ act_fac_load
    act_spec_var = np.sum(specific_risk * np.square(optimal_h))
    realized_active_risk = float(np.sqrt(max(0.0, act_fac_var + act_spec_var)))

    if te_constraint.dual_value is not None:
        calibrated_gamma = float(2.0 * te_constraint.dual_value)
    else:
        calibrated_gamma = 0.0

    if active_weights:  # Simplified
        output_weights = optimal_h
    else:
        output_weights = optimal_h + benchmark_weights

    weights = pl.DataFrame({
        "barrid": ids,
        "weight": output_weights,
        "gamma": calibrated_gamma,
        "active_risk": realized_active_risk
    })
    
    return weights

def _quadratic_program(
    alphas: np.ndarray,
    factor_exposures: np.ndarray,  # B: (N x K)
    factor_covariance: np.ndarray, # F: (K x K)
    specific_risk: np.ndarray,     # D: (N,) vector of idiosyncratic variance
    gamma: float,
    constraints: list[cp.Constraint],
) -> np.ndarray:
    """
    Solve a mean-variance optimization problem using a factor risk model.

    This is a private helper function that uses CVXPY to solve:

    .. math::

        \\max_w \\left( w^T \\alpha - \\frac{\\gamma}{2} w^T \\Sigma w \\right)

    where :math:`\\Sigma` is the factor model covariance matrix:

    .. math::

        \\Sigma = B F B^T + \\text{diag}(D)

    Parameters
    ----------
    alphas : np.ndarray
        Expected returns, shape (n_assets,).
    factor_exposures : np.ndarray
        B matrix of shape (n_assets, n_factors), asset exposures to factors.
    factor_covariance : np.ndarray
        F matrix of shape (n_factors, n_factors), factor covariance.
    specific_risk : np.ndarray
        D vector of shape (n_assets,), idiosyncratic variance per asset.
    gamma : float
        Risk aversion parameter. Higher values penalize variance more.
    constraints : list[cp.Constraint]
        List of instantiated CVXPY constraints.

    Returns
    -------
    np.ndarray
        Optimal portfolio weights, shape (n_assets,).
    """
    n_assets = len(alphas)

    weights = cp.Variable(n_assets)
    constraints = [constraint(weights) for constraint in constraints]

    portfolio_return = weights.T @ alphas
    factor_loadings = factor_exposures.T @ weights
    factor_variance = cp.quad_form(factor_loadings, factor_covariance)
    specific_variance = cp.sum(cp.multiply(specific_risk, cp.square(weights)))
    portfolio_variance = factor_variance + specific_variance

    objective = cp.Maximize(portfolio_return - 0.5 * gamma * portfolio_variance)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver="OSQP")

    return weights.value