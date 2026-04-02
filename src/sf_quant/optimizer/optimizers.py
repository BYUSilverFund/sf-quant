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
    initial_gamma: float = 100,
    betas: np.ndarray | None = None,
    target_active_risk: float | None = None,
    benchmark_weights: np.ndarray | None = None,
    active_weights: bool = False
) -> pl.DataFrame:
    """
    Mean-variance optimizer with optional active risk calibration.

    Extends :func:`mve_optimizer` with the ability to automatically calibrate
    the risk aversion parameter (gamma) to achieve a target level of active risk
    (tracking error) relative to a benchmark portfolio. Uses iterative optimization
    with linear regression to solve for the gamma that produces the desired active risk.

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
        List of constraint objects implementing the ``Constraint`` protocol.
    initial_gamma : float, optional
        Starting value for gamma in calibration. Also used as the warm-start seed
        if ``target_active_risk`` is specified. Default is 100.
    betas : np.ndarray, optional
        Predicted betas or other asset-level values required by certain constraints.
    target_active_risk : float, optional
        If specified, automatically calibrate gamma to achieve this target
        annualized active risk (e.g., 0.05 for 5%). Requires ``benchmark_weights``.
        If not specified, uses ``initial_gamma`` directly.
    benchmark_weights : np.ndarray, optional
        Benchmark portfolio weights of shape (n_assets,), required if
        ``target_active_risk`` is specified.
    active_weights : bool
        Flag indicating how to treat output weights of optimizer. False (default)
        means that we subtract of benchmark weights before computing active risk.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns:

        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.
        - ``gamma`` : float, calibrated risk aversion parameter.
        - ``active_risk`` : float, achieved annualized active risk.

    See Also
    --------
    mve_optimizer : Base mean-variance optimizer without active risk calibration.
    _calibrate_gamma : Gamma calibration routine.

    Examples
    --------
    >>> import sf_quant.optimizer as sfo
    >>> import numpy as np
    >>> ids = ['AAPL', 'IBM']
    >>> alphas = np.array([1.1, 1.2])
    >>> factor_exposures = np.array([[0.8, 0.5], [1.2, 0.3]])
    >>> factor_covariance = np.array([[0.5, 0.1], [0.1, 0.2]])
    >>> specific_risk = np.array([0.1, 0.15])
    >>> benchmark_weights = np.array([0.4, 0.6])
    >>> constraints = [sfo.FullInvestment()]
    >>> weights = sfo.dynamic_mve_optimizer(
    ...     ids=ids,
    ...     alphas=alphas,
    ...     factor_exposures=factor_exposures,
    ...     factor_covariance=factor_covariance,
    ...     specific_risk=specific_risk,
    ...     constraints=constraints,
    ...     initial_gamma=100,
    ...     target_active_risk=0.05,
    ...     benchmark_weights=benchmark_weights
    ... )
    >>> weights
    shape: (2, 4)
    ┌────────┬────────┬───────┬──────────────┐
    │ barrid ┆ weight ┆ gamma ┆ active_risk  │
    │ ---    ┆ ---    ┆ ---   ┆ ---          │
    │ str    ┆ f64    ┆ f64   ┆ f64          │
    ╞════════╪════════╪═══════╪══════════════╡
    │ AAPL   ┆ 0.40   ┆ 75.5  ┆ 0.05         │
    │ IBM    ┆ 0.60   ┆ 75.5  ┆ 0.05         │
    └────────┴────────┴───────┴──────────────┘
    """
    constructed_constraints = _construct_constraints(constraints, betas=betas)

    # Build constraints for the calibration loop
    calibrated_gamma, active_risk = _calibrate_gamma(
        alphas=alphas,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        benchmark_weights=benchmark_weights,
        constraints=constructed_constraints,
        target_active_risk=target_active_risk,
        initial_gamma=initial_gamma,
        active_weights=active_weights
    )

    optimal_weights = _quadratic_program(
        alphas=alphas,
        gamma=calibrated_gamma,
        constraints=constructed_constraints,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
    )

    weights = pl.DataFrame({"barrid": ids, "weight": optimal_weights, 'gamma': calibrated_gamma, 'active_risk' : active_risk})
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

def _calibrate_gamma(
    alphas: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_risk: np.ndarray,
    benchmark_weights: np.ndarray,
    constraints: list[Constraint],
    target_active_risk: float,
    initial_gamma: float = 100.0,
    error: float = 0.005,
    max_iterations: int = 5,
    active_weights: bool = False
):
    """
    Calibrate gamma to hit a target annualized active risk.

    Uses iterative optimization with linear regression on 1/(2*lambda) vs
    observed active risk to predict the gamma that produces the desired
    annualized tracking error vs. the benchmark.

    Parameters
    ----------
    alphas : np.ndarray
        Expected returns, shape (n_assets,).
    factor_exposures : np.ndarray
        B matrix, shape (n_assets, n_factors).
    factor_covariance : np.ndarray
        F matrix, shape (n_factors, n_factors).
    specific_risk : np.ndarray
        D vector, shape (n_assets,).
    benchmark_weights : np.ndarray
        Benchmark portfolio weights, shape (n_assets,).
    constraints : list[cp.Constraint]
        List of already-built CVXPY constraints.
    target_active_risk : float
        Desired annualized active risk (e.g., 0.05 for 5%).
    initial_gamma : float
        Starting value for gamma. Default 100.
    error : float
        Convergence tolerance. Default 0.005 (0.5%).
    max_iterations : int
        Maximum number of iterations. Default 5.
    active_weights : bool
        Flag indicating how to treat output weights of optimizer. False (default)
        means that we subtract of benchmark weights before computing active risk.

    Returns
    -------
    float
        Calibrated gamma.
    """
    gamma = initial_gamma
    active_risk = float('inf')
    iterations = 1
    data = []

    while abs(active_risk - target_active_risk) > error and iterations <= max_iterations:
        # Solve the optimization
        weights = _quadratic_program(alphas, factor_exposures, factor_covariance, specific_risk, gamma, constraints)
  
        # Compute active weights if active_weights flag is False
        if not active_weights:
            active_w = weights - benchmark_weights
        else:
            active_w = weights

        # Compute active risk using factor model
        active_factor_loadings = factor_exposures.T @ active_w
        active_factor_var = active_factor_loadings @ factor_covariance @ active_factor_loadings
        active_specific_var = np.sum(specific_risk * np.square(active_w))
        active_variance = active_factor_var + active_specific_var
        active_risk = float(np.sqrt(active_variance))
        data.append((gamma, active_risk))

        if abs(active_risk - target_active_risk) <= error:
            break

        # Predict next gamma using linear regression on 1/(2*lambda) vs active risk
        data_np = np.array(data)
        gammas = data_np[:, 0]
        ar_vals = data_np[:, 1]
        X = 1.0 / (2.0 * gammas)
        M = np.dot(X, ar_vals) / np.dot(X, X)
        gamma = M / (2.0 * target_active_risk)

        iterations += 1
    return gamma, active_risk