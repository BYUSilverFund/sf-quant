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
    """
    constructed_constraints = _construct_constraints(constraints, betas=betas)

    optimal_weights = _quadratic_program(
        alphas=alphas,
        gamma=gamma,
        constraints=constructed_constraints,
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
    target_active_risk: float = 0.05,
    betas: np.ndarray | None = None,
    benchmark_weights: np.ndarray | None = None,
    active_weights: bool = True,
) -> pl.DataFrame:
    """
    Mean-variance optimizer with dynamic risk calibration in active weight space.

    Solves for optimal active weights :math:`h` by maximizing expected alpha subject
    to a conic risk cap (:math:`h^T V h \\le \\sigma_{\\text{target}}^2`) and any
    user-supplied ``Constraint`` objects. Implied risk aversion (:math:`\\gamma`) is
    extracted analytically from the conic dual shadow price.

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
        List of modular ``Constraint`` objects to enforce:
        - Silver Fund Mandate: ``[ZeroInvestment(), LongOnly(), ZeroBeta()]``
        - 130/30 Extension: ``[ZeroInvestment(), GrossLeverage(1.60)]``
        - Dollar-neutral Research: ``[ZeroInvestment()]``
        - Risk-only Research: ``[]``
        An empty list applies only the risk cap. No investment, beta-neutrality,
        or long-only constraints are implied.
    target_active_risk : float, default 0.05
        Target annualized risk cap (e.g., 0.05 for 5%).
    betas : np.ndarray, optional
        Predicted asset betas. Required if beta constraints (``ZeroBeta``,
        ``UnitBeta``) are included in ``constraints``.
    benchmark_weights : np.ndarray, optional
        Benchmark portfolio weights of shape (n_assets,). Used by total-space
        constraints (``LongOnly``, ``FullInvestment``, ``NoBuyingOnMargin``,
        ``GrossLeverage``, ``UnitBeta``) to map :math:`w = h + b`, and required
        when ``active_weights=False``.
    active_weights : bool, default True
        If True, returns the optimized active weight vector :math:`h`.
        If False, returns total portfolio weights :math:`w = h + b`.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns:

        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight.
        - ``gamma`` : float, calibrated risk aversion parameter.
        - ``active_risk`` : float, model-predicted annualized active risk.
    """
    if not active_weights and benchmark_weights is None:
        raise ValueError(
            "benchmark_weights must be provided when active_weights is False."
        )

    n_assets = len(alphas)
    h = cp.Variable(n_assets)

    factor_loadings = factor_exposures.T @ h
    factor_variance = cp.quad_form(factor_loadings, factor_covariance)
    specific_variance = cp.sum(cp.multiply(specific_risk, cp.square(h)))
    active_variance = factor_variance + specific_variance
    te_constraint = active_variance <= (target_active_risk ** 2)

    cvxpy_constraints = [te_constraint]

    if constraints:
        constructed = _construct_constraints(
            constraints,
            betas=betas,
            benchmark_weights=benchmark_weights,
        )
        cvxpy_constraints.extend([c(h) for c in constructed])

    objective = cp.Maximize(alphas @ h)
    problem = cp.Problem(objective, cvxpy_constraints)
    # One thread per solve: the parallel backtesters run one solve per worker
    # thread, and the solver's own auto-threading would oversubscribe the CPUs.
    problem.solve(solver="CLARABEL", max_threads=1)

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

    if active_weights:
        output_weights = optimal_h
    else:
        output_weights = optimal_h + benchmark_weights

    weights = pl.DataFrame({
        "barrid": ids,
        "weight": output_weights,
        "gamma": calibrated_gamma,
        "active_risk": realized_active_risk,
    })

    return weights


def _quadratic_program(
    alphas: np.ndarray,
    factor_exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_risk: np.ndarray,
    gamma: float,
    constraints: list,
) -> np.ndarray:
    """
    Solve a mean-variance optimization problem using a factor risk model.
    """
    n_assets = len(alphas)

    weights = cp.Variable(n_assets)
    cvxpy_constraints = [constraint(weights) for constraint in constraints]

    portfolio_return = weights.T @ alphas
    factor_loadings = factor_exposures.T @ weights
    factor_variance = cp.quad_form(factor_loadings, factor_covariance)
    specific_variance = cp.sum(cp.multiply(specific_risk, cp.square(weights)))
    portfolio_variance = factor_variance + specific_variance

    objective = cp.Maximize(portfolio_return - 0.5 * gamma * portfolio_variance)

    problem = cp.Problem(objective, cvxpy_constraints)
    problem.solve(solver="OSQP")

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed. Status: {problem.status}")

    return weights.value
