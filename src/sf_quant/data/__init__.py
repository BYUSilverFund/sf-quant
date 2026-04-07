"""
The ``data`` module provides access to assets and related datasets used
throughout the package. It exposes a clean public API for loading asset data
and retrieving available columns, while keeping internal implementation
details hidden.
"""

from ._config import env, EnvNotConfiguredError
from .assets import load_assets, load_assets_by_date, get_assets_columns
from .crsp_daily import load_crsp_daily, get_crsp_daily_columns
from .crsp_v2_daily import load_crsp_v2_daily, get_crsp_v2_daily_columns
from .crsp_monthly import load_crsp_monthly, get_crsp_monthly_columns
from .crsp_v2_monthly import load_crsp_v2_monthly, get_crsp_v2_monthly_columns
from .exposures import load_exposures, load_exposures_by_date, get_exposures_columns
from .covariances import load_covariances_by_date, get_covariances_columns
from .covariance_matrix import construct_covariance_matrix, construct_factor_model_components
from .benchmark import load_benchmark, load_benchmark_returns
from .factors import load_factors, get_factors_columns, get_factor_names
from .fama_french import load_fama_french, get_fama_french_columns
from .signals import load_signals, load_signals_by_date, get_signal_names
from .scores import load_scores, load_scores_by_date, get_score_names
from .alphas import load_alphas, load_alphas_by_date, get_alpha_names
from .composite_alphas import load_composite_alphas, load_composite_alphas_by_date

__all__ = [
    "env",
    "EnvNotConfiguredError",
    "load_assets",
    "load_assets_by_date",
    "get_assets_columns",
    "load_crsp_daily",
    "get_crsp_daily_columns",
    "load_crsp_v2_daily",
    "get_crsp_v2_daily_columns",
    "load_crsp_monthly",
    "get_crsp_monthly_columns",
    "load_crsp_v2_monthly",
    "get_crsp_v2_monthly_columns",
    "load_exposures",
    "load_exposures_by_date",
    "get_exposures_columns",
    "load_covariances_by_date",
    "get_covariances_columns",
    "construct_covariance_matrix",
    "construct_factor_model_components",
    "load_benchmark",
    "load_benchmark_returns",
    "load_factors",
    "get_factors_columns",
    "get_factor_names",
    "load_fama_french",
    "get_fama_french_columns",
    "load_signals",
    "load_signals_by_date",
    "get_signal_names",
    "load_scores",
    "load_scores_by_date",
    "get_score_names",
    "load_alphas",
    "load_alphas_by_date",
    "get_alpha_names",
    "load_composite_alphas",
    "load_composite_alphas_by_date",
]
