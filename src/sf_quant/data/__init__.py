"""
The ``data`` module provides access to assets and related datasets used
throughout the package. It exposes a clean public API for loading asset data
and retrieving available columns, while keeping internal implementation
details hidden.
"""

from .assets import load_assets, load_assets_by_date, get_assets_columns
from .crsp_daily import load_crsp_daily, get_crsp_daily_columns
from .crsp_monthly import load_crsp_monthly, get_crsp_monthly_columns
from .exposures import load_exposures_by_date, get_exposures_columns
from .covariances import load_covariances_by_date, get_covariances_columns
from .covariance_matrix import construct_covariance_matrix
from .benchmark import load_benchmark

__all__ = [
    "load_assets",
    "load_assets_by_date",
    "get_assets_columns",
    "load_crsp_daily",
    "get_crsp_daily_columns",
    "load_crsp_monthly",
    "get_crsp_monthly_columns",
    "load_exposures_by_date",
    "get_exposures_columns",
    "load_covariances_by_date",
    "get_covariances_columns",
    "construct_covariance_matrix",
    "load_benchmark"
]
