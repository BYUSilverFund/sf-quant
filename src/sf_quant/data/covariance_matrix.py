import datetime as dt
import numpy as np
import polars as pl
from .exposures import load_exposures_by_date
from .covariances import load_covariances_by_date
from .assets import load_assets_by_date

def construct_covariance_matrix(date_: dt.date, barrids: list[str]) -> pl.DataFrame:
    """
    Constructs the covariance matrix based on exposures, factor covariances, and specific risks.

    Args:
        date_ (dt.date): The date for which the covariance matrix is computed.
        barrids (List[str]): List of Barrid identifiers for the assets.

    Returns:
        CovarianceMatrix: The computed covariance matrix wrapped in a CovarianceMatrix object.
    """
    # Load
    exposures_matrix = _construct_factor_exposure_matrix(date_, barrids).drop("barrid").to_numpy()
    covariance_matrix = _construct_factor_covariance_matrix(date_).drop("factor_1").to_numpy()
    idio_risk_matrix = _construct_specific_risk_matrix(date_, barrids).drop("barrid").to_numpy()

    # Compute covariance matrix
    covariance_matrix = exposures_matrix @ covariance_matrix @ exposures_matrix.T + idio_risk_matrix

    # Put in decimal space
    covariance_matrix = covariance_matrix / (100**2)

    # Package
    covariance_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: covariance_matrix[:, i] for i, id in enumerate(barrids)},
        }
    )

    return covariance_matrix


def _construct_factor_exposure_matrix(date_: dt.date, barrids: list[str]) -> pl.DataFrame:
    """
    Constructs the factor exposure matrix for the given date and Barrids.

    Args:
        date_ (date): The date for which the factor exposure matrix is computed.
        barrids (List[str]): List of Barrid identifiers for the assets.

    Returns:
        pl.DataFrame: The factor exposure matrix.
    """
    exp_mat = (
        load_exposures_by_date(date_)
        .drop('date')
        .filter(pl.col('barrid').is_in(barrids)).fill_null(0)
        .sort('barrid')
    )

    return exp_mat


def _construct_factor_covariance_matrix(date_: dt.date) -> pl.DataFrame:
    """
    Constructs the factor covariance matrix for the given date.

    Args:
        date_ (date): The date for which the factor covariance matrix is computed.

    Returns:
        pl.DataFrame: The factor covariance matrix.
    """
    # Load
    fc_df = load_covariances_by_date(date_).drop('date')

    # Sort headers and columns
    fc_df = fc_df.select(["factor_1"] + sorted([col for col in fc_df.columns if col != "factor_1"]))
    fc_df = fc_df.sort("factor_1")

    # Record factor ids
    factors = fc_df.select("factor_1").to_numpy().flatten()

    # Convert from upper triangular to symetric
    utm = fc_df.drop("factor_1").to_numpy()
    cov_mat = np.where(np.isnan(utm), utm.T, utm)

    # Package
    cov_mat = pl.DataFrame(
        {
            "factor_1": factors,
            **{col: cov_mat[:, idx] for idx, col in enumerate(factors)},
        }
    )

    # Fill NaN (from Barra)
    cov_mat = cov_mat.fill_nan(0)

    return cov_mat


def _construct_specific_risk_matrix(date_: dt.date, barrids: list[str]) -> pl.DataFrame:
    """
    Constructs the specific risk matrix for the given date and Barrids.

    Args:
        date_ (date): The date for which the specific risk matrix is computed.
        barrids (List[str]): List of Barrid identifiers for the assets.

    Returns:
        pl.DataFrame: The specific risk matrix.
    """
    # Barrids
    barrids_df = pl.DataFrame({"barrid": barrids})

    # Load
    sr_df = load_assets_by_date(date_, in_universe=False, columns=['date', 'barrid', 'specific_risk'])

    # Filter
    sr_df = barrids_df.join(sr_df, on=["barrid"], how="left").fill_null(
        0
    )  # ask Brandon about this.

    # Convert vector to diagonal matrix
    diagonal = np.power(np.diag(sr_df["specific_risk"]), 2)

    # Package
    risk_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: diagonal[:, i] for i, id in enumerate(barrids)},
        }
    )

    return risk_matrix