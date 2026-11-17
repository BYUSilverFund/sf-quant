import datetime as dt
import numpy as np
import polars as pl
from .exposures import load_exposures_by_date
from .covariances import load_covariances_by_date
from .assets import load_assets_by_date
from ._factors import factors


def _reindex(
    df: pl.DataFrame, barrids: list[str], *, date_: dt.date, label: str, probe: str
) -> pl.DataFrame:
    """
    Put the rows of ``df`` in the same order as ``barrids``. Raise an error if any
    barrid is missing or appears twice.

    The optimizer matches its inputs by position: row 1 of the exposures, row 1 of
    the specific risk and row 1 of the alphas must all be the same stock. So every
    input must list the barrids in the same order.

    A missing barrid means the data is incomplete, so it raises an error instead of
    filling the gap. Filling it would quietly give wrong results. For example, a
    stock with no specific risk would be filled with 0, which makes it look riskless,
    and the optimizer would then put as much weight as it can on that one stock.
    """
    out = pl.DataFrame({"barrid": barrids}).join(
        df, on="barrid", how="left", maintain_order="left"
    )
    if out.height != len(barrids):
        dupes = (
            out["barrid"]
            .value_counts()
            .filter(pl.col("count") > 1)["barrid"]
            .to_list()
        )
        raise ValueError(
            f"{date_}: {label} table has duplicate rows for barrids {dupes[:10]}"
        )
    if out["barrid"].to_list() != list(barrids):
        raise ValueError(f"{date_}: {label} reindex did not preserve barrid order")
    if n_missing := out[probe].null_count():
        missing = out.filter(pl.col(probe).is_null())["barrid"].to_list()
        raise ValueError(
            f"{date_}: {label} missing for {n_missing} of {len(barrids)} barrids: "
            f"{missing[:10]}"
        )
    return out


def construct_factor_model_components(
    date_: dt.date, barrids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the factor model components for the given assets.

    All three are returned in ``barrids`` order. A barrid with no exposures or
    no specific risk on ``date_`` raises rather than being filled.

    Returns
    -------
    tuple of np.ndarray
        - factor_exposures: shape (N, K)
        - factor_covariance: shape (K, K), in decimal space
        - specific_risk: shape (N,), variance in decimal space
    """
    exposures = _construct_factor_exposure_matrix(date_, barrids).drop("barrid").to_numpy()
    
    factor_cov = _construct_factor_covariance_matrix(date_).drop("factor_1").to_numpy()
    factor_cov = factor_cov / 100**2

    sr_df = _reindex(
        load_assets_by_date(
            date_, in_universe=False, columns=["date", "barrid", "specific_risk"]
        ),
        barrids,
        date_=date_,
        label="specific_risk",
        probe="specific_risk",
    )
    
    specific_risk = np.power(sr_df["specific_risk"].to_numpy(), 2) / 100**2

    return exposures, factor_cov, specific_risk


def construct_covariance_matrix(date_: dt.date, barrids: list[str]) -> pl.DataFrame:
    """
    Construct the asset covariance matrix from a factor model.

    This function builds a covariance matrix for the given assets by
    combining factor exposures, factor covariances, and specific
    (idiosyncratic) risks. The resulting covariance matrix can be used
    as an input to mean-variance optimization.

    Parameters
    ----------
    date_ : datetime.date
        The date for which the covariance matrix is computed.
    barrids : list of str
        List of Barrid identifiers for the assets.

    Returns
    -------
    pl.DataFrame
        A square covariance matrix stored in a Polars DataFrame.

        - Rows and columns are indexed by ``barrid``.
        - Column ``barrid`` lists the asset identifiers.
        - Each subsequent column corresponds to the covariance of
          the row asset with the column asset.

    Notes
    -----
    The input factor covariance matrix is assumed to be positive
    semidefinite (PSD).

    Examples
    --------
    >>> import sf_quant.data as sfd
    >>> import datetime as dt
    >>> date_ = dt.date(2024, 1, 3)
    >>> barrids = ['USA06Z1', 'USA0771']
    >>> covariance_matrix = sfd.construct_covariance_matrix(
    ...     date_=date_,
    ...     barrids=barrids
    ... )
    >>> covariance_matrix
    shape: (2, 3)
    ┌─────────┬─────────────┬──────────────┐
    │ barrid  ┆ USA06Z1     ┆ USA0771      │
    │ ---     ┆ ---         ┆ ---          │
    │ str     ┆ f64         ┆ f64          │
    ╞═════════╪═════════════╪══════════════╡
    │ USA06Z1 ┆ 3224.338938 ┆ 697.641425   │
    │ USA0771 ┆ 697.641425  ┆ 11158.366868 │
    └─────────┴─────────────┴──────────────┘
    """
    # Load
    exposures_matrix = (
        _construct_factor_exposure_matrix(date_, barrids).drop("barrid").to_numpy()
    )
    covariance_matrix = (
        _construct_factor_covariance_matrix(date_).drop("factor_1").to_numpy()
    )
    idio_risk_matrix = (
        _construct_specific_risk_matrix(date_, barrids).drop("barrid").to_numpy()
    )

    # Compute covariance matrix
    covariance_matrix = (
        exposures_matrix @ covariance_matrix @ exposures_matrix.T + idio_risk_matrix
    )

    # Put in decimal space
    covariance_matrix /= 100**2

    # Package
    covariance_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: covariance_matrix[:, i] for i, id in enumerate(barrids)},
        }
    )

    return covariance_matrix


def _construct_factor_exposure_matrix(
    date_: dt.date, barrids: list[str]
) -> pl.DataFrame:
    exp_mat = (
        load_exposures_by_date(date_)
        .drop("date")
        .filter(pl.col("barrid").is_in(barrids))
        .fill_null(0)
        .select(["barrid"] + factors)
    )

    # Reindex after the fill so a surviving null means "absent from the exposures
    # table", not "null exposure".
    return _reindex(
        exp_mat, barrids, date_=date_, label="exposures", probe=factors[0]
    )


def _construct_factor_covariance_matrix(date_: dt.date) -> pl.DataFrame:
    # Load
    fc_df = load_covariances_by_date(date_).drop("date")

    # Sort headers and columns
    fc_df = fc_df.select(["factor_1"] + factors)

    fc_df = fc_df.filter(pl.col('factor_1').is_in(factors))

    fc_df = fc_df.sort("factor_1")

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
    # Barrids
    barrids_df = pl.DataFrame({"barrid": barrids})

    # Load
    sr_df = load_assets_by_date(
        date_, in_universe=False, columns=["date", "barrid", "specific_risk"]
    )

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
