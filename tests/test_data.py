"""Data loader tests. Table.scan is stubbed, so these need no data lake."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

import sf_quant.data.covariance_matrix as cov_module
from sf_quant.data._factors import factors


BARRIDS = ["USA0001", "USA0002", "USA0003"]
SPECIFIC_RISK = np.array([20.0, 35.0, 50.0])  # percent, as stored


def stub_risk_model(monkeypatch, date, exposure_barrids=BARRIDS, specific_barrids=BARRIDS):
    """Stub the three risk tables; returns B, F (percent^2) in BARRIDS order."""
    rng = np.random.default_rng(0)
    B = rng.normal(size=(len(BARRIDS), len(factors)))
    root = rng.normal(size=(len(factors), len(factors)))
    F = root @ root.T  # percent^2, as stored

    keep = [BARRIDS.index(b) for b in exposure_barrids]
    exposures = pl.DataFrame({"date": [date] * len(keep), "barrid": exposure_barrids}).with_columns(
        pl.Series(f, B[keep, j]) for j, f in enumerate(factors)
    )
    upper = np.where(np.triu(np.ones_like(F)) > 0, F, np.nan)  # stored upper-triangular
    covariances = pl.DataFrame({"date": [date] * len(factors), "factor_1": factors}).with_columns(
        pl.Series(f, upper[:, j]) for j, f in enumerate(factors)
    )
    keep = [BARRIDS.index(b) for b in specific_barrids]
    assets = pl.DataFrame(
        {"date": [date] * len(keep), "barrid": specific_barrids, "specific_risk": SPECIFIC_RISK[keep]}
    )

    monkeypatch.setattr(cov_module, "load_exposures_by_date", lambda d: exposures)
    monkeypatch.setattr(cov_module, "load_covariances_by_date", lambda d: covariances)
    monkeypatch.setattr(cov_module, "load_assets_by_date", lambda d, in_universe, columns: assets)
    return B, F


def test_components_follow_the_order_barrids_are_given_in(monkeypatch):
    date = dt.date(2024, 3, 5)
    B, F = stub_risk_model(monkeypatch, date)
    order = ["USA0003", "USA0001", "USA0002"]
    rows = [BARRIDS.index(b) for b in order]

    exposures, factor_covariance, specific = cov_module.construct_factor_model_components(date, order)
    np.testing.assert_allclose(exposures, B[rows])
    np.testing.assert_allclose(factor_covariance, F / 100**2)
    np.testing.assert_allclose(specific, SPECIFIC_RISK[rows] ** 2 / 100**2)


@pytest.mark.parametrize("missing_from", ["exposures", "specific_risk"])
def test_a_barrid_without_risk_data_raises(monkeypatch, missing_from):
    date = dt.date(2024, 3, 5)
    present = BARRIDS[:2]
    if missing_from == "exposures":
        stub_risk_model(monkeypatch, date, exposure_barrids=present)
    else:
        stub_risk_model(monkeypatch, date, specific_barrids=present)

    with pytest.raises(ValueError, match=f"{missing_from} missing for 1 of 3 barrids.*USA0003"):
        cov_module.construct_factor_model_components(date, BARRIDS)
