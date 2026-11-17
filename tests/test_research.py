"""research helper tests. Fama-French data is stubbed, so these need no data lake."""

import datetime as dt

import numpy as np
import polars as pl

import sf_quant.research.ff_regression as ff_module
from sf_quant.research import run_quantile_ff_regression

FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma"]
DATES = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(60)]


def test_quantile_ff_regression_matches_ols(monkeypatch):
    rng = np.random.default_rng(0)
    ff = pl.DataFrame(
        {"date": DATES, **{f: rng.normal(0, 0.01, len(DATES)) for f in FACTORS}, "rf": [0.0001] * len(DATES)}
    )
    monkeypatch.setattr(ff_module, "load_fama_french", lambda start, end: ff.clone())
    X = ff.select(FACTORS).to_numpy()
    ports = pl.DataFrame(
        {
            "date": DATES,
            "p_1": X @ [1.0, 0.2, 0.0, 0.1, 0.0] + rng.normal(0, 0.002, len(DATES)),
            "p_2": X @ [0.9, -0.3, 0.4, 0.0, 0.2] + 0.001 + rng.normal(0, 0.002, len(DATES)),
        }
    )

    out = run_quantile_ff_regression(ports)

    assert out["feature_names"].to_list() == ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "alpha"]
    design = np.column_stack([X, np.ones(len(DATES))])
    for p in ["p_1", "p_2"]:
        y = ports[p].to_numpy() - 0.0001
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        sigma2 = resid @ resid / (len(y) - design.shape[1])
        t = coef / np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
        expected = pl.DataFrame({"c": coef, "t": t}).select(
            pl.format(
                "{} ({}){}",
                pl.col("c").round(4),
                pl.col("t").round(2),
                pl.when(pl.col("t").abs() > 2).then(pl.lit("*")).otherwise(pl.lit("")),
            )
        ).to_series()
        assert out[p].to_list() == expected.to_list()
