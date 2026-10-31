from datetime import date, timedelta

import numpy as np
import polars as pl


def test_realization_date_active_regression_does_not_shift_or_subtract_rf(monkeypatch):
    from sf_quant.research import ff_regression as regression
    rng = np.random.default_rng(38)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    factors = rng.normal(0, .01, (30, 5))
    names = ["mkt_rf", "smb", "hml", "rmw", "cma"]
    ff = pl.DataFrame({"date": dates, "rf": [.01]*30,
                       **{name: factors[:, i] for i, name in enumerate(names)}})
    returns = pl.DataFrame({"date": dates, "return": .002 + factors[:, 0]*2})
    monkeypatch.setattr(regression, "load_fama_french", lambda **kw: ff)
    fit = regression.run_ff_regression(returns, return_timing="realization", active_returns=True)
    np.testing.assert_allclose(fit["coefficient"], [.2, 2, 0, 0, 0, 0], atol=1e-10)
