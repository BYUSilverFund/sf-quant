"""Summary-table tests. Synthetic returns only, so these need no data lake."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from sf_quant.performance import (
    generate_multi_returns_summary_table,
    generate_returns_summary_table,
)

DATES = [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
RETURNS = [0.01, -0.005, 0.02, 0.0, 0.004]
MEAN, STD = np.mean(RETURNS), np.std(RETURNS, ddof=1)


def single():
    return pl.DataFrame({"date": DATES, "return": RETURNS})


def multi():
    portfolios = ["total", "benchmark", "active"]
    return pl.DataFrame(
        {
            "date": DATES * len(portfolios),
            "portfolio": [p for p in portfolios for _ in DATES],
            "return": RETURNS * len(portfolios),
        }
    )


def test_defaults_annualize_daily_returns_with_no_risk_free():
    out = generate_returns_summary_table(single())
    assert out["Sharpe"].item() == pytest.approx(round(MEAN / STD * 252**0.5, 2))


def test_periods_per_year_sets_the_annualization():
    out = generate_returns_summary_table(single(), periods_per_year=12)
    assert out["Mean Return (%)"].item() == pytest.approx(round(MEAN * 12 * 100, 2))
    assert out["Volatility (%)"].item() == pytest.approx(round(STD * 12**0.5 * 100, 2))
    assert out["Sharpe"].item() == pytest.approx(round(MEAN / STD * 12**0.5, 2))


def test_risk_free_is_subtracted_for_the_sharpe_ratio():
    out = generate_returns_summary_table(single(), risk_free=0.5)
    expected = (MEAN * 252 - 0.5) / (STD * 252**0.5)
    assert out["Sharpe"].item() == pytest.approx(round(expected, 2))


def test_multi_table_leaves_the_active_portfolio_excess_free():
    out = generate_multi_returns_summary_table(multi(), risk_free=0.5)
    sharpe = dict(zip(out["Portfolio"], out["Sharpe"]))
    excess = round((MEAN * 252 - 0.5) / (STD * 252**0.5), 2)
    assert sharpe["Total"] == pytest.approx(excess)
    assert sharpe["Benchmark"] == pytest.approx(excess)
    assert sharpe["Active"] == pytest.approx(round(MEAN / STD * 252**0.5, 2))
