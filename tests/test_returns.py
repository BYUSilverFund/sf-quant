"""Return tests. The assets table and benchmark are stubbed, so these need no data lake."""

import datetime as dt

import polars as pl
import pytest

import sf_quant.performance.returns as returns_module
from sf_quant.performance import (
    generate_multi_returns_from_weights,
    generate_returns_from_weights,
)

DATES = [dt.date(2024, 1, d) for d in (2, 3, 4)]
# Daily returns in percent, as the assets table stores them. B is out of the
# universe on DATES[1].
ASSETS = pl.DataFrame(
    {
        "date": DATES * 2,
        "barrid": ["A"] * 3 + ["B"] * 3,
        "return": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        "in_universe": [True] * 3 + [True, False, True],
    }
)


class StubTable:
    def scan(self, year=None):
        return ASSETS.lazy()


@pytest.fixture(autouse=True)
def stub_assets(monkeypatch):
    monkeypatch.setattr(returns_module, "assets_table", StubTable())


def stub_benchmark(monkeypatch, barrids, weights):
    frame = pl.DataFrame(
        {
            "date": [d for d in DATES for _ in barrids],
            "barrid": barrids * len(DATES),
            "weight": weights * len(DATES),
        }
    )
    monkeypatch.setattr(returns_module, "load_benchmark", lambda start, end: frame)


def test_held_names_earn_returns_on_days_out_of_the_universe():
    weights = pl.DataFrame({"date": DATES, "barrid": ["B"] * 3, "weight": [1.0] * 3})
    out = generate_returns_from_weights(weights)
    assert out["return"].to_list() == pytest.approx([-0.01, -0.02, -0.03])


def test_benchmark_names_the_portfolio_does_not_hold_still_count(monkeypatch):
    """Weights list only A; the benchmark is half A, half B."""
    stub_benchmark(monkeypatch, ["A", "B"], [0.5, 0.5])
    weights = pl.DataFrame({"date": DATES, "barrid": ["A"] * 3, "weight": [1.0] * 3})
    out = generate_multi_returns_from_weights(weights).filter(pl.col("date") == DATES[0])
    by_portfolio = dict(zip(out["portfolio"], out["return"]))
    assert by_portfolio["benchmark"] == pytest.approx(0.5 * 0.01 + 0.5 * -0.01)
    assert by_portfolio["active"] == pytest.approx(0.01 - 0.0)


def test_held_names_outside_the_benchmark_count_in_active(monkeypatch):
    """The benchmark is all A; the portfolio is half A, half B."""
    stub_benchmark(monkeypatch, ["A"], [1.0])
    weights = pl.DataFrame(
        {"date": [d for d in DATES for _ in "AB"], "barrid": ["A", "B"] * 3, "weight": [0.5, 0.5] * 3}
    )
    out = generate_multi_returns_from_weights(weights).filter(pl.col("date") == DATES[0])
    by_portfolio = dict(zip(out["portfolio"], out["return"]))
    assert by_portfolio["total"] == pytest.approx(0.0)
    assert by_portfolio["active"] == pytest.approx(0.0 - 0.01)
