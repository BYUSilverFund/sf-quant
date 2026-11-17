"""Benchmark return tests. The assets table is stubbed, so these need no data lake."""

import datetime as dt

import polars as pl
import pytest

import sf_quant.data._views as views
from sf_quant.data import load_benchmark_returns

DAYS = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]


class StubTable:
    def __init__(self, frame):
        self.frame = frame

    def scan(self, year=None):
        return self.frame.lazy()


def test_weights_come_from_the_previous_close(monkeypatch):
    """A and B start equal; A rises 10% on day two. Day two's benchmark return is
    the 50/50 average (5%), not weighted by A's larger closing cap (5.24%)."""
    assets = pl.DataFrame({
        "date": DAYS * 2,
        "barrid": ["A", "A", "B", "B"],
        "market_cap": [100.0, 110.0, 100.0, 100.0],
        "return": [0.0, 10.0, 0.0, 0.0],
        "in_universe": [True] * 4,
    })
    monkeypatch.setattr(views, "assets_table", StubTable(assets))
    out = load_benchmark_returns(DAYS[0], DAYS[1])
    assert out["date"].to_list() == [DAYS[1]]  # day one has no previous close in the data
    assert out["bmk_return"].item() == pytest.approx(5.0)


def test_only_stocks_in_the_universe_at_the_previous_close_count(monkeypatch):
    assets = pl.DataFrame({
        "date": DAYS * 2,
        "barrid": ["A", "A", "B", "B"],
        "market_cap": [100.0, 100.0, 100.0, 100.0],
        "return": [0.0, 2.0, 0.0, 8.0],
        "in_universe": [True, True, False, True],  # B joins on day two
    })
    monkeypatch.setattr(views, "assets_table", StubTable(assets))
    assert load_benchmark_returns(DAYS[0], DAYS[1])["bmk_return"].item() == pytest.approx(2.0)
