"""Parallel backtester tests. Synthetic risk model throughout, so these need no data lake.

These cover the thread-pool dispatch -- splitting by date, ordering,
reassembly: the parallel paths must reproduce single-worker results exactly.
"""

import datetime as dt
import importlib

import numpy as np
import polars as pl
import pytest

import sf_quant.optimizer as sfo
from sf_quant.backtester import (
    backtest_parallel,
    backtest_sequential,
    dynamic_backtest_parallel,
)

N, K = 30, 4
BARRIDS = [f"USA{i:04d}" for i in range(N)]
DATES = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(10)]


def fake_components(date_, barrids):
    """A well-conditioned factor model per date, returned in ``barrids`` order."""
    rng = np.random.default_rng(date_.toordinal())
    exposures = rng.normal(size=(N, K))
    root = rng.normal(size=(K, K))
    covariance = root @ root.T / K + np.eye(K) * 0.02
    specific = rng.uniform(0.02, 0.09, size=N) ** 2
    rows = [BARRIDS.index(b) for b in barrids]
    return exposures[rows], covariance, specific[rows]


def panel() -> pl.DataFrame:
    """Ten dates with a universe that changes date to date, rows shuffled."""
    rng = np.random.default_rng(0)
    frames = []
    for i, date_ in enumerate(DATES):
        names = BARRIDS[: N - i % 3]
        cap = rng.pareto(1.5, size=len(names)) + 1.0
        frames.append(
            pl.DataFrame(
                {
                    "date": [date_] * len(names),
                    "barrid": names,
                    "alpha": rng.normal(scale=0.02, size=len(names)),
                    "benchmark_weight": cap / cap.sum(),
                    "predicted_beta": rng.uniform(0.6, 1.5, size=len(names)),
                }
            )
        )
    return pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=1)


@pytest.fixture(autouse=True)
def synthetic_risk_model(monkeypatch):
    for name in ("sequential", "parallel", "dynamic_parallel"):
        module = importlib.import_module(f"sf_quant.backtester.{name}")
        monkeypatch.setattr(module, "construct_factor_model_components", fake_components)


def test_parallel_matches_sequential():
    data = panel()
    constraints = [sfo.FullInvestment()]
    sequential = backtest_sequential(data, constraints).sort("date", "barrid")
    parallel = backtest_parallel(data, constraints, n_cpus=4).sort("date", "barrid")
    assert parallel.equals(sequential)


def test_dynamic_parallel_matches_one_worker():
    data = panel()
    constraints = [sfo.ZeroInvestment(), sfo.LongOnly(), sfo.ZeroBeta()]
    one = dynamic_backtest_parallel(data, constraints, n_cpus=1)
    four = dynamic_backtest_parallel(data, constraints, n_cpus=4)
    assert four.equals(one)
    assert one["date"].unique().sort().to_list() == DATES
    assert one.select("date", "barrid").equals(
        data.select("date", "barrid").sort("date", "barrid")
    )
