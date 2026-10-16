"""IC alignment tests. Synthetic frames only, so these need no data lake."""

import datetime as dt

import polars as pl
import pytest

from sf_quant.performance import generate_alpha_ics

DATES = [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8)]


def frame(values: dict[str, list[float]], column: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": DATES * len(values),
            "barrid": [b for b in values for _ in DATES],
            column: [v for b in values for v in values[b]],
        }
    )


def ic_on(alphas, rets, date, **kwargs):
    out = generate_alpha_ics(alphas, rets, **kwargs)
    return out.filter(pl.col("date") == date)["ic"].item()


def test_formation_day_return_is_excluded():
    """Day 0's return ranks A > B > C, the next day's C > B > A."""
    alphas = frame({"A": [3.0] * 5, "B": [2.0] * 5, "C": [1.0] * 5}, "alpha")
    rets = frame(
        {
            "A": [0.03, -0.01, 0.0, 0.0, 0.0],
            "B": [0.02, 0.0, 0.0, 0.0, 0.0],
            "C": [0.01, 0.01, 0.0, 0.0, 0.0],
        },
        "return",
    )
    assert ic_on(alphas, rets, DATES[0], method="rank", window=1) == pytest.approx(-1.0)
    assert ic_on(alphas, rets, DATES[0], method="pearson", window=1) == pytest.approx(-1.0)


def test_dates_without_a_full_forward_window_are_dropped():
    """The last ``window`` dates have no t+1..t+window return, so no IC."""
    alphas = frame({"A": [1.0] * 5, "B": [0.0] * 5}, "alpha")
    rets = frame({"A": [0.01] * 5, "B": [0.02] * 5}, "return")
    out = generate_alpha_ics(alphas, rets, method="pearson", window=2)
    assert out["date"].max() == DATES[2]
