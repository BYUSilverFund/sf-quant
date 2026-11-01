from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest
from sf_quant.simulation import SimulationConfig, rebalance_sessions, simulate_portfolio


def panel(dates, values, column="weight", assets=("A", "B")):
    return pl.DataFrame(
        {
            "date": [d for d in dates for _ in assets],
            "barrid": list(assets) * len(dates),
            column: np.array(values, dtype=float).ravel(),
        }
    )


D = [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9)]
NOW = SimulationConfig(execution_lag_sessions=0)


def run(targets=None, returns=None, config=NOW, benchmark=None):
    return simulate_portfolio(
        targets if targets is not None else panel(D[:1], [[0.5, 0.5]]),
        returns
        if returns is not None
        else panel(D[:3], [[0, 0], [-0.5, 0], [0, 0]], "return"),
        config=config,
        benchmark_weights=benchmark,
    )


def test_half_loss_exact_drift_and_terminal_wealth():
    result = run()
    weights = result.weights().filter(pl.col("date") == D[1])["weight"].to_numpy()
    np.testing.assert_allclose(weights, [1 / 3, 2 / 3], atol=1e-14)
    np.testing.assert_allclose(result.daily["nav"], [1, 0.75, 0.75], atol=1e-14)
    np.testing.assert_allclose(
        result.holdings.filter(pl.col("barrid") == "A")["units"], [0.5] * 3
    )


def test_unchanged_prices_and_zero_turnover():
    result = run(returns=panel(D, np.zeros((6, 2)), "return"))
    assert result.daily["nav"].to_list() == [1] * 6
    assert result.daily["two_sided_turnover"].to_list() == [1, 0, 0, 0, 0, 0]
    assert result.weights()["weight"].to_list() == [0.5] * 12


def test_drift_without_trades_has_zero_turnover():
    assert run().daily["two_sided_turnover"].to_list() == [1, 0, 0]


def test_rebalance_trade_dollars_and_turnover():
    result = run(targets=panel(D[:2], [[0.5, 0.5], [0.5, 0.5]]))
    trades = result.trades.filter(pl.col("date") == D[1])
    np.testing.assert_allclose(trades["traded_value"], [0.125, -0.125], atol=1e-14)
    assert result.daily["two_sided_turnover"][1] == pytest.approx(1 / 3)
    assert result.daily["one_way_turnover"][1] == pytest.approx(1 / 6)


def test_daily_analytic_returns_and_no_same_day_earnings():
    result = run(
        targets=panel(D[:3], [[0.5, 0.5]] * 3),
        returns=panel(D[:3], [[9, 2], [0.1, 0], [0, 0.2]], "return"),
    )
    np.testing.assert_allclose(result.daily["return"], [0, 0.05, 0.1], atol=1e-14)
    assert result.daily["nav"][-1] == pytest.approx(1.155)


@pytest.mark.parametrize("frequency", ["weekly", "monthly"])
def test_constant_units_between_calendar_trades(frequency):
    dates = [
        date(2023, 12, 28),
        date(2023, 12, 29),
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 8),
        date(2024, 2, 1),
    ]
    result = run(
        targets=panel(dates, [[0.5, 0.5]] * 6),
        returns=panel(dates, [[0.1, 0]] * 6, "return"),
        config=replace(NOW, frequency=frequency),
    )
    for d in result.daily.filter(~pl.col("rebalanced"))["date"]:
        rows = result.holdings.filter(pl.col("date") == d)
        np.testing.assert_array_equal(rows["pre_units"], rows["units"])
    expected = (
        [dates[0], dates[2], dates[4], dates[5]]
        if frequency == "weekly"
        else [dates[0], dates[2], dates[5]]
    )
    assert result.daily.filter(pl.col("rebalanced"))["date"].to_list() == expected


def test_year_boundary_does_not_reset_holdings():
    dates = [date(2023, 12, 29), date(2024, 1, 2)]
    result = run(
        panel(dates[:1], [[0.5, 0.5]]), panel(dates, [[0, 0], [0.1, 0]], "return")
    )
    assert result.daily["nav"][-1] == pytest.approx(1.05)
    assert result.daily["two_sided_turnover"][-1] == 0


def test_active_conversion_and_benchmark_only_assets():
    bm = panel(D[:3], [[0.5, 0.5]] * 3)
    targets = panel(D[:1], [[0.1]], assets=("A",))
    result = run(
        targets,
        config=replace(NOW, weight_type="active", allow_short=True),
        benchmark=bm,
    )
    np.testing.assert_allclose(result.weights()["weight"][:2], [0.6, 0.5])
    # Holdings 0.6 A + 0.5 B funded by -0.1 cash: loss = 0.3.
    assert result.daily["return"][1] == pytest.approx(-0.3)
    assert result.daily["benchmark_return"][1] == pytest.approx(-0.25)
    assert result.daily["active_return"][1] == pytest.approx(-0.05)


def test_zero_active_target_does_not_stay_zero_with_drift():
    result = run(
        targets=panel(D[:1], [[0, 0]]),
        config=replace(NOW, weight_type="active"),
        benchmark=panel(D[:3], [[0.5, 0.5]] * 3),
    )
    h = result.holdings.filter(pl.col("date") == D[1])["active_weight"].to_numpy()
    np.testing.assert_allclose(h, [-1 / 6, 1 / 6], atol=1e-14)
    assert h @ np.array([2.0, 1.0]) == pytest.approx(-1 / 6)


def test_benchmark_uses_previous_weights_not_end_weights():
    bm = panel(D[:3], [[0.5, 0.5], [0.9, 0.1], [0.5, 0.5]])
    result = run(benchmark=bm)
    assert result.daily["benchmark_return"][1] == -0.25


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_held_returns_fail(missing):
    returns = panel(D[:3], [[0, 0], [missing, 0], [0, 0]], "return")
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        run(returns=returns)


def test_delisting_requires_explicit_total_loss_and_cannot_rebuy():
    returns = panel(D[:3], [[0, 0], [-1, 0], [None, 0]], "return")
    result = run(returns=returns)
    assert result.daily["nav"][-1] == 0.5
    with pytest.raises(ValueError, match="Cannot trade"):
        run(panel(D[:2], [[0.5, 0.5]] * 2), returns)


def test_missing_entire_session_not_silently_skipped():
    returns = panel([D[0], D[2]], [[0, 0], [0, 0]], "return")
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        simulate_portfolio(
            panel(D[:1], [[0.5, 0.5]]), returns, sessions=D[:3], config=NOW
        )


def test_default_lag_no_lookahead_no_double_shift():
    result = run(
        returns=panel(D[:3], [[4, 0], [2, 0], [0.2, 0]], "return"),
        config=SimulationConfig(),
    )
    np.testing.assert_allclose(result.daily["return"], [0, 0, 0.1], atol=1e-14)
    assert result.daily["decision_date"][1] == D[0]


def test_signed_cash_financing_accounting():
    result = run(
        panel(D[:1], [[1.3, -0.3]]),
        panel(D[:3], [[0, 0], [0.1, -0.2], [0, 0]], "return"),
        replace(NOW, allow_short=True),
    )
    assert result.daily["nav"][1] == pytest.approx(1.19)
    assert result.daily["cash"][1] == pytest.approx(0)
    funded = run(
        panel(D[:1], [[1.5, 0]]),
        panel(D[:3], [[0, 0]] * 3, "return"),
        replace(NOW, allow_short=True, cash_borrow_return=0.01),
    )
    assert funded.daily["nav"][1] == pytest.approx(0.995)


def test_transaction_cost_self_financing_and_target_post_nav():
    result = run(config=replace(NOW, transaction_cost_bps=100))
    assert result.daily["nav"][0] == pytest.approx(1 / 1.01)
    assert result.daily["transaction_cost"][0] == pytest.approx(0.01 / 1.01)
    for row in result.daily.iter_rows(named=True):
        assert row["nav"] == pytest.approx(
            row["beginning_nav"]
            + row["investment_pnl"]
            + row["financing_pnl"]
            - row["transaction_cost"],
            abs=1e-14,
        )


def test_independent_buy_hold_wealth():
    returns = np.array([[0, 0], [0.1, -0.2], [-0.05, 0.3], [0.2, -0.1]])
    result = run(returns=panel(D[:4], returns, "return"))
    assert result.daily["nav"][-1] == pytest.approx(
        0.5 * np.prod(1 + returns[:, 0]) + 0.5 * np.prod(1 + returns[:, 1]), abs=1e-14
    )


def test_input_validation_and_output_preservation(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        run(targets=panel([D[0], D[0]], [[0.5, 0.5]] * 2))
    with pytest.raises(ValueError, match="benchmark"):
        run(config=replace(NOW, weight_type="active"))
    with pytest.raises(ValueError, match="allow_short"):
        run(targets=panel(D[:1], [[1.1, -0.1]]))
    result = run()
    result.save(tmp_path / "result")
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "result")


def test_last_session_schedule_holidays():
    assert rebalance_sessions(D, "weekly", "last") == {D[3], D[5]}


def test_universe_exit_retains_holding_until_scheduled_trade():
    # B exits the target universe on Wednesday, but is still valued until Monday.
    targets = pl.concat([panel(D[:1], [[.5, .5]]),
        panel(D[1:], [[1.]]*5, assets=("A",))])
    result = run(targets, panel(D, [[0, .1]]*6, "return"), replace(NOW, frequency="weekly"))
    b = result.holdings.filter(pl.col("barrid") == "B")
    assert b.filter(pl.col("date") == D[3])["units"][0] > 0
    assert b.filter(pl.col("date") == D[4])["units"][0] == 0
    assert result.trades.filter((pl.col("date") == D[4]) & (pl.col("barrid") == "B"))["traded_value"][0] < 0


def test_trade_statistics_and_leverage_use_realized_records():
    from sf_quant.performance import (
        generate_leverage_from_weights,
        get_trade_turnover_stats,
    )
    result = run()
    assert get_trade_turnover_stats(result.daily)["Mean Turnover"][0] == pytest.approx(1/3)
    np.testing.assert_allclose(generate_leverage_from_weights(result.weights())["leverage"], [1, 1, 1])


def test_legacy_weighted_return_interfaces_keep_same_date_semantics(monkeypatch):
    from sf_quant.performance import returns as legacy

    returns = panel(D[:2], [[10, 0], [0, 20]], "return")
    weights = panel(D[:2], [[0.5, 0.5]] * 2)
    monkeypatch.setattr(legacy, "load_assets", lambda **kw: returns)
    monkeypatch.setattr(legacy, "load_benchmark", lambda **kw: weights)
    with pytest.warns(RuntimeWarning, match="ignores portfolio drift"):
        single = legacy.legacy_generate_returns_from_weights(weights)
    np.testing.assert_allclose(single["return"], [0.05, 0.1])
    with pytest.warns(RuntimeWarning, match="ignores portfolio drift"):
        multi = legacy.legacy_generate_multi_returns_from_weights(weights)
    np.testing.assert_allclose(
        multi.filter(pl.col("portfolio") == "active")["return"], [0, 0]
    )


@pytest.mark.parametrize("old_name", [
    "generate_returns_from_weights", "generate_multi_returns_from_weights",
])
def test_retired_return_names_cannot_be_called(old_name):
    import sf_quant.performance as performance
    from sf_quant.performance import returns as legacy

    assert old_name not in performance.__all__
    for module in (performance, legacy):
        with pytest.raises(AttributeError, match="Use simulate_portfolio"):
            getattr(module, old_name)
    assert callable(getattr(performance, "legacy_" + old_name))
    assert performance.simulate_portfolio is simulate_portfolio


@pytest.mark.parametrize("module_name,function_name", [
    ("tables", "generate_returns_summary_table"),
    ("tables", "generate_multi_returns_summary_table"),
    ("tables", "generate_drawdown_summary_table"),
    ("drawdown", "generate_drawdown_from_returns"),
])
def test_reporting_examples_execute_simulation(module_name, function_name):
    import doctest
    from importlib import import_module

    module = import_module("sf_quant.performance." + module_name)
    docstring = getattr(module, function_name).__doc__
    assert "generate_returns_from_weights" not in docstring
    assert "simulate_portfolio" in docstring
    example = doctest.DocTestParser().get_doctest(
        docstring, {}, function_name, module.__file__, 0)
    outcome = doctest.DocTestRunner().run(example)
    assert outcome.failed == 0
