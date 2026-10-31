"""Deterministic close-to-close execution, separate from portfolio construction.

Inputs use decimal total returns labelled by interval END. Positions are units
of a reinvested total-return index (not unadjusted security shares). No return
shifts, universe filtering, or missing-return imputation occur here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SimulationConfig:
    frequency: Literal["daily", "weekly", "monthly"] = "daily"
    rebalance_day: Literal["first", "last"] = "first"
    weight_type: Literal["total", "active"] = "total"
    execution_lag_sessions: int = 1
    initial_nav: float = 1.0
    transaction_cost_bps: float = 0.0
    allow_short: bool = False
    cash_lending_return: float = 0.0
    cash_borrow_return: float = 0.0
    short_fee_per_session: float = 0.0
    missing_return_policy: Literal["error"] = "error"

    def __post_init__(self):
        if self.frequency not in ("daily", "weekly", "monthly"):
            raise ValueError("frequency must be daily, weekly, or monthly")
        if self.rebalance_day not in ("first", "last"):
            raise ValueError("rebalance_day must be first or last")
        if self.weight_type not in ("active", "total"):
            raise ValueError("weight_type must be active or total")
        if (
            type(self.execution_lag_sessions) is not int
            or self.execution_lag_sessions < 0
        ):
            raise ValueError("execution_lag_sessions must be a nonnegative integer")
        if not np.isfinite(self.initial_nav) or self.initial_nav <= 0:
            raise ValueError("initial_nav must be finite and positive")
        for name in (
            "transaction_cost_bps",
            "cash_lending_return",
            "cash_borrow_return",
            "short_fee_per_session",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.missing_return_policy != "error":
            raise ValueError(
                "Only explicit failure on missing held returns is supported"
            )


@dataclass
class SimulationResult:
    daily: pl.DataFrame
    holdings: pl.DataFrame
    trades: pl.DataFrame
    config: SimulationConfig

    def returns(self, portfolio: str = "total") -> pl.DataFrame:
        """Analytics-compatible return table, including entry costs on inception."""
        column = {
            "total": "return",
            "benchmark": "benchmark_return",
            "active": "active_return",
        }[portfolio]
        if portfolio != "total" and self.daily[column].null_count():
            raise ValueError("Benchmark compositions are required for benchmark/active returns")
        return self.daily.select("date", pl.col(column).alias("return"))

    def multi_returns(self) -> pl.DataFrame:
        return pl.concat(
            [
                self.returns(p).with_columns(pl.lit(p).alias("portfolio"))
                for p in ("total", "benchmark", "active")
            ]
        )

    def weights(self) -> pl.DataFrame:
        return self.holdings.select(
            "date", "barrid", pl.col("post_weight").alias("weight")
        )

    def save(self, output: str | Path) -> None:
        """Write a NEW directory; never replace existing research results."""
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        temporary = output.with_name(output.name + ".partial-" + uuid4().hex)
        temporary.mkdir(parents=True, exist_ok=False)
        for name in ("daily", "holdings", "trades"):
            getattr(self, name).write_parquet(temporary / f"{name}.parquet")
        (temporary / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2)
        )
        temporary.rename(output)


def _panel(frame: pl.DataFrame, value: str, name: str) -> pl.DataFrame:
    required = {"date", "barrid", value}
    if not required.issubset(frame.columns) or frame.is_empty():
        raise ValueError(f"{name} requires nonempty columns {sorted(required)}")
    frame = frame.select("date", "barrid", value)
    if frame["date"].dtype != pl.Date or frame["barrid"].dtype != pl.String:
        raise ValueError(f"{name}: date must be pl.Date and barrid pl.String")
    if any(frame[c].null_count() for c in ("date", "barrid")):
        raise ValueError(f"{name}: null keys")
    if frame.select(pl.struct("date", "barrid").is_duplicated().any()).item():
        raise ValueError(f"{name}: duplicate date/barrid")
    if value != "return" and (
        frame[value].null_count() or not frame[value].is_finite().all()
    ):
        raise ValueError(f"{name}: nonfinite weights")
    return frame.sort("date", "barrid")


def rebalance_sessions(
    sessions: Sequence[date], frequency: str, day: str = "first"
) -> set[date]:
    """Calendar groups over supplied sessions; 'last' requires complete period data."""
    if frequency == "daily":
        return set(sessions)
    if frequency not in ("weekly", "monthly") or day not in ("first", "last"):
        raise ValueError("Invalid rebalance schedule")
    groups = {}
    for d in sessions:
        key = d.isocalendar()[:2] if frequency == "weekly" else (d.year, d.month)
        if day == "last" or key not in groups:
            groups[key] = d
    return set(groups.values())


def simulate_portfolio(
    targets: pl.DataFrame,
    asset_returns: pl.DataFrame,
    *,
    benchmark_weights: pl.DataFrame | None = None,
    sessions: Sequence[date] | None = None,
    config: SimulationConfig | None = None,
    output_path: str | Path | None = None,
) -> SimulationResult:
    """Execute complete target snapshots; omitted assets are zero target weights.

    Each session values OLD units first, then executes the latest decision whose
    session index is at least execution_lag_sessions in the past. Lag zero is
    appropriate only for decisions available before that close. Initial cash is
    invested at the first eligible session (also for weekly/monthly schedules).
    Benchmark snapshots must be complete, fully invested, long-only compositions
    for every session. Its daily return uses the previous close's composition.

    Short/leveraged mode is explicitly opt-in: unrestricted signed cash, fixed
    per-session financing/borrow rates, no margin, recall, collateral or impact.
    Total returns assume distributions reinvested in adjusted units. A -1 return
    is terminal, writes off the holding, and prevents future purchases of that ID.
    Other delisting settlements must be incorporated by the caller in total returns.
    """
    cfg = config or SimulationConfig()
    targets = _panel(targets, "weight", "targets")
    asset_returns = _panel(asset_returns, "return", "asset_returns")
    bm = (
        _panel(benchmark_weights, "weight", "benchmark")
        if benchmark_weights is not None
        else None
    )
    if cfg.weight_type == "active" and bm is None:
        raise ValueError("Active targets require full benchmark compositions")
    dates = (
        list(sessions)
        if sessions is not None
        else asset_returns["date"].unique().sort().to_list()
    )
    if (
        not dates
        or dates != sorted(set(dates))
        or any(type(d) is not date for d in dates)
    ):
        raise ValueError("sessions must be nonempty, unique, increasing dates")
    date_set = set(dates)
    if not set(targets["date"]).issubset(date_set):
        raise ValueError(
            "Every target decision date must be a supplied trading session"
        )
    if bm is not None and not date_set.issubset(set(bm["date"])):
        raise ValueError("Missing benchmark composition on a trading session")
    assets = sorted(
        set(targets["barrid"]) | (set(bm["barrid"]) if bm is not None else set())
    )
    ids = np.array(assets)
    index = {a: i for i, a in enumerate(assets)}
    n = len(assets)

    def partitions(frame, column):
        result = {}
        for (d,), sub in frame.partition_by("date", as_dict=True).items():
            idx = np.array([index[a] for a in sub["barrid"]], dtype=int)
            result[d] = (idx, sub[column].to_numpy().astype(float))
        return result

    target_map = partitions(targets, "weight")
    return_map = partitions(
        asset_returns.filter(pl.col("barrid").is_in(assets)), "return"
    )
    bm_map = partitions(bm, "weight") if bm is not None else {}

    def dense(mapping, d):
        out = np.zeros(n)
        ix, val = mapping[d]
        out[ix] = val
        return out

    if bm is not None:
        for d in dates:
            b = dense(bm_map, d)
            if np.min(b) < 0 or not np.isclose(b.sum(), 1, atol=1e-8, rtol=0):
                raise ValueError(f"Benchmark must be fully invested and long-only: {d}")
    schedule = rebalance_sessions(dates, cfg.frequency, cfg.rebalance_day)
    prices = np.ones(n)
    units = np.zeros(n)
    cash = cfg.initial_nav
    previous_nav = cfg.initial_nav
    prev_b = np.zeros(n)
    latest = None
    last_executed = None
    started = False
    daily_rows, holding_frames, trade_frames = [], [], []
    cost_rate = cfg.transaction_cost_bps / 10000
    for t, d in enumerate(dates):
        available = t - cfg.execution_lag_sessions
        if available >= 0 and dates[available] in target_map:
            latest = dates[available]
        b = dense(bm_map, d) if bm is not None else np.zeros(n)
        r = np.full(n, np.nan)
        if d in return_map:
            ix, vals = return_map[d]
            r[ix] = vals
        required = (units != 0) | (prev_b != 0)
        missing = required & ~np.isfinite(r)
        if np.any(missing):
            raise ValueError(
                f"Missing/nonfinite held or benchmark return at {d}: {ids[missing][:10].tolist()}"
            )
        if np.any(np.isfinite(r) & (r < -1)) or np.any(np.isinf(r)):
            raise ValueError(f"Invalid total return below -100% or infinite at {d}")
        # Unheld, unobserved assets carry only an arbitrary index base; never P&L.
        valid = np.isfinite(r)
        before_values = units * prices
        gains = (
            float(before_values[required] @ r[required]) if np.any(required) else 0.0
        )
        financing = (
            0.0
            if t == 0
            else (
                cash
                * (cfg.cash_lending_return if cash >= 0 else cfg.cash_borrow_return)
                - float(np.abs(np.minimum(before_values, 0)).sum())
                * cfg.short_fee_per_session
            )
        )
        cash += financing
        prices[valid] *= 1 + r[valid]
        # Terminal zero-value positions have no subsequent missing-return requirement.
        units[prices == 0] = 0
        values = units * prices
        pre_nav = float(cash + values.sum())
        if not np.isfinite(pre_nav) or pre_nav <= 0:
            raise ValueError(f"Portfolio insolvent at {d}")
        benchmark_return = (
            float(prev_b[prev_b != 0] @ r[prev_b != 0]) if np.any(prev_b) else 0.0
        )
        pre_units = units.copy()
        pre_weights = values / pre_nav
        pre_cash = cash
        execute = (
            latest is not None
            and latest != last_executed
            and (d in schedule or not started)
        )
        cost, traded = 0.0, np.zeros(n)
        if execute:
            w = dense(target_map, latest)
            if cfg.weight_type == "active":
                w += dense(bm_map, latest)
            # Retain numerical optimizer tolerances as explicit residual cash.
            if not cfg.allow_short and (w.min() < -1e-7 or w.sum() > 1 + 1e-7):
                raise ValueError(
                    "Short positions or borrowing require allow_short=True"
                )
            if np.any((w != 0) & ((prices == 0) | ~np.isfinite(r))):
                raise ValueError(
                    f"Cannot trade an asset without a current valid mark at {d}"
                )
            if cost_rate * np.abs(w).sum() >= 1:
                raise ValueError(
                    "Transaction cost times target gross exposure must be below one"
                )
            post_nav = pre_nav
            if cost_rate:
                lo, hi = 0.0, pre_nav
                if cost_rate * np.abs(values).sum() >= pre_nav:
                    raise ValueError("Liquidation costs exhaust NAV")
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if mid + cost_rate * np.abs(w * mid - values).sum() > pre_nav:
                        hi = mid
                    else:
                        lo = mid
                post_nav = (lo + hi) / 2
            new_values = w * post_nav
            traded = new_values - values
            cost = cost_rate * float(np.abs(traded).sum())
            cash = pre_cash - float(traded.sum()) - cost
            units = np.divide(new_values, prices, out=np.zeros(n), where=prices != 0)
            values = new_values
            started = True
            last_executed = latest
        nav = float(cash + values.sum())
        residual = nav - (previous_nav + gains + financing - cost)
        if not np.isclose(residual, 0, atol=1e-10 * max(1, abs(previous_nav)), rtol=0):
            raise ArithmeticError(f"Accounting does not reconcile at {d}: {residual}")
        turnover = float(np.abs(traded).sum()) / pre_nav
        total_return = nav / previous_nav - 1
        daily_rows.append(
            dict(
                date=d,
                beginning_nav=previous_nav,
                investment_pnl=gains,
                financing_pnl=financing,
                pre_nav=pre_nav,
                nav=nav,
                cash=cash,
                pre_cash=pre_cash,
                transaction_cost=cost,
                **{"return": total_return},
                benchmark_return=benchmark_return if bm is not None else None,
                active_return=total_return - benchmark_return
                if bm is not None
                else None,
                rebalanced=execute,
                decision_date=latest if execute else None,
                two_sided_turnover=turnover,
                one_way_turnover=turnover / 2,
                accounting_residual=residual,
            )
        )
        keep = (pre_units != 0) | (units != 0) | (b != 0)
        k = int(keep.sum())
        if k:
            holding_frames.append(
                pl.DataFrame(
                    dict(
                        date=[d] * k,
                        barrid=ids[keep],
                        pre_units=pre_units[keep],
                        units=units[keep],
                        index_price=prices[keep],
                        pre_weight=pre_weights[keep],
                        post_weight=values[keep] / nav,
                        position_value=values[keep],
                        benchmark_weight=b[keep]
                        if bm is not None
                        else np.full(k, np.nan),
                        active_weight=values[keep] / nav - b[keep]
                        if bm is not None
                        else np.full(k, np.nan),
                    )
                )
            )
        moved = traded != 0
        k = int(moved.sum())
        if k:
            trade_frames.append(
                pl.DataFrame(
                    dict(
                        date=[d] * k,
                        barrid=ids[moved],
                        traded_value=traded[moved],
                        traded_units=units[moved] - pre_units[moved],
                        pre_nav=np.full(k, pre_nav),
                        cost=cost_rate * np.abs(traded[moved]),
                    )
                )
            )
        previous_nav, prev_b = nav, b
    trade_schema = {
        "date": pl.Date,
        "barrid": pl.String,
        "traded_value": pl.Float64,
        "traded_units": pl.Float64,
        "pre_nav": pl.Float64,
        "cost": pl.Float64,
    }
    holding_schema = {
        "date": pl.Date,
        "barrid": pl.String,
        **{
            c: pl.Float64
            for c in (
                "pre_units",
                "units",
                "index_price",
                "pre_weight",
                "post_weight",
                "position_value",
                "benchmark_weight",
                "active_weight",
            )
        },
    }
    result = SimulationResult(
        pl.DataFrame(daily_rows),
        pl.concat(holding_frames)
        if holding_frames
        else pl.DataFrame(schema=holding_schema),
        pl.concat(trade_frames) if trade_frames else pl.DataFrame(schema=trade_schema),
        cfg,
    )
    if output_path is not None:
        result.save(output_path)
    return result
