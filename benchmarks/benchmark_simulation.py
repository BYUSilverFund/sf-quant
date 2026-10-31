"""Measured synthetic scale sample, not a claim about full-history production."""
import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from sf_quant.simulation import SimulationConfig, simulate_portfolio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, default=1000)
    parser.add_argument("--sessions", type=int, default=252)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dates = []
    d = date(2023, 12, 1)
    while len(dates) < args.sessions:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    keys = {"date": [d for d in dates for _ in range(args.assets)],
            "barrid": [f"A{i:05d}" for i in range(args.assets)]*len(dates)}
    rng = np.random.default_rng(101)
    returns = pl.DataFrame({**keys, "return": rng.normal(0, .01, len(dates)*args.assets)})
    targets = pl.DataFrame({**keys, "weight": np.full(len(dates)*args.assets, 1/args.assets)})
    start = time.perf_counter()
    result = simulate_portfolio(targets, returns, config=SimulationConfig(frequency="monthly"))
    elapsed = time.perf_counter() - start
    output = {"assets": args.assets, "sessions": len(dates), "input_rows": returns.height,
        "seconds": elapsed, "holdings_rows": result.holdings.height,
        "result_mebibytes": sum(getattr(result, f).estimated_size() for f in ("daily", "holdings", "trades"))/2**20,
        "max_accounting_residual": result.daily["accounting_residual"].abs().max(),
        "polars_version": pl.__version__, "numpy_version": np.__version__}
    print(json.dumps(output, indent=2))
    if args.output:
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
