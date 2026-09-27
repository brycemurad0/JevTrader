#!/usr/bin/env python
"""Rolling-origin, no-look-ahead evaluation of the honest forecast baselines (and, on a machine
that has TimesFM installed, TimesFM itself) against real bars.

Default data: `data/cache/btcusd_bitstamp_1min_2025_2026.csv` (1-minute BTC/USD bars, Jan 2025 -
Sep 2026), columns `timestamp` (unix open time), `open, high, low, close, volume`.

This container has no GPU and cannot reach huggingface.co, so it can only run the baselines
(`random_walk`, `ewma`, `garch`) -- see docs/FORECASTING.md for the resulting numbers. On your
own machine, once you've `pip install timesfm[torch]`'d and the weights have downloaded:

    python scripts/forecast/eval_real.py --model timesfm2.5

Usage:

    python scripts/forecast/eval_real.py                                   # all 3 baselines
    python scripts/forecast/eval_real.py --model garch --model ewma
    python scripts/forecast/eval_real.py --horizons 5 15 60 --cost-bps 5
    python scripts/forecast/eval_real.py --out docs/_forecast_eval.md       # also write a file
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jevtrader.forecast.baselines import EWMAVolForecaster, GARCH11Forecaster, RandomWalkForecaster  # noqa: E402
from jevtrader.forecast.evaluate import EvalResult, evaluate_returns, evaluate_volatility, to_markdown  # noqa: E402

DEFAULT_CSV = _REPO_ROOT / "data" / "cache" / "btcusd_bitstamp_1min_2025_2026.csv"

# which targets each model can honestly be evaluated on -- passing a vol-only model through
# evaluate_returns (or vice versa) would score it on a target it never claims to forecast.
_BASELINE_TARGETS: Dict[str, set[str]] = {
    "random_walk": {"return"},
    "ewma": {"vol"},
    "garch": {"vol"},
    "timesfm2.5": {"return", "vol"},
}


def _load_csv_bars(path: Path) -> pd.DataFrame:
    """Same convention `jevtrader.cli._load_csv_bars` uses: `timestamp` is the bar's *open*
    time (unix seconds); the bar closes one minute later."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], unit="s", utc=True) + pd.Timedelta(minutes=1)
    else:
        col = "datetime" if "datetime" in df.columns else df.columns[0]
        idx = pd.to_datetime(df[col], utc=True)
    df = df.copy()
    df.index = idx
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols].sort_index()


def _build_model(name: str):
    if name == "timesfm2.5":
        from jevtrader.core.broker import TradingMode
        from jevtrader.forecast.timesfm_backend import TimesFMForecaster

        return TimesFMForecaster(version="2.5", mode=TradingMode.BACKTEST)
    if name == "random_walk":
        return RandomWalkForecaster()
    if name == "ewma":
        return EWMAVolForecaster()
    if name == "garch":
        return GARCH11Forecaster()
    raise SystemExit(f"unknown --model {name!r}; choose from {sorted(_BASELINE_TARGETS)}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="bars CSV (timestamp,open,high,low,close,volume)")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        choices=sorted(_BASELINE_TARGETS),
        help="repeatable; default: random_walk, ewma, garch (all baselines)",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 15, 60], help="bars ahead")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="round-trip cost for the directional-hit-rate metric")
    parser.add_argument("--vol-window", type=int, default=20, help="realized-vol rolling window, bars")
    parser.add_argument("--min-train", type=int, default=1500, help="bars of history before the first rolling origin")
    parser.add_argument("--step", type=int, default=240, help="bars between rolling origins")
    parser.add_argument("--max-windows", type=int, default=300, help="cap on scored rolling-origin windows per (model, horizon)")
    parser.add_argument("--max-context", type=int, default=1000, help="cap on history bars fed to a single forecast call (bounds GARCH fit cost)")
    parser.add_argument("--rows", type=int, default=None, help="only use the first N rows of the CSV (debugging)")
    parser.add_argument("--out", type=Path, default=None, help="also write the combined Markdown report to this path")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    bars = _load_csv_bars(args.csv)
    if args.rows:
        bars = bars.iloc[: args.rows]

    models = args.models or ["random_walk", "ewma", "garch"]

    return_results: Dict[str, Dict[int, EvalResult]] = {}
    vol_results: Dict[str, Dict[int, EvalResult]] = {}

    for name in models:
        targets = _BASELINE_TARGETS[name]
        model = _build_model(name)
        if "return" in targets:
            return_results[name] = {}
        if "vol" in targets:
            vol_results[name] = {}
        for h in args.horizons:
            if "return" in targets:
                t0 = time.perf_counter()
                print(f"[{name}] returns horizon={h} ...", file=sys.stderr, end=" ", flush=True)
                return_results[name][h] = evaluate_returns(
                    bars,
                    model,
                    horizon=h,
                    cost_bps=args.cost_bps,
                    min_train=args.min_train,
                    step=args.step,
                    max_windows=args.max_windows,
                    max_context=args.max_context,
                )
                print(f"n={return_results[name][h].n_windows} ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)
            if "vol" in targets:
                t0 = time.perf_counter()
                print(f"[{name}] vol horizon={h} ...", file=sys.stderr, end=" ", flush=True)
                vol_results[name][h] = evaluate_volatility(
                    bars,
                    model,
                    horizon=h,
                    vol_window=args.vol_window,
                    min_train=args.min_train,
                    step=args.step,
                    max_windows=args.max_windows,
                    max_context=args.max_context,
                )
                print(f"n={vol_results[name][h].n_windows} ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

    parts = [
        f"# Forecast baseline evaluation -- `{args.csv.name}`",
        (
            f"Rows: {len(bars)}  |  cost_bps={args.cost_bps}  |  vol_window={args.vol_window}  |  "
            f"min_train={args.min_train}  |  step={args.step}  |  max_windows={args.max_windows}  |  "
            f"max_context={args.max_context}"
        ),
    ]
    if return_results:
        parts.append(to_markdown(return_results, title="Returns (cumulative, bps)"))
    if vol_results:
        parts.append(to_markdown(vol_results, title="Realized volatility"))
    report = "\n\n".join(parts)

    print()
    print(report)
    if args.out:
        args.out.write_text(report + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
