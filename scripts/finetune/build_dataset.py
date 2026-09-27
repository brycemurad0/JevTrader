#!/usr/bin/env python
"""Build a Kev-format fine-tune dataset (train/calibration/dev/test JSONL, chronological,
embargoed) from a bars CSV, using EXACTLY `jevtrader.jev.state.build_state` and
`jevtrader.jev.questions` -- no train/serve skew.

Example (Bitstamp BTC/USD 1-minute bars, Alpaca crypto taker round-trip cost, 5-minute horizon):

    . .venv/bin/activate
    python scripts/finetune/build_dataset.py \\
        --bars data/cache/btcusd_bitstamp_1min_2025_2026.csv --symbol BTC/USD \\
        --horizon 5min --cost-bps 50 --lookback-bars 120 \\
        --out data/cache/finetune/btcusd_5min_taker50bps

The bars CSV must have columns `timestamp` (unix seconds, bar OPEN time), `open`, `high`, `low`,
`close`, `volume` -- the shape most exchanges (incl. Bitstamp) export 1-minute history in.

Writes `train.jsonl`, `calibration.jsonl`, `dev.jsonl`, `test.jsonl` (Kev-format labelled
requests, each line also carrying a `"_meta"` key with the realized outcome -- ignored by
`kev.train`/Laya's loader, read back by `jevtrader.jev.finetune.dataset.load_examples_jsonl`
for `scoreboard.py`) and a `manifest.json` (config used + example/class counts) to `--out`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `jevtrader` imports without an install

from jevtrader.jev.finetune.dataset import DatasetConfig, build_dataset, write_jsonl  # noqa: E402


def load_bars_csv(path: str, *, ts_col: str = "timestamp", ts_unit: str = "s") -> pd.DataFrame:
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"{path}: expected a {ts_col!r} column, got {list(df.columns)}")
    df[ts_col] = pd.to_datetime(df[ts_col], unit=ts_unit, utc=True)
    df = df.set_index(ts_col).sort_index()
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing OHLCV column(s) {missing}")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _label_counts(examples: list[dict], key: str = "direction") -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        label = ex["labels"].get(key)
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bars", required=True, help="CSV of OHLCV bars")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--horizon", default="5min", help="pandas-timedelta string, e.g. 5min, 15min, 1h")
    ap.add_argument("--cost-bps", type=float, default=0.0, help="round-trip cost hurdle in bps (0 = no-cost variant)")
    ap.add_argument("--lookback-bars", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1, help="sample every Nth bar")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--no-regime", action="store_true", help="skip the regime/risk_off heuristic labels")
    ap.add_argument("--no-entry-quality", action="store_true")
    ap.add_argument("--balance-train", action="store_true", help="undersample train split to equal class counts (direction)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--calibration-frac", type=float, default=0.1)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    bars = load_bars_csv(args.bars)
    config = DatasetConfig(
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        lookback_bars=args.lookback_bars,
        stride=args.stride,
        include_regime=not args.no_regime,
        include_entry_quality=not args.no_entry_quality,
        max_examples=args.max_examples,
        train_frac=args.train_frac,
        calibration_frac=args.calibration_frac,
        dev_frac=args.dev_frac,
        test_frac=args.test_frac,
        balance_train_classes=args.balance_train,
        seed=args.seed,
    )
    splits = build_dataset(args.symbol, bars, config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    label_counts = {}
    for name, examples in (("train", splits.train), ("calibration", splits.calibration), ("dev", splits.dev), ("test", splits.test)):
        counts[name] = write_jsonl(examples, out_dir / f"{name}.jsonl", source=f"jevtrader:{args.symbol}")
        label_counts[name] = _label_counts(examples)

    manifest = {
        "symbol": args.symbol,
        "bars_csv": str(Path(args.bars).resolve()),
        "n_bars": int(len(bars)),
        "config": {
            "horizon": config.horizon,
            "cost_bps": config.cost_bps,
            "lookback_bars": config.lookback_bars,
            "stride": config.stride,
            "include_regime": config.include_regime,
            "include_entry_quality": config.include_entry_quality,
            "balance_train_classes": config.balance_train_classes,
            "train_frac": config.train_frac,
            "calibration_frac": config.calibration_frac,
            "dev_frac": config.dev_frac,
            "test_frac": config.test_frac,
            "seed": config.seed,
        },
        "example_counts": counts,
        "direction_label_counts": label_counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
