"""Loaders for the cached real-data CSVs under `data/cache/` (git-ignored; never committed).

Every loader returns a UTC-tz-indexed DataFrame with the canonical columns
`open, high, low, close, volume` (plus `vwap` = NaN placeholder so `ctx.bars()` shapes match),
indexed by **bar CLOSE time** -- the convention `jevtrader.core.types.Bar.ts` uses and the one
the `SimBroker` bar-mode fill model relies on for its no-lookahead rule.

Data sets and their quirks (read before trusting a number):

* `btcusd_bitstamp_1min_2025_2026.csv` -- Bitstamp BTC/USD, 1-min. The `timestamp` column is the
  bar's OPEN time in unix seconds, so we add one minute. Bitstamp is a real venue with real
  volume; Alpaca crypto prices track it closely but Alpaca's spread is typically wider. Gaps
  (exchange downtime) are left as gaps, not forward-filled.
* `SPX500_1m_sample.csv` -- an S&P 500 index CFD, near-24h. Used as a proxy for SPY intraday
  behaviour. No spread information; `volume` is CFD ticks, not shares. Equity strategies restrict
  themselves to the US cash session (13:30-20:00 UTC) via `us_session_only`.
* `C_1m_sample.csv` -- Citigroup, 1-min, US session. `volume` looks like a sampled feed; do not
  use absolute volume (relative volume ratios are OK).
* `GOOG_1d.csv`, `BTCUSD_1d_sample.csv` -- daily samples from backtesting.py (capitalised
  columns, monthly-ish for BTC). Small; useful for smoke tests only.

`chrono_split` implements the non-negotiable methodology: design/tune on the first 60%, validate
on the next 20%, and touch the final 20% once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from jevtrader.core.types import Bar

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

BTC_1M = "btcusd_bitstamp_1min_2025_2026.csv"
SPX_1M = "SPX500_1m_sample.csv"
C_1M = "C_1m_sample.csv"
GOOG_1D = "GOOG_1d.csv"
BTC_1D = "BTCUSD_1d_sample.csv"

COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    if "vwap" not in df.columns:
        df["vwap"] = np.nan
    df.index.name = "ts"
    return df[COLUMNS].astype(float)


def load_btc_1m(path: Optional[Path] = None) -> pd.DataFrame:
    """Bitstamp BTC/USD 1-min bars, indexed by bar close (open time + 60s)."""
    p = Path(path) if path else CACHE_DIR / BTC_1M
    df = pd.read_csv(p)
    idx = pd.to_datetime(df["timestamp"].astype("int64") + 60, unit="s", utc=True)
    df = df.drop(columns=["timestamp"]).set_index(idx)
    return _finish(df)


def _load_iso_1m(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    idx = pd.to_datetime(df["datetime"], utc=True) + pd.Timedelta(minutes=1)
    df = df.drop(columns=["datetime"]).set_index(idx)
    return _finish(df)


def load_spx_1m(path: Optional[Path] = None) -> pd.DataFrame:
    """SPX500 CFD 1-min bars (near 24h), indexed by bar close."""
    return _load_iso_1m(Path(path) if path else CACHE_DIR / SPX_1M)


def load_c_1m(path: Optional[Path] = None) -> pd.DataFrame:
    """Citigroup 1-min bars (US session), indexed by bar close."""
    return _load_iso_1m(Path(path) if path else CACHE_DIR / C_1M)


def _load_daily(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC") + pd.Timedelta(hours=21)  # 16:00 ET close-ish
    return _finish(df)


def load_goog_1d(path: Optional[Path] = None) -> pd.DataFrame:
    return _load_daily(Path(path) if path else CACHE_DIR / GOOG_1D)


def load_btc_1d(path: Optional[Path] = None) -> pd.DataFrame:
    return _load_daily(Path(path) if path else CACHE_DIR / BTC_1D)


def available() -> dict[str, bool]:
    """Which cached files exist on this machine (tests skip gracefully when they don't)."""
    return {name: (CACHE_DIR / name).exists() for name in (BTC_1M, SPX_1M, C_1M, GOOG_1D, BTC_1D)}


# ----------------------------------------------------------------------------- transforms


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV resample. Because the index is bar CLOSE time, we label/close on the right so a
    5-minute bar stamped 10:05 contains the 1-min bars closing 10:01..10:05 -- nothing after."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df[list(agg)].resample(rule, label="right", closed="right").agg(agg).dropna(subset=["close"])
    out["vwap"] = np.nan
    return out


def us_session_only(df: pd.DataFrame, start: str = "13:31", end: str = "20:00") -> pd.DataFrame:
    """Keep bars whose close time falls within the US cash session (UTC; 13:30-20:00 = 9:30-16:00 ET
    outside DST shifts -- close enough for a proxy set; C_1m is already session-only)."""
    t = df.index.tz_convert("UTC")
    minutes = t.hour * 60 + t.minute
    lo = int(start[:2]) * 60 + int(start[3:])
    hi = int(end[:2]) * 60 + int(end[3:])
    mask = (minutes >= lo) & (minutes <= hi) & (t.dayofweek < 5)
    return df[mask]


@dataclass(frozen=True)
class ChronoSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    holdout: pd.DataFrame

    def describe(self) -> str:
        def rng(d: pd.DataFrame) -> str:
            return f"{d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d} ({len(d):,} bars)" if len(d) else "empty"

        return f"train {rng(self.train)} | val {rng(self.val)} | holdout {rng(self.holdout)}"


def chrono_split(df: pd.DataFrame, train: float = 0.6, val: float = 0.2) -> ChronoSplit:
    """Chronological 60/20/20 split by row count. The holdout is to be evaluated ONCE per strategy."""
    n = len(df)
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return ChronoSplit(df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:])


def to_bar_events(df: pd.DataFrame, symbol: str) -> Iterator[Bar]:
    """DataFrame -> `Bar` events for the event-driven Backtester (fast itertuples path)."""
    has_vwap = "vwap" in df.columns
    for row in df.itertuples(index=True):
        vwap = getattr(row, "vwap", None) if has_vwap else None
        yield Bar(
            symbol=symbol,
            ts=row.Index,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
            vwap=float(vwap) if vwap is not None and not (isinstance(vwap, float) and np.isnan(vwap)) else None,
        )
