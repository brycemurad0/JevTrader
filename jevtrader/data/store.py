"""On-disk cache for historical market data.

Cache format: one file per `(symbol, timeframe, kind)` under `settings.data_dir`, holding the
full known history for that key (new fetches are merged in, not appended blindly, so re-running
a load with an overlapping range never duplicates rows). Parquet (via pyarrow) is used when
available; if pyarrow is not installed we fall back to gzip-compressed CSV, which round-trips
through this module's own (de)serialization so callers never need to know which backend is
active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import pyarrow  # noqa: F401

    _HAVE_PYARROW = True
except ImportError:  # pragma: no cover - environment dependent
    _HAVE_PYARROW = False


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "-").replace(" ", "_")


def cache_path(data_dir: Path, symbol: str, timeframe: str, kind: str = "bars") -> Path:
    ext = "parquet" if _HAVE_PYARROW else "csv.gz"
    return Path(data_dir) / kind / f"{_safe_symbol(symbol)}_{timeframe}.{ext}"


def save_cache(path: Path, df: pd.DataFrame) -> None:
    """Write `df` (must have a tz-aware DatetimeIndex) to `path`, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if _HAVE_PYARROW:
        out.to_parquet(path)
    else:
        # csv.gz fallback: preserve tz-aware index by writing ISO8601 strings explicitly.
        out = out.reset_index()
        idx_name = df.index.name or "ts"
        out = out.rename(columns={out.columns[0]: idx_name})
        out.to_csv(path, index=False, compression="gzip", date_format="%Y-%m-%dT%H:%M:%S.%f%z")


def load_cache(path: Path) -> Optional[pd.DataFrame]:
    """Read a cache file written by `save_cache`, or return None if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    if _HAVE_PYARROW and path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="gzip" if path.suffixes[-2:] == [".csv", ".gz"] else None)
        idx_col = df.columns[0]
        df[idx_col] = pd.to_datetime(df[idx_col], utc=True)
        df = df.set_index(idx_col)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def merge_and_save(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge `new_df` into whatever is already cached at `path` (new rows win on duplicate
    timestamps) and persist the result. Returns the merged DataFrame."""
    existing = load_cache(path)
    if existing is None or existing.empty:
        merged = new_df.sort_index()
    else:
        merged = pd.concat([existing, new_df]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    save_cache(path, merged)
    return merged


def load_bars(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str = "1Min",
    data_dir: Optional[Path] = None,
    fetch_fn=None,
) -> pd.DataFrame:
    """Load bars for `symbol` between `[start, end]`, using the on-disk cache first.

    If the cache fully covers `[start, end]` no network call is made. Otherwise, if `fetch_fn`
    is given (signature `fetch_fn(symbol, start, end, timeframe) -> pd.DataFrame`, e.g. one of
    `jevtrader.data.alpaca_history`'s loaders), it is called for the missing range, the result is
    merged into the cache, and the requested slice is returned. Without `fetch_fn`, only what is
    already cached is returned (sliced to the requested range) -- this keeps the function usable
    fully offline.
    """
    from jevtrader.config import load_settings

    data_dir = Path(data_dir) if data_dir is not None else load_settings().data_dir
    path = cache_path(data_dir, symbol, timeframe, kind="bars")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    cached = load_cache(path)
    covers = cached is not None and not cached.empty and cached.index.min() <= start and cached.index.max() >= end

    if not covers and fetch_fn is not None:
        fetched = fetch_fn(symbol, start, end, timeframe)
        if fetched is not None and not fetched.empty:
            cached = merge_and_save(path, fetched)

    if cached is None or cached.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return cached.loc[(cached.index >= start) & (cached.index <= end)]
