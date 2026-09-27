"""Compact, deterministic, JSON-serializable state builder for Jev.

Jev ("System One") pays for every input token and reasons better over a handful of
pre-computed, rounded numbers than over raw tick arrays. `build_state` turns a bars
DataFrame (+ optional quote/order book/position) into a small flat dict: returns over a
few lookbacks, realized vol, a mean-reversion z-score, RSI, ATR%, a volume ratio, spread
and book-imbalance microstructure features, a coarse session/time-of-day bucket, a trend
slope, position/PnL context, and the round-trip trading cost in bps (so Jev's yes/no on
"does this move clear costs" is grounded in the same number the strategy uses).

Everything here is a pure function of its inputs: same bars -> same dict, always. That
determinism is what makes the TTL cache in `advisor.py` (keyed off a hash of this dict)
and the offline/replay advisors safe, and what makes tests reproducible.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from jevtrader.core.types import OrderBook, Position, Quote

DEFAULT_LOOKBACKS: tuple[int, ...] = (5, 15, 30, 60)


def _round(value: Optional[float], ndigits: int) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def _ret_bps(closes: pd.Series, lookback: int) -> Optional[float]:
    if len(closes) <= lookback:
        return None
    base = float(closes.iloc[-lookback - 1])
    if base == 0:
        return None
    return 1e4 * (float(closes.iloc[-1]) / base - 1.0)


def _realized_vol_bps(closes: pd.Series, window: int) -> Optional[float]:
    n = min(window + 1, len(closes))
    if n < 3:
        return None
    rets = closes.iloc[-n:].pct_change().dropna()
    if rets.empty:
        return None
    return 1e4 * float(rets.std(ddof=0))


def _z_vs_reference(closes: pd.Series, reference: Optional[pd.Series], window: int) -> Optional[float]:
    n = min(window, len(closes))
    if n < 3:
        return None
    window_close = closes.iloc[-n:]
    if reference is not None and reference.iloc[-n:].notna().all():
        ref = float(reference.iloc[-n:].mean())
    else:
        ref = float(window_close.mean())
    std = float(window_close.std(ddof=0))
    if std == 0:
        return 0.0
    return (float(closes.iloc[-1]) - ref) / std


def _rsi(closes: pd.Series, window: int) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = float(gain.rolling(window).mean().iloc[-1])
    avg_loss = float(loss.rolling(window).mean().iloc[-1])
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_pct(bars: pd.DataFrame, window: int) -> Optional[float]:
    if len(bars) < window + 1:
        return None
    high, low, close = bars["high"].astype(float), bars["low"].astype(float), bars["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(window).mean().iloc[-1])
    last_px = float(close.iloc[-1])
    if last_px == 0 or math.isnan(atr):
        return None
    return 100.0 * atr / last_px


def _vol_ratio(volumes: pd.Series, window: int) -> Optional[float]:
    if len(volumes) < 2:
        return None
    n = min(window, len(volumes) - 1)
    if n < 3:
        return None
    avg = float(volumes.iloc[-n - 1 : -1].mean())
    if avg == 0:
        return None
    return float(volumes.iloc[-1]) / avg


def _trend_slope_bps(closes: pd.Series, window: int) -> Optional[float]:
    n = min(window, len(closes))
    if n < 3:
        return None
    y = closes.iloc[-n:].to_numpy(dtype=float)
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0 or y_mean == 0:
        return None
    slope = float(((x - x_mean) * (y - y_mean)).sum()) / denom
    return 1e4 * slope / y_mean


def _session_bucket(ts: pd.Timestamp) -> str:
    """Coarse, deterministic UTC-hour bucket. JevTrader trades both 24/7 crypto and
    9:30-16:00 ET equities, so this is intentionally asset-agnostic; strategies that care
    about exchange-specific sessions can compute their own feature and pass it through
    `extra` in a future revision."""
    t = pd.Timestamp(ts)
    h = t.tz_convert("UTC").hour if t.tzinfo is not None else t.hour
    if 13 <= h < 14:
        return "us_open"
    if 14 <= h < 19:
        return "us_midday"
    if 19 <= h < 21:
        return "us_close"
    if 21 <= h < 24 or h < 0:
        return "us_after"
    if 0 <= h < 8:
        return "asia"
    return "europe"


def _position_pnl_bps(position: Optional[Position], last_px: float) -> float:
    if position is None or position.qty == 0 or position.avg_price == 0:
        return 0.0
    upnl = position.unrealized_pnl(last_px)
    notional = abs(position.qty) * position.avg_price
    if notional == 0:
        return 0.0
    return round(1e4 * upnl / notional, 1)


def _round_tree(value: Any, ndigits: int = 4) -> Any:
    """Recursively round every float in a (JSON-shaped) nested structure, leaving everything
    else -- ints, bools, strings, `None` -- untouched. Used for `extra_features`, which comes
    from another subsystem (e.g. a forecast model) and may not already be rounded."""
    if isinstance(value, Mapping):
        return {k: _round_tree(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_tree(v, ndigits) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _round(float(value), ndigits)
    return value


def build_state(
    symbol: str,
    bars: pd.DataFrame,
    *,
    quote: Optional[Quote] = None,
    book: Optional[OrderBook] = None,
    position: Optional[Position] = None,
    cost_bps: float = 0.0,
    now: Optional[pd.Timestamp] = None,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    vol_window: int = 20,
    rsi_window: int = 14,
    atr_window: int = 14,
    trend_window: int = 20,
    zscore_window: int = 20,
    extra_features: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a compact, deterministic, JSON-serializable feature dict for Jev.

    `bars` must have columns `open, high, low, close, volume[, vwap]`, at least one row,
    indexed by bar-close timestamp (ascending). All rolling features silently degrade to
    `None` when there isn't enough history yet (no NaNs are ever emitted). `now` defaults
    to the last bar's timestamp, keeping the function pure given the same `bars`.

    `cost_bps` should be the round-trip trading cost for this instrument (fees + typical
    spread/slippage), e.g. from `FeeModel.round_trip_bps` plus half the quoted spread; it
    tells Jev what size of move is actually worth trading, which is exactly the "up"/"down"
    threshold used by `questions.direction_questions`.

    `extra_features`, when given, is merged in verbatim (recursively rounded to keep it
    compact) under a `"forecast"` sub-key -- e.g. quantile forecasts from a separate
    `jevtrader.forecast` model (`tfm_ret_q10_bps`, `tfm_p_up_gt_cost`, ...). This module has no
    dependency on that one: it just merges whatever dict it's handed. Passing `None` (the
    default) leaves the state byte-for-byte identical to a call without this argument.
    """
    if bars is None or len(bars) == 0:
        raise ValueError("build_state requires at least one bar")

    closes = bars["close"].astype(float)
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp(bars.index[-1])
    last_px = float(closes.iloc[-1])
    vwap = bars["vwap"].astype(float) if "vwap" in bars.columns else None
    volumes = bars["volume"].astype(float) if "volume" in bars.columns else None

    book_quote = book.to_quote() if book is not None else None

    ret_bps = {str(int(lb)): _round(_ret_bps(closes, int(lb)), 1) for lb in lookbacks}

    state: dict[str, Any] = {
        "symbol": symbol,
        "n_bars": int(len(bars)),
        "px": round(last_px, 6),
        "ret_bps": ret_bps,
        "vol_bps": _round(_realized_vol_bps(closes, vol_window), 1),
        "z_vwap": _round(_z_vs_reference(closes, vwap, zscore_window), 3),
        "rsi": _round(_rsi(closes, rsi_window), 1),
        "atr_pct": _round(_atr_pct(bars, atr_window), 3),
        "vol_ratio": _round(_vol_ratio(volumes, vol_window), 3) if volumes is not None else None,
        "trend_slope_bps": _round(_trend_slope_bps(closes, trend_window), 2),
        "spread_bps": _round(1e4 * quote.spread / quote.mid, 2) if (quote is not None and quote.mid) else None,
        "book_imbalance": _round(book.imbalance(), 3) if book is not None else None,
        "microprice_offset_bps": (
            _round(1e4 * (book_quote.microprice - book_quote.mid) / book_quote.mid, 2)
            if (book_quote is not None and book_quote.mid)
            else None
        ),
        "session": _session_bucket(ts),
        "minute_of_day": int(ts.hour * 60 + ts.minute),
        "dow": int(ts.dayofweek),
        "pos_qty_sign": 0 if (position is None or position.qty == 0) else (1 if position.qty > 0 else -1),
        "unrealized_pnl_bps": _position_pnl_bps(position, last_px),
        "cost_bps": round(float(cost_bps), 2),
    }
    if extra_features is not None:
        state["forecast"] = _round_tree(dict(extra_features))
    return state


def to_json(state: Mapping[str, Any]) -> str:
    """Deterministic, compact JSON encoding of a state dict (sorted keys, no extra whitespace).

    Used both to check the state's token footprint and, in `advisor.py`, as the basis for
    the cache key hash — the same encoding must be used for both so that "rounded-state
    hash" is stable.
    """
    return json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
