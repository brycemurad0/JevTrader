"""Turn a `Forecaster`'s output into the compact, rounded, JSON-safe feature dict Jev (and
`ForecastAdvisor`) consume.

`forecast_features` is the single integration point between this package and the rest of the
decision stack: another agent is adding an `extra_features` parameter to
`jevtrader.jev.state.build_state` that merges a dict like this one into Jev's compact state, so
Jev/Kev/Laya see TimesFM's forecasts as a handful of extra numbers, exactly like every other
`build_state` feature -- no arrays, no objects, always JSON-round-trippable, always `None` (never
NaN) when there isn't enough history or the forecaster failed.

Cached per `(symbol, last bar timestamp, horizon, cost_bps, forecaster identity)` so re-running
the same bar close through multiple strategies/symbols only pays for one forecast call.
"""

from __future__ import annotations

import math
from typing import Any, Dict, MutableMapping, Optional

import numpy as np
import pandas as pd

from jevtrader.forecast.base import Forecaster
from jevtrader.forecast.targets import (
    cumulative_log_return_bps,
    exp_levels,
    last_valid,
    log_return_series,
    log_volume_series,
    realized_vol_series,
    to_bps,
)

DEFAULT_VOL_WINDOW = 20  # bars; matches jevtrader.jev.state.build_state's default vol_window

FEATURE_KEYS: tuple[str, ...] = (
    "tfm_ret_q10_bps",
    "tfm_ret_q50_bps",
    "tfm_ret_q90_bps",
    "tfm_p_up_gt_cost",
    "tfm_p_down_gt_cost",
    "tfm_vol_fcst_bps",
    "tfm_vol_ratio",
    "tfm_volume_ratio",
)

_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _empty() -> Dict[str, Any]:
    return {k: None for k in FEATURE_KEYS}


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


def _cache_key(symbol: str, forecaster: Forecaster, horizon: int, cost_bps: float, ts: pd.Timestamp) -> tuple:
    return (symbol, id(forecaster), int(horizon), round(float(cost_bps), 4), pd.Timestamp(ts))


def clear_cache() -> None:
    """Drop every cached forecast-features result (tests, or a forced recompute)."""
    _CACHE.clear()


def forecast_features(
    symbol: str,
    bars: pd.DataFrame,
    forecaster: Forecaster,
    horizon: int,
    cost_bps: float = 0.0,
    *,
    vol_window: int = DEFAULT_VOL_WINDOW,
    cache: Optional[MutableMapping[tuple, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the compact TimesFM feature dict for `symbol` from `bars` (same shape contract as
    `jevtrader.jev.state.build_state`: columns `open, high, low, close, volume[, ...]`, indexed
    by bar-close timestamp, ascending, no look-ahead -- `bars` must already be truncated to
    "history up to and including now").

    `horizon` is the number of bars ahead to forecast (return/volume) or the vol-forecast step
    used for `tfm_vol_fcst_bps`/`tfm_vol_ratio`. `cost_bps` is the round-trip trading cost, the
    same number `build_state`'s `cost_bps` and `direction_questions` use, so
    `tfm_p_up_gt_cost`/`tfm_p_down_gt_cost` mean exactly "clears that cost", not "moved at all".

    Never raises on a forecaster failure or insufficient history: falls back to an all-`None`
    dict (every key still present, so callers/tests can rely on a stable shape).
    """
    if bars is None or len(bars) == 0:
        raise ValueError("forecast_features requires at least one bar")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    store = _CACHE if cache is None else cache
    ts = pd.Timestamp(bars.index[-1])
    key = _cache_key(symbol, forecaster, horizon, cost_bps, ts)
    if key in store:
        return store[key]

    out = _empty()
    closes = bars["close"].astype(float).to_numpy()
    if closes.shape[0] < 3 or np.any(closes <= 0):
        store[key] = out
        return out

    volumes = bars["volume"].astype(float).to_numpy() if "volume" in bars.columns else None

    series: Dict[str, np.ndarray] = {"return": log_return_series(closes)}
    rv = realized_vol_series(closes, vol_window)
    rv_hist = rv[~np.isnan(rv)]
    if rv_hist.size >= 5:
        series["vol"] = rv_hist
    if volumes is not None:
        log_vol = log_volume_series(volumes)
        if log_vol.size >= 5:
            series["volume"] = log_vol

    try:
        raw = forecaster.forecast(series, horizon)
    except Exception:
        store[key] = out
        return out

    ret_qf = raw.get("return")
    if ret_qf is not None and ret_qf.horizon >= 1:
        try:
            cum = cumulative_log_return_bps(ret_qf)
            out["tfm_ret_q10_bps"] = _round(cum.quantile_at(0.1), 1)
            out["tfm_ret_q50_bps"] = _round(cum.quantile_at(0.5), 1)
            out["tfm_ret_q90_bps"] = _round(cum.quantile_at(0.9), 1)
            out["tfm_p_up_gt_cost"] = _round(cum.prob_above(float(cost_bps)), 4)
            out["tfm_p_down_gt_cost"] = _round(cum.prob_below(-float(cost_bps)), 4)
        except Exception:
            pass

    vol_qf = raw.get("vol")
    if vol_qf is not None:
        try:
            vol_bps_qf = to_bps(vol_qf)
            forecast_vol_bps = float(vol_bps_qf.point[-1])
            out["tfm_vol_fcst_bps"] = _round(forecast_vol_bps, 1)
            realized_vol_frac = last_valid(rv)
            if realized_vol_frac is not None and realized_vol_frac > 0:
                out["tfm_vol_ratio"] = _round(forecast_vol_bps / (realized_vol_frac * 1e4), 3)
        except Exception:
            pass

    volume_qf = raw.get("volume")
    if volume_qf is not None and volumes is not None and volumes[-1] > 0:
        try:
            level_qf = exp_levels(volume_qf)
            out["tfm_volume_ratio"] = _round(float(level_qf.point[-1]) / float(volumes[-1]), 3)
        except Exception:
            pass

    store[key] = out
    return out
