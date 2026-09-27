"""What we ask a `Forecaster` to forecast, and how to get there and back.

A `Forecaster` (TimesFM or a baseline) only knows how to extend a plain 1-D numeric series.
Everything about *what that series means* -- log returns vs. realized vol vs. log volume --
lives here: a `transform` that turns a bars DataFrame into the series fed to the model, and an
`inverse` that turns the model's per-step `QuantileForecast` back into the unit the rest of the
system wants (bps for returns, bps for vol, a ratio for volume).

Honest-prior note (see `docs/FORECASTING.md`): returns are transformed as *1-step log returns*
and the model forecasts each future step's return, not the cumulative move directly. Turning
that H-step-ahead path of marginal return quantiles into one "cumulative return over the next H
bars" quantile requires an assumption, because the model gives us marginals, not the joint
distribution across steps. `cumulative_log_return_bps` uses the standard practitioner
approximation -- treat steps as independent, so log-variances add -- via each step's own
quantile-implied normal z-score. That is an approximation, not a fact about the market; it is
most defensible for the volatility/volume targets (which are forecast as their own levels, no
compounding needed) and weakest for long-horizon direction, which is exactly why the honest
prior says direction should be scored on the scoreboard rather than assumed to work.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import norm

from jevtrader.forecast.base import QUANTILE_LEVELS, QuantileForecast

BPS = 1e4


# ----------------------------------------------------------------------------- transforms (bars -> series)


def log_return_series(closes: np.ndarray) -> np.ndarray:
    """1-step log returns, length `len(closes) - 1`. This is the series fed to the forecaster
    for the return target; positive/negative log returns are (to first order) symmetric bps
    moves, which is what `direction_questions`/`kelly_from_jev` want."""
    closes = np.asarray(closes, dtype=float)
    if np.any(closes <= 0):
        raise ValueError("log_return_series requires strictly positive prices")
    return np.diff(np.log(closes))


def realized_vol_series(closes: np.ndarray, window: int) -> np.ndarray:
    """Rolling realized volatility (population std of 1-step log returns) over `window` bars,
    length `len(closes) - 1` with the first `window - 1` entries `nan` (not enough history
    yet). This -- not price -- is the series fed to the forecaster for the vol target: we
    forecast the *level* of realized vol directly, the same quantity EWMA/GARCH forecast."""
    rets = log_return_series(closes)
    n = rets.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if window < 2:
        raise ValueError("window must be >= 2")
    csum = np.cumsum(np.concatenate(([0.0], rets)))
    csum2 = np.cumsum(np.concatenate(([0.0], rets * rets)))
    for i in range(window - 1, n):
        s = csum[i + 1] - csum[i + 1 - window]
        s2 = csum2[i + 1] - csum2[i + 1 - window]
        mean = s / window
        var = max(s2 / window - mean * mean, 0.0)
        out[i] = math.sqrt(var)
    return out


def log_volume_series(volumes: np.ndarray) -> np.ndarray:
    """log(volume), floored to avoid `-inf` on zero-volume bars."""
    vol = np.asarray(volumes, dtype=float)
    return np.log(np.clip(vol, 1e-9, None))


def spread_bps_series(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Quoted spread in bps of mid, `(ask - bid) / mid * 1e4`."""
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    mid = 0.5 * (bid + ask)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(mid > 0, (ask - bid) / mid * BPS, np.nan)
    return out


# ----------------------------------------------------------------------------- inverse transforms (forecast -> unit)


def cumulative_log_return_bps(qf: QuantileForecast) -> QuantileForecast:
    """Collapse an H-step-ahead *per-step* log-return `QuantileForecast` into a single-step
    (`horizon == 1`) forecast of the cumulative return over the whole window, in bps.

    Point: `1e4 * (exp(sum(point)) - 1)`. Quantiles: for each level and step, infer the
    z-score implied by that step's quantile deviation from its own point forecast (assuming a
    locally normal conditional distribution -- the same assumption the quantile head itself
    typically makes), sum the implied variances across steps (independence approximation, see
    module docstring), then map the total z back through the normal CDF at the cumulative
    point. This keeps the grid monotonic by construction (z is monotonic in `level`).
    """
    if qf.horizon == 0:
        raise ValueError("cumulative_log_return_bps requires at least one horizon step")
    point_sum = float(np.sum(qf.point))
    total_var = 0.0
    for h in range(qf.horizon):
        step_point = float(qf.point[h])
        # use the two most informative levels (q10/q90) to estimate step std, averaged for
        # robustness against a single noisy tail.
        stds = []
        for level in (0.1, 0.9):
            z = norm.ppf(level)
            if z == 0:
                continue
            v = float(qf.quantiles[level][h])
            stds.append(abs((v - step_point) / z))
        step_std = float(np.mean(stds)) if stds else 0.0
        total_var += step_std * step_std
    total_std = math.sqrt(total_var)
    quantiles = {}
    for level in QUANTILE_LEVELS:
        z = norm.ppf(level)
        log_val = point_sum + z * total_std
        quantiles[level] = np.array([BPS * (math.exp(log_val) - 1.0)])
    point_bps = np.array([BPS * (math.exp(point_sum) - 1.0)])
    return QuantileForecast(point=point_bps, quantiles=quantiles, latency_ms=qf.latency_ms)


def to_bps(qf: QuantileForecast) -> QuantileForecast:
    """Scale a fractional-level forecast (e.g. realized vol as std of log returns) to bps."""
    return qf.scaled(BPS)


def exp_levels(qf: QuantileForecast) -> QuantileForecast:
    """Invert a log-level forecast (e.g. log volume) back to levels via elementwise `exp`."""
    return QuantileForecast(
        point=np.exp(qf.point),
        quantiles={level: np.exp(arr) for level, arr in qf.quantiles.items()},
        latency_ms=qf.latency_ms,
    )


def last_valid(series: np.ndarray) -> Optional[float]:
    """Last non-nan value of a series, or `None` if there isn't one (e.g. too short for the
    rolling window in `realized_vol_series`)."""
    arr = np.asarray(series, dtype=float)
    finite = arr[~np.isnan(arr)]
    return float(finite[-1]) if finite.size else None
