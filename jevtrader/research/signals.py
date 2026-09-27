"""Vectorized target-position generators used by the fast-screen. Each function takes a bars
DataFrame (bar-close indexed) and returns a target position series in [-1, 1] (fraction of
equity) decided at each bar's CLOSE -- the strategy modules in `jevtrader/strategies/` implement
the same logic event-by-event; tests in `tests/strategies/test_fastscreen_agreement.py` check
they agree.

All rolling statistics use only past/current bars (pandas rolling is trailing), so nothing here
looks ahead. Positions are "what we want to hold from the next bar's open".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _log_ret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Trailing std of log returns per bar (NOT annualized)."""
    return _log_ret(close).rolling(window).std()


def vol_target_scale(close: pd.Series, window: int, target_bar_vol: float, cap: float = 1.0) -> pd.Series:
    """Position scale so that (scale * realized bar vol) ~= target_bar_vol, capped at `cap`."""
    rv = realized_vol(close, window)
    return (target_bar_vol / rv).clip(upper=cap).fillna(0.0)


# ----------------------------------------------------------------------------- mean reversion


def zscore_vwap_reversion(
    bars: pd.DataFrame,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    vol_window: int = 240,
    max_vol_mult: Optional[float] = 1.5,
    long_only: bool = False,
    hold_max: Optional[int] = None,
) -> pd.Series:
    """Bollinger/VWAP z-score mean reversion with a volatility-regime filter.

    Enter when close deviates > `entry_z` sigma from the rolling mean (rolling VWAP if the bars
    have volume; we use volume-weighted close), exit when |z| < `exit_z`. Trades are suppressed
    when trailing short-window vol exceeds `max_vol_mult` x its long-window average (the
    "trending/news" regime where reversion fails). `hold_max` force-exits after that many bars.
    """
    z, ok = zscore_features(bars, window, vol_window, max_vol_mult)
    return _band_position(z, entry_z, exit_z, ok, long_only, hold_max)


def zscore_features(bars: pd.DataFrame, window: int, vol_window: int = 240, max_vol_mult: Optional[float] = 1.5) -> tuple[pd.Series, pd.Series]:
    """(z, ok): z = (close - rolling VWAP) / rolling std(close); ok = vol-regime filter passes.
    Shared by the fast-screen and `jevtrader.strategies.vwap_reversion` (single source of truth)."""
    close = bars["close"].astype(float)
    vol = bars["volume"].astype(float).replace(0.0, np.nan)
    pv = (close * vol).rolling(window).sum()
    v = vol.rolling(window).sum()
    ref = (pv / v).where(v > 0, close.rolling(window).mean())
    sd = close.rolling(window).std()
    z = ((close - ref) / sd).replace([np.inf, -np.inf], np.nan)
    ok = pd.Series(True, index=bars.index)
    if max_vol_mult is not None:
        rv_short = realized_vol(close, window)
        rv_long = realized_vol(close, vol_window)
        ok = (rv_short <= max_vol_mult * rv_long).fillna(False)
    return z, ok


def _band_position(z: pd.Series, entry: float, exit_: float, ok: pd.Series, long_only: bool, hold_max: Optional[int]) -> pd.Series:
    zv = z.to_numpy()
    okv = ok.to_numpy()
    n = len(zv)
    pos = np.zeros(n)
    cur = 0.0
    held = 0
    for i in range(n):
        zi = zv[i]
        if np.isnan(zi):
            pos[i] = cur
            continue
        if cur == 0.0:
            if okv[i]:
                if zi < -entry:
                    cur = 1.0
                    held = 0
                elif zi > entry and not long_only:
                    cur = -1.0
                    held = 0
        else:
            held += 1
            if (cur > 0 and zi > -exit_) or (cur < 0 and zi < exit_) or (hold_max is not None and held >= hold_max):
                cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=z.index)


# ----------------------------------------------------------------------------- trend / momentum


def ema_crossover(bars: pd.DataFrame, fast: int = 20, slow: int = 100, long_only: bool = False, vol_window: int = 0, target_bar_vol: float = 0.0) -> pd.Series:
    close = bars["close"].astype(float)
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    sig = np.sign(f - s).fillna(0.0)
    if long_only:
        sig = sig.clip(lower=0.0)
    if vol_window and target_bar_vol > 0:
        sig = sig * vol_target_scale(close, vol_window, target_bar_vol)
    return sig


def donchian(bars: pd.DataFrame, n: int = 60, exit_n: Optional[int] = None, long_only: bool = False) -> pd.Series:
    """Donchian channel breakout: long on close > prior n-bar high, short on < prior n-bar low,
    exit on the opposite `exit_n` channel (default n//2)."""
    hi = bars["high"].astype(float).shift(1).rolling(n).max()
    lo = bars["low"].astype(float).shift(1).rolling(n).min()
    m = exit_n or max(n // 2, 2)
    xhi = bars["high"].astype(float).shift(1).rolling(m).max()
    xlo = bars["low"].astype(float).shift(1).rolling(m).min()
    c = bars["close"].astype(float).to_numpy()
    hiv, lov, xhiv, xlov = hi.to_numpy(), lo.to_numpy(), xhi.to_numpy(), xlo.to_numpy()
    pos = np.zeros(len(c))
    cur = 0.0
    for i in range(len(c)):
        if np.isnan(hiv[i]) or np.isnan(lov[i]):
            continue
        if cur == 0.0:
            if c[i] > hiv[i]:
                cur = 1.0
            elif c[i] < lov[i] and not long_only:
                cur = -1.0
        elif cur > 0 and c[i] < xlov[i]:
            cur = 0.0
        elif cur < 0 and c[i] > xhiv[i]:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=bars.index)


def tsmom(bars: pd.DataFrame, lookback: int = 60, long_only: bool = False, vol_window: int = 0, target_bar_vol: float = 0.0) -> pd.Series:
    """Time-series momentum: sign of trailing `lookback`-bar return."""
    close = bars["close"].astype(float)
    sig = np.sign(close / close.shift(lookback) - 1.0).fillna(0.0)
    if long_only:
        sig = sig.clip(lower=0.0)
    if vol_window and target_bar_vol > 0:
        sig = sig * vol_target_scale(close, vol_window, target_bar_vol)
    return sig


# ----------------------------------------------------------------------------- breakouts


def session_breakout(
    bars: pd.DataFrame,
    session_start_utc: int = 13,
    session_start_minute: int = 30,
    range_minutes: int = 30,
    hold_minutes: int = 240,
    long_only: bool = False,
    min_range_bps: float = 0.0,
) -> pd.Series:
    """Opening-range breakout: define the range over the first `range_minutes` after the session
    open (UTC hour/minute), go long on a close above the range high (short below the low) within
    the session, flat after `hold_minutes` or at the next session start. One trade per session."""
    idx = bars.index
    minute_of_day = idx.hour * 60 + idx.minute
    start_m = session_start_utc * 60 + session_start_minute
    day_key = (idx - pd.Timedelta(minutes=start_m)).date  # sessions may cross midnight UTC
    df = pd.DataFrame({"high": bars["high"].astype(float), "low": bars["low"].astype(float), "close": bars["close"].astype(float), "m": minute_of_day, "day": day_key}, index=idx)
    rel = (df["m"] - start_m) % 1440
    in_range = rel < range_minutes
    rng_hi = df["high"].where(in_range).groupby(df["day"]).cummax().ffill()
    rng_lo = df["low"].where(in_range).groupby(df["day"]).cummin().ffill()
    # only trade after the range is formed and before hold expiry
    tradable = (rel >= range_minutes) & (rel < range_minutes + hold_minutes)
    wide_enough = (1e4 * (rng_hi - rng_lo) / df["close"]) >= min_range_bps
    c = df["close"].to_numpy()
    hi, lo = rng_hi.to_numpy(), rng_lo.to_numpy()
    tr = (tradable & wide_enough).to_numpy()
    days = df["day"].to_numpy()
    pos = np.zeros(len(c))
    cur = 0.0
    traded_day = None
    for i in range(len(c)):
        if not tr[i]:
            cur = 0.0
            pos[i] = 0.0
            continue
        if cur == 0.0 and traded_day != days[i] and not np.isnan(hi[i]):
            if c[i] > hi[i]:
                cur = 1.0
                traded_day = days[i]
            elif c[i] < lo[i] and not long_only:
                cur = -1.0
                traded_day = days[i]
        pos[i] = cur
    return pd.Series(pos, index=idx)


def vol_squeeze(bars: pd.DataFrame, bb_window: int = 20, kc_mult: float = 1.5, hold: int = 60, long_only: bool = False) -> pd.Series:
    """Bollinger-inside-Keltner squeeze: when BB width < KC width the market is compressed; on
    the first bar the squeeze releases, take the direction of the `bb_window`-bar momentum and
    hold for `hold` bars."""
    close = bars["close"].astype(float)
    ma = close.rolling(bb_window).mean()
    sd = close.rolling(bb_window).std()
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - close.shift()).abs(), (bars["low"] - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(bb_window).mean()
    squeeze = (2 * sd) < (kc_mult * atr)
    mom = close - close.shift(bb_window)
    sq = squeeze.to_numpy()
    mo = mom.to_numpy()
    pos = np.zeros(len(sq))
    cur = 0.0
    left = 0
    for i in range(1, len(sq)):
        if cur != 0.0:
            left -= 1
            if left <= 0:
                cur = 0.0
        if cur == 0.0 and sq[i - 1] and not sq[i] and not np.isnan(mo[i]):
            d = np.sign(mo[i])
            if d > 0 or (d < 0 and not long_only):
                cur = float(d)
                left = hold
        pos[i] = cur
    return pd.Series(pos, index=bars.index)


# ----------------------------------------------------------------------------- seasonality


def hour_of_day(bars: pd.DataFrame, long_hours: tuple[int, ...], short_hours: tuple[int, ...] = ()) -> pd.Series:
    """Hold long during `long_hours` (UTC bar-close hours), short during `short_hours`."""
    h = bars.index.hour
    pos = np.zeros(len(bars))
    pos[np.isin(h, long_hours)] = 1.0
    pos[np.isin(h, short_hours)] = -1.0
    return pd.Series(pos, index=bars.index)


def hourly_return_table(bars: pd.DataFrame) -> pd.DataFrame:
    """Mean/t-stat of log returns by UTC hour -- the seasonality evidence table."""
    r = _log_ret(bars["close"].astype(float)).dropna()
    g = r.groupby(r.index.hour)
    out = pd.DataFrame({"mean_bps": 1e4 * g.mean(), "std_bps": 1e4 * g.std(), "n": g.count()})
    out["t"] = out["mean_bps"] / out["std_bps"] * np.sqrt(out["n"])
    return out


def dow_return_table(bars: pd.DataFrame) -> pd.DataFrame:
    r = _log_ret(bars["close"].astype(float)).dropna()
    g = r.groupby(r.index.dayofweek)
    out = pd.DataFrame({"mean_bps": 1e4 * g.mean(), "std_bps": 1e4 * g.std(), "n": g.count()})
    out["t"] = out["mean_bps"] / out["std_bps"] * np.sqrt(out["n"])
    return out


# ----------------------------------------------------------------------------- lead-lag


def lead_lag(leader: pd.DataFrame, follower: pd.DataFrame, lookback: int = 5, threshold_bps: float = 10.0, hold: int = 5) -> tuple[pd.Series, pd.DataFrame]:
    """Trade `follower` in the direction of the leader's trailing `lookback`-bar return when it
    exceeds `threshold_bps`; hold `hold` bars. Returns (target aligned to follower bars, aligned df)."""
    lead_close = leader["close"].astype(float)
    df = follower.copy()
    lr = 1e4 * (lead_close / lead_close.shift(lookback) - 1.0)
    df["lead_ret_bps"] = lr.reindex(df.index, method="ffill")
    sig = np.where(df["lead_ret_bps"] > threshold_bps, 1.0, np.where(df["lead_ret_bps"] < -threshold_bps, -1.0, 0.0))
    pos = np.zeros(len(df))
    cur, left = 0.0, 0
    for i in range(len(df)):
        if cur != 0.0:
            left -= 1
            if left <= 0:
                cur = 0.0
        if cur == 0.0 and sig[i] != 0.0:
            cur, left = float(sig[i]), hold
        pos[i] = cur
    return pd.Series(pos, index=df.index), df
