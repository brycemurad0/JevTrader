"""Synthetic market data generators, fully offline and seeded for reproducibility.

Everything here is deterministic given a seed: same seed + same params -> byte-identical
DataFrames. This is what makes strategies and the backtester testable without any exchange
connection.

What is modelled, and the assumptions baked in (read before trusting statistics from it):

* **Bars** (`generate_bars`): a 3-state Markov regime switch (trend / mean-revert / high-vol
  chop) drives log-returns each bar. Trend regimes get a persistent per-regime drift, mean-revert
  regimes pull the log-price back toward the level it had when the regime started, chop regimes
  add Student-t fat-tailed noise with no drift. Rare Poisson "jump" shocks are layered on top of
  every regime to fatten the unconditional tail further. Intrabar OHLC is synthesized as a
  Brownian bridge in log-space pinned at the bar's open/close, so `high >= max(open, close)` and
  `low <= min(open, close)` always hold and there is no lookahead baked into the shape. Volume
  follows a deterministic intraday U-shape for equities (high at the open/close, low at midday)
  and a flatter, mildly diurnal profile for 24/7 crypto, both with lognormal multiplicative noise
  and a boost during high-realized-vol bars (volume clusters with volatility, as in real markets).
* **Quote/trade tape** (`generate_quote_trade_stream`): order-flow imbalance (OFI) is an AR(1)
  process in [-1, 1]. Crucially, RETURNS ARE PARTIALLY CAUSED BY LAGGED OFI (`ret[t] = alpha *
  ofi[t-1] + noise[t]`), not the other way around, so there is a real (small, `alpha` controls
  the effect size) short-horizon predictive relationship a strategy could in principle discover
  from *past* imbalance -- with no lookahead, since ofi[t-1] is known before ret[t] happens.
  Typical default `alpha` gives an out-of-sample IC on the order of a few percent: a genuine but
  small edge, consistent with real market microstructure. Bid/ask sizes are driven by the same
  OFI so the imbalance is visible in the book, spread widens with realized local vol, and trade
  aggressor side is biased by OFI (more buy-aggressed prints when OFI > 0).
* **L2 order book** (`generate_order_book_stream`): built directly on top of the quote stream --
  best bid/ask match the quotes exactly -- with `n_levels` synthetic price levels stepped out by
  `tick_size` and sizes decaying geometrically with depth plus noise.
* **Cointegrated pairs** (`generate_cointegrated_pair`): a shared latent random-walk factor plus
  a stationary OU-process spread gives two I(1) series whose linear combination
  `p_b - beta * p_a` is stationary by construction -- the textbook stat-arb setup.

None of this claims to reproduce any real instrument's actual statistics; it is meant to exercise
strategy and backtester logic (regime handling, fat tails, imbalance edges, cointegration) under
controlled, known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from jevtrader.core.types import AssetClass, Bar, BookLevel, OrderBook, Quote, Trade

_NY = ZoneInfo("America/New_York")

# Regime indices
TREND, MEAN_REVERT, CHOP = 0, 1, 2
_REGIME_NAMES = {TREND: "trend", MEAN_REVERT: "mean_revert", CHOP: "chop"}

_EQUITY_SECONDS_PER_YEAR = 252 * 6.5 * 3600
_CRYPTO_SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass(frozen=True)
class RegimeParams:
    """Per-regime return dynamics. All rates are in "per unit bar_vol" terms."""

    vol_mult: float  # multiple of the base per-bar vol
    theta: float = 0.0  # mean-reversion strength (mean_revert regime only)
    fat_tail_df: float = 30.0  # Student-t degrees of freedom (lower = fatter tails)
    drift_range: tuple[float, float] = (0.0, 0.0)  # abs drift range per bar (trend regime)
    jump_prob: float = 0.001  # probability per bar of an extra jump shock
    jump_scale: float = 4.0  # jump size, in multiples of base per-bar vol


DEFAULT_REGIME_PARAMS: dict[int, RegimeParams] = {
    TREND: RegimeParams(vol_mult=0.9, fat_tail_df=8.0, drift_range=(0.15, 0.6), jump_prob=0.0008, jump_scale=4.0),
    MEAN_REVERT: RegimeParams(vol_mult=0.7, theta=0.06, fat_tail_df=6.0, jump_prob=0.0008, jump_scale=4.0),
    CHOP: RegimeParams(vol_mult=2.2, fat_tail_df=3.0, jump_prob=0.003, jump_scale=6.0),
}


def _t_innovation(rng: np.random.Generator, df: float, size: int) -> np.ndarray:
    """Unit-variance Student-t draws (falls back to normal as df -> inf)."""
    if df is None or df > 200:
        return rng.standard_normal(size)
    z = rng.standard_t(df, size=size)
    return z / np.sqrt(df / (df - 2.0))


def regime_chain(n: int, rng: np.random.Generator, persistence: float = 0.985, n_states: int = 3) -> np.ndarray:
    """Markov regime path. `persistence` is P(stay); average regime length ~= 1/(1-persistence) bars."""
    off = (1.0 - persistence) / (n_states - 1)
    trans = np.full((n_states, n_states), off)
    np.fill_diagonal(trans, persistence)
    states = np.empty(n, dtype=np.int8)
    states[0] = rng.integers(0, n_states)
    for i in range(1, n):
        states[i] = rng.choice(n_states, p=trans[states[i - 1]])
    return states


def simulate_regime_log_returns(
    n: int,
    seed: int,
    bar_vol: float,
    regime_params: dict[int, RegimeParams] = DEFAULT_REGIME_PARAMS,
    persistence: float = 0.985,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate `n` regime-switching log-returns. Returns (log_returns, regime_states)."""
    rng = np.random.default_rng(seed)
    states = regime_chain(n, rng, persistence=persistence)
    returns = np.empty(n)
    cum = 0.0
    anchor = 0.0
    trend_drift = 0.0
    prev_regime = -1
    for i in range(n):
        regime = int(states[i])
        p = regime_params[regime]
        if regime != prev_regime:
            if regime == TREND:
                trend_drift = rng.choice([-1.0, 1.0]) * rng.uniform(*p.drift_range) * bar_vol
            elif regime == MEAN_REVERT:
                anchor = cum
            prev_regime = regime
        if regime == TREND:
            mu = trend_drift
        elif regime == MEAN_REVERT:
            mu = -p.theta * (cum - anchor)
        else:
            mu = 0.0
        z = _t_innovation(rng, p.fat_tail_df, 1)[0]
        jump = 0.0
        if rng.random() < p.jump_prob:
            jump = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * p.jump_scale * bar_vol
        r = mu + p.vol_mult * bar_vol * z + jump
        returns[i] = r
        cum += r
    return returns, states


def _brownian_bridge_ohlc(open_price: float, log_ret: float, rng: np.random.Generator, sub_steps: int = 8) -> tuple[float, float, float, np.ndarray]:
    """Synthesize (high, low, close, intrabar_price_path) for one bar via a log-space Brownian bridge
    pinned at the open and close, so wicks look organic but never violate high>=max(o,c), low<=min(o,c)."""
    log_open = np.log(open_price)
    log_close = log_open + log_ret
    extra_vol = (abs(log_ret) + 1e-5) * 0.5
    z = rng.standard_normal(sub_steps)
    z -= z.mean()
    bridge = np.cumsum(z) * extra_vol
    bridge -= bridge[-1] * (np.arange(1, sub_steps + 1) / sub_steps)  # pin end deviation to 0
    frac = np.arange(1, sub_steps + 1) / sub_steps
    log_path = log_open + frac * log_ret + bridge
    log_path[-1] = log_close
    path = np.exp(np.concatenate([[log_open], log_path]))
    return float(path.max()), float(path.min()), float(np.exp(log_close)), path


def _equity_session_mask(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert(_NY)
    minutes = local.hour * 60 + local.minute
    is_weekday = local.weekday < 5
    in_session = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return np.asarray(is_weekday & in_session)


def _intraday_volume_shape(index: pd.DatetimeIndex, asset_class: AssetClass) -> np.ndarray:
    if asset_class is AssetClass.EQUITY:
        local = index.tz_convert(_NY)
        minutes = local.hour * 60 + local.minute - (9 * 60 + 30)
        x = np.clip(np.asarray(minutes) / (6.5 * 60), 0.0, 1.0)
        # U-shape: high near the open and the close, low at midday.
        return 1.0 + 2.2 * (2 * x - 1) ** 2
    # crypto: mild diurnal bump for the US/EU trading overlap (14:00-20:00 UTC), else flat 24/7.
    hour = np.asarray(index.hour)
    return 1.0 + 0.35 * np.exp(-0.5 * ((hour - 17) / 3.5) ** 2)


def generate_bars(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "1min",
    asset_class: AssetClass = AssetClass.EQUITY,
    seed: int = 0,
    start_price: float = 100.0,
    annual_vol: float = 0.35,
    base_volume: float = 2_000_000.0,
    regime_params: Optional[dict[int, RegimeParams]] = None,
) -> pd.DataFrame:
    """Synthetic OHLCV bars with regime switching, U-shaped/24-7 volume, and fat tails.

    Equities: only regular-session bars (09:30-16:00 America/New_York, Mon-Fri) are emitted; no
    holiday calendar is applied (a documented simplification). Crypto: bars for every timestamp
    in `[start, end)`, 24/7.

    Returns a UTC-tz-indexed DataFrame with columns open, high, low, close, volume, vwap,
    trade_count.
    """
    start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
    full_index = pd.date_range(start, end, freq=freq, inclusive="left", tz="UTC")
    if asset_class is AssetClass.EQUITY:
        mask = _equity_session_mask(full_index)
        index = full_index[mask]
    else:
        index = full_index
    n = len(index)
    if n == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"])

    bar_seconds = pd.Timedelta(freq).total_seconds()
    seconds_per_year = _EQUITY_SECONDS_PER_YEAR if asset_class is AssetClass.EQUITY else _CRYPTO_SECONDS_PER_YEAR
    bar_vol = annual_vol / np.sqrt(seconds_per_year / bar_seconds)

    log_returns, regimes = simulate_regime_log_returns(n, seed, bar_vol, regime_params or DEFAULT_REGIME_PARAMS)

    rng = np.random.default_rng(seed + 7919)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    closes = np.empty(n)
    vwaps = np.empty(n)
    price = start_price
    for i in range(n):
        opens[i] = price
        h, l, c, path = _brownian_bridge_ohlc(price, log_returns[i], rng)
        highs[i], lows[i], closes[i] = h, l, c
        vwaps[i] = float(path.mean())
        price = c

    vol_shape = _intraday_volume_shape(index, asset_class)
    # volume clusters with volatility: bars with a bigger |return| relative to base vol get more volume.
    realized_vol_boost = 1.0 + 3.0 * np.clip(np.abs(log_returns) / (bar_vol + 1e-12), 0, 5) / 5.0
    lognoise = np.exp(rng.normal(0.0, 0.35, size=n) - 0.35 ** 2 / 2)
    volume = base_volume * (bar_seconds / 60.0) * vol_shape * realized_vol_boost * lognoise
    volume = np.maximum(volume, 1.0)
    trade_count = np.maximum(1, rng.poisson(np.maximum(volume / (base_volume * 0.0005 + 1e-9), 1.0)))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "vwap": vwaps,
            "trade_count": trade_count,
            "regime": [_REGIME_NAMES[int(r)] for r in regimes],
        },
        index=index,
    )
    df.index.name = "ts"
    return df


def bars_to_events(df: pd.DataFrame, symbol: str) -> Iterator[Bar]:
    """Convert a `generate_bars`-shaped DataFrame into a time-ordered iterator of `Bar` events."""
    for ts, row in df.iterrows():
        yield Bar(
            symbol=symbol,
            ts=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            vwap=float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None,
            trade_count=int(row["trade_count"]) if "trade_count" in row and pd.notna(row["trade_count"]) else None,
        )


@dataclass(frozen=True)
class QuoteTradeParams:
    ofi_persistence: float = 0.85  # AR(1) rho for order-flow imbalance
    ofi_noise: float = 0.25
    predictive_alpha: float = 0.12  # effect of lagged OFI on next return (the "real, weak edge"; ~5% IC by default)
    base_spread_bps: float = 2.0
    spread_vol_sensitivity: float = 3.0
    base_size: float = 500.0
    trade_prob: float = 0.6  # probability a tick also produces a trade print


def generate_quote_trade_stream(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "1s",
    asset_class: AssetClass = AssetClass.EQUITY,
    seed: int = 0,
    mid0: float = 100.0,
    annual_vol: float = 0.35,
    tick_size: float = 0.01,
    params: QuoteTradeParams = QuoteTradeParams(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """L1 quote + trade tape with an order-flow-imbalance signal that has weak, genuine short-
    horizon predictive power on the next return (see module docstring for the causal mechanism).

    Returns (quotes_df, trades_df), both UTC-indexed. `quotes_df` has bid, ask, bid_size,
    ask_size, ofi. `trades_df` has price, size, aggressor ("buy"/"sell") and is a (possibly
    sparse) subset of the same timestamps.
    """
    start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    full_index = pd.date_range(start, end, freq=freq, inclusive="left", tz="UTC")
    if asset_class is AssetClass.EQUITY:
        mask = _equity_session_mask(full_index)
        index = full_index[mask]
    else:
        index = full_index
    n = len(index)
    if n == 0:
        empty_q = pd.DataFrame(columns=["bid", "ask", "bid_size", "ask_size", "ofi"])
        empty_t = pd.DataFrame(columns=["price", "size", "aggressor"])
        return empty_q, empty_t

    tick_seconds = pd.Timedelta(freq).total_seconds()
    seconds_per_year = _EQUITY_SECONDS_PER_YEAR if asset_class is AssetClass.EQUITY else _CRYPTO_SECONDS_PER_YEAR
    tick_vol = annual_vol / np.sqrt(seconds_per_year / tick_seconds)

    rng = np.random.default_rng(seed)
    ofi = np.empty(n)
    ofi[0] = 0.0
    ofi_eps = rng.normal(0.0, params.ofi_noise, size=n)
    for i in range(1, n):
        ofi[i] = np.clip(params.ofi_persistence * ofi[i - 1] + ofi_eps[i], -1.0, 1.0)

    resid = rng.standard_normal(n)
    log_returns = np.empty(n)
    log_returns[0] = tick_vol * resid[0]
    for i in range(1, n):
        log_returns[i] = params.predictive_alpha * tick_vol * ofi[i - 1] + tick_vol * resid[i]

    mid = mid0 * np.exp(np.cumsum(log_returns))
    local_vol = pd.Series(log_returns).rolling(20, min_periods=1).std().fillna(tick_vol).to_numpy()
    spread = np.maximum(
        tick_size, mid * (params.base_spread_bps / 1e4) * (1.0 + params.spread_vol_sensitivity * local_vol / (tick_vol + 1e-12))
    )
    spread = np.round(spread / tick_size) * tick_size
    spread = np.maximum(spread, tick_size)

    bid = mid - spread / 2.0
    ask = mid + spread / 2.0
    size_noise_b = np.exp(rng.normal(0, 0.4, n) - 0.08)
    size_noise_a = np.exp(rng.normal(0, 0.4, n) - 0.08)
    bid_size = np.maximum(1.0, params.base_size * (1.0 + 0.8 * ofi) * size_noise_b)
    ask_size = np.maximum(1.0, params.base_size * (1.0 - 0.8 * ofi) * size_noise_a)

    quotes_df = pd.DataFrame(
        {"bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size, "ofi": ofi, "mid": mid},
        index=index,
    )
    quotes_df.index.name = "ts"

    has_trade = rng.random(n) < params.trade_prob
    buy_prob = np.clip(0.5 + 0.3 * ofi, 0.05, 0.95)
    is_buy = rng.random(n) < buy_prob
    trade_size = np.maximum(1.0, rng.lognormal(mean=np.log(params.base_size * 0.15), sigma=0.9, size=n))
    trade_price = np.where(is_buy, ask, bid)

    trades_df = pd.DataFrame(
        {
            "price": trade_price[has_trade],
            "size": trade_size[has_trade],
            "aggressor": np.where(is_buy[has_trade], "buy", "sell"),
        },
        index=index[has_trade],
    )
    trades_df.index.name = "ts"
    return quotes_df, trades_df


def quotes_to_events(df: pd.DataFrame, symbol: str) -> Iterator[Quote]:
    for ts, row in df.iterrows():
        yield Quote(symbol=symbol, ts=ts, bid=float(row["bid"]), ask=float(row["ask"]), bid_size=float(row["bid_size"]), ask_size=float(row["ask_size"]))


def trades_to_events(df: pd.DataFrame, symbol: str) -> Iterator[Trade]:
    from jevtrader.core.types import Side

    for ts, row in df.iterrows():
        side = Side.BUY if row["aggressor"] == "buy" else Side.SELL
        yield Trade(symbol=symbol, ts=ts, price=float(row["price"]), size=float(row["size"]), aggressor=side)


def generate_order_book_stream(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "1s",
    n_levels: int = 10,
    asset_class: AssetClass = AssetClass.EQUITY,
    seed: int = 0,
    mid0: float = 100.0,
    annual_vol: float = 0.35,
    tick_size: float = 0.01,
    depth_decay: float = 0.75,
) -> list[OrderBook]:
    """L2 snapshots built on top of `generate_quote_trade_stream`: best bid/ask match the quote
    stream exactly; `n_levels` are stepped out by `tick_size` with geometrically decaying size."""
    quotes_df, _ = generate_quote_trade_stream(
        symbol, start, end, freq=freq, asset_class=asset_class, seed=seed, mid0=mid0, annual_vol=annual_vol, tick_size=tick_size
    )
    rng = np.random.default_rng(seed + 424242)
    books: list[OrderBook] = []
    for ts, row in quotes_df.iterrows():
        bid0, ask0, bsz0, asz0 = row["bid"], row["ask"], row["bid_size"], row["ask_size"]
        bid_noise = np.exp(rng.normal(0, 0.25, n_levels))
        ask_noise = np.exp(rng.normal(0, 0.25, n_levels))
        bids = tuple(
            BookLevel(price=round(bid0 - i * tick_size, 8), size=max(1.0, bsz0 * (depth_decay ** i) * bid_noise[i]))
            for i in range(n_levels)
        )
        asks = tuple(
            BookLevel(price=round(ask0 + i * tick_size, 8), size=max(1.0, asz0 * (depth_decay ** i) * ask_noise[i]))
            for i in range(n_levels)
        )
        books.append(OrderBook(symbol=symbol, ts=ts, bids=bids, asks=asks))
    return books


def books_to_events(books: Sequence[OrderBook]) -> Iterator[OrderBook]:
    yield from books


def generate_cointegrated_pair(
    symbol_a: str,
    symbol_b: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "1min",
    asset_class: AssetClass = AssetClass.EQUITY,
    seed: int = 0,
    price_a0: float = 100.0,
    beta: float = 1.5,
    spread_mean: float = 0.0,
    half_life_bars: float = 30.0,
    spread_vol: float = 0.15,
    factor_annual_vol: float = 0.30,
) -> dict[str, pd.DataFrame]:
    """Two cointegrated price series for stat-arb tests: `p_b = alpha + beta * p_a + spread`,
    where `spread` is a stationary OU process (so `p_b - beta * p_a` is stationary by
    construction) and `p_a` is a regime-switching random walk (the shared latent factor).

    `alpha` is chosen so both series start at the same level as `price_a0`. Returns
    `{symbol_a: ohlcv_df, symbol_b: ohlcv_df}` (both with a `spread` column holding the true,
    noiseless cointegrating residual for validation in tests).
    """
    df_a = generate_bars(symbol_a, start, end, freq=freq, asset_class=asset_class, seed=seed, start_price=price_a0, annual_vol=factor_annual_vol)
    n = len(df_a)
    if n == 0:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "spread"])
        return {symbol_a: df_a, symbol_b: empty}

    rng = np.random.default_rng(seed + 99991)
    theta = 1.0 - 0.5 ** (1.0 / max(half_life_bars, 1e-6))  # AR(1) coefficient for the given half-life
    spread = np.empty(n)
    spread[0] = spread_mean
    eps = rng.normal(0.0, spread_vol, size=n)
    for i in range(1, n):
        spread[i] = spread[i - 1] + theta * (spread_mean - spread[i - 1]) + eps[i]

    alpha = price_a0 - beta * price_a0
    close_b = alpha + beta * df_a["close"].to_numpy() + spread
    open_b = alpha + beta * df_a["open"].to_numpy() + np.concatenate([[spread[0]], spread[:-1]])
    high_b = np.maximum(open_b, close_b) + np.abs(rng.normal(0, spread_vol * 0.1, n))
    low_b = np.minimum(open_b, close_b) - np.abs(rng.normal(0, spread_vol * 0.1, n))
    volume_b = df_a["volume"].to_numpy() * np.exp(rng.normal(0, 0.1, n))

    df_b = pd.DataFrame(
        {
            "open": open_b,
            "high": high_b,
            "low": low_b,
            "close": close_b,
            "volume": volume_b,
            "vwap": (open_b + close_b) / 2.0,
            "spread": spread,
        },
        index=df_a.index,
    )
    df_b.index.name = "ts"
    df_a = df_a.copy()
    df_a["spread"] = spread
    return {symbol_a: df_a, symbol_b: df_b}
