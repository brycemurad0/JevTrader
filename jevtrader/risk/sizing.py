"""Position sizing helpers. Pure functions - no state, no I/O, easy to unit test.

These are the *sizing* layer: given a risk budget and an edge estimate, how many units to trade.
They do not know about account-wide exposure caps; `RiskManager` applies those afterwards.
"""

from __future__ import annotations

import math
from typing import Optional

from jevtrader.core.types import Instrument


def fixed_fractional_qty(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """Classic fixed-fractional sizing: risk `risk_pct` of equity if the stop is hit.

    qty = (equity * risk_pct) / |entry_price - stop_price|
    Returns 0 if inputs are degenerate (no stop distance, non-positive equity/price).
    """
    if equity <= 0 or risk_pct <= 0 or entry_price <= 0:
        return 0.0
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0
    risk_amount = equity * risk_pct
    return risk_amount / stop_distance


def vol_target_qty(
    equity: float,
    target_annual_vol: float,
    price: float,
    asset_annual_vol: float,
    max_leverage: float = 1.0,
) -> float:
    """Volatility targeting: size so the position's own annualized return-vol contribution
    matches `target_annual_vol`, i.e. notional = equity * target_annual_vol / asset_annual_vol.

    `max_leverage` caps notional at `max_leverage * equity` (default: no leverage).
    """
    if equity <= 0 or price <= 0 or asset_annual_vol <= 0 or target_annual_vol <= 0:
        return 0.0
    notional = equity * (target_annual_vol / asset_annual_vol)
    notional = min(notional, max_leverage * equity)
    return notional / price


def kelly_fraction(
    p_win: float,
    payoff_ratio: float,
    fraction: float = 0.25,
    cap_pct_equity: float = 0.05,
) -> float:
    """Generic fractional Kelly for a binary win/lose-the-stake bet.

    `payoff_ratio` (b) is the amount won per unit staked on a win (net odds); a full loss of the
    stake is assumed on a loss. f* = (b*p - q) / b. Returns 0 when the edge is non-positive.
    Applies `fraction` (default quarter-Kelly) and a hard cap on the returned equity fraction.
    """
    if payoff_ratio <= 0 or not (0.0 <= p_win <= 1.0):
        return 0.0
    q = 1.0 - p_win
    f_star = (payoff_ratio * p_win - q) / payoff_ratio
    if f_star <= 0:
        return 0.0
    return float(min(fraction * f_star, cap_pct_equity))


def kelly_from_jev(
    p_up: float,
    p_down: float,
    up_move_bps: float,
    down_move_bps: float,
    cost_bps: float,
    fraction: float = 0.25,
    cap_pct_equity: float = 0.05,
) -> float:
    """Fractional Kelly sized directly from Jev's typed probabilities.

    `p_up`/`p_down` are probabilities of an up/down move over the trade horizon (from a Jev
    `direction` question). `up_move_bps`/`down_move_bps` are the expected magnitude of each move
    (positive numbers, in bps). `cost_bps` is the round-trip trading cost (fees + spread estimate).

    Cost is charged on both outcomes (you pay it regardless of direction): the effective win is
    `up_move_bps - cost_bps` and the effective loss magnitude is `down_move_bps + cost_bps`. Using
    the exact two-outcome Kelly solution f* = p/l - q/g (g = effective win, l = effective loss,
    both as fractions), scaled by `fraction` and hard-capped at `cap_pct_equity` of equity.

    Returns 0.0 whenever the edge net of costs is <= 0, or on any degenerate input.
    """
    if not (0.0 <= p_up <= 1.0) or not (0.0 <= p_down <= 1.0):
        return 0.0
    edge_bps = p_up * up_move_bps - p_down * down_move_bps - cost_bps
    if edge_bps <= 0:
        return 0.0
    g = (up_move_bps - cost_bps) / 1e4  # effective fractional gain on an up move
    l = (down_move_bps + cost_bps) / 1e4  # effective fractional loss on a down move
    if g <= 0 or l <= 0:
        return 0.0
    f_star = p_up / l - p_down / g
    if f_star <= 0:
        return 0.0
    return float(min(fraction * f_star, cap_pct_equity))


def lot_round(qty: float, instrument: Instrument, min_notional: Optional[float] = None, price: Optional[float] = None) -> float:
    """Round `qty` down (towards zero) to the instrument's lot size.

    If `min_notional` and `price` are both given and the rounded order's notional falls below
    `min_notional`, returns 0.0 (the order isn't worth sending).
    """
    lot = instrument.lot_size or 1.0
    sign = 1.0 if qty >= 0 else -1.0
    n = math.floor(abs(qty) / lot + 1e-9)
    rounded = sign * n * lot
    if min_notional is not None and price is not None and abs(rounded) * price < min_notional:
        return 0.0
    return rounded
