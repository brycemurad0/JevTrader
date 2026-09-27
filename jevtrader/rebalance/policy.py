"""Cost-aware rebalance policy: tolerance bands, a calendar check, and a hard rule - trade only
when the expected tracking-error reduction is worth more than the estimated trading cost.

An optional Jev regime tilt scales down risky-asset weights (never up) by a bounded amount when
Jev judges the market `risk_off`; the tilt is logged and capped, and with `jev=None` this module
is pure quant (no behavior change), matching the "Jev judges, code executes" pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from jevtrader.core.fees import FeeModel
from jevtrader.core.interfaces import JevAdvisorProtocol
from jevtrader.core.types import AssetClass, Instrument, Liquidity, Side


@dataclass
class RebalancePolicy:
    """All the knobs governing *whether* to rebalance. See `RebalancePlanner` for *how*."""

    # 5/25 tolerance bands (Swensen-style): a symbol is "out of band" if its weight has drifted
    # by more than `abs_band` in absolute terms, OR by more than `rel_band` relative to its own
    # target (whichever triggers first). Small target weights are dominated by the relative band.
    abs_band: float = 0.05
    rel_band: float = 0.25

    # Calendar cadence: informational for the caller (e.g. the bot only evaluates on this cadence
    # via `calendar_due`); a band breach is always actionable regardless of the calendar.
    frequency: str = "1w"  # "1d" | "1w" | "1m"

    # Don't bother trading a symbol for less than this many dollars.
    min_trade_notional: float = 50.0

    # Half-spread estimate (bps of notional) used when a live quote's own spread isn't available.
    default_spread_bps: float = 5.0

    # Jev regime tilt: `tilt_k` scales the down-weighting by p(risk_off); `tilt_max` hard-caps how
    # far any single asset's weight can be pulled down (as a fraction of its own weight).
    tilt_k: float = 0.5
    tilt_max: float = 0.20
    risk_off_question: str = "risk_off"


@dataclass
class RebalanceDecision:
    should_trade: bool
    target_weights: dict[str, float]  # post-tilt targets actually used for this decision
    trade_weights: dict[str, float]  # symbol -> (target - current) weight delta, bands-filtered
    reasons: list[str] = field(default_factory=list)
    est_cost_usd: float = 0.0
    est_benefit_usd: float = 0.0
    tilt_log: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------------- tolerance bands


def band_breach(current_weights: Mapping[str, float], target_weights: Mapping[str, float], policy: RebalancePolicy) -> dict[str, bool]:
    """The 5/25 rule per symbol: breach if drift exceeds the absolute band, OR exceeds the
    relative band as a fraction of the target weight."""
    symbols = set(current_weights) | set(target_weights)
    out: dict[str, bool] = {}
    for sym in symbols:
        cur = current_weights.get(sym, 0.0)
        tgt = target_weights.get(sym, 0.0)
        drift = abs(cur - tgt)
        abs_hit = drift > policy.abs_band
        rel_hit = tgt > 0 and (drift / tgt) > policy.rel_band
        out[sym] = bool(abs_hit or rel_hit)
    return out


_FREQ_DAYS = {"1d": 1, "1w": 7, "1m": 30}


def calendar_due(last_rebalance_ts: Optional[pd.Timestamp], now: pd.Timestamp, frequency: str) -> bool:
    """Whether the routine calendar cadence has elapsed. A `None` last_rebalance_ts is always due."""
    if last_rebalance_ts is None:
        return True
    days = _FREQ_DAYS.get(frequency, 7)
    return (now - last_rebalance_ts) >= pd.Timedelta(days=days)


# ----------------------------------------------------------------------------- cost & benefit


def estimated_cost(
    trade_notional: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    fee_model: FeeModel,
    spread_bps: Mapping[str, float] | float,
    prices: Mapping[str, float],
) -> float:
    """Round-trip-agnostic single-leg cost estimate: taker fee + half-spread slippage on each
    trade's notional. `spread_bps` may be a per-symbol map or a single fallback value."""
    total = 0.0
    for sym, notional in trade_notional.items():
        if notional == 0:
            continue
        inst = instruments.get(sym) or Instrument.infer(sym)
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        qty = abs(notional) / price
        side = Side.BUY if notional > 0 else Side.SELL
        fee = fee_model.fee(inst, side, qty, price, Liquidity.TAKER)
        sbps = spread_bps.get(sym, 0.0) if isinstance(spread_bps, Mapping) else spread_bps
        slippage = abs(notional) * (sbps / 2.0) / 1e4  # cross half the spread on entry
        total += fee + slippage
    return total


def expected_te_reduction(current_weights: Mapping[str, float], target_weights: Mapping[str, float], cov: pd.DataFrame) -> float:
    """Tracking error (periodic, fractional) between current and target weights under `cov`.
    Trading fully to target eliminates it, so this IS the expected benefit of trading, in the
    same (periodic return) units as `cov`."""
    symbols = list(cov.columns)
    dw = np.array([current_weights.get(s, 0.0) - target_weights.get(s, 0.0) for s in symbols])
    var = float(dw @ cov.to_numpy() @ dw)
    return float(np.sqrt(max(var, 0.0)))


# ----------------------------------------------------------------------------- Jev regime tilt


def apply_jev_tilt(
    target_weights: Mapping[str, float],
    risky_symbols: Mapping[str, Any],  # symbol -> features passed to jev.regime(symbol, features)
    jev: Optional[JevAdvisorProtocol],
    policy: RebalancePolicy,
) -> tuple[dict[str, float], dict[str, float]]:
    """Scale down (never up) each risky symbol's weight by up to `policy.tilt_max`, proportional
    to Jev's `p(risk_off)` for that symbol. The reduced weight becomes cash (capital preservation:
    when unsure, hold less risk, not more). Returns (tilted_weights, tilt_log) where `tilt_log`
    maps symbol -> the multiplicative factor actually applied (1.0 = no tilt).

    With `jev is None`, returns `target_weights` unchanged and an empty log - this module is pure
    quant by default.
    """
    tilted = dict(target_weights)
    log: dict[str, float] = {}
    if jev is None:
        return tilted, log

    for sym, features in risky_symbols.items():
        if sym not in tilted or tilted[sym] <= 0:
            continue
        view = jev.regime(sym, features)
        if view is None or view.late:
            continue  # no view, or too slow to trust: leave weight untouched
        p_risk_off = view.p(policy.risk_off_question, "yes", default=0.0)
        raw_factor = 1.0 - policy.tilt_k * p_risk_off
        factor = float(np.clip(raw_factor, 1.0 - policy.tilt_max, 1.0))  # never increases weight
        tilted[sym] = tilted[sym] * factor
        log[sym] = factor
    return tilted, log


# ----------------------------------------------------------------------------- top-level decision


def decide(
    policy: RebalancePolicy,
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    equity: float,
    cov: pd.DataFrame,
    prices: Mapping[str, float],
    instruments: Mapping[str, Instrument],
    fee_model: FeeModel,
    spread_bps: Mapping[str, float] | float = 0.0,
    jev: Optional[JevAdvisorProtocol] = None,
    jev_features: Optional[Mapping[str, Any]] = None,
    risky_symbols: Optional[Mapping[str, Any]] = None,
) -> RebalanceDecision:
    """The full cost-aware decision. Bands override the calendar (a large drift is always worth
    evaluating); the calendar is exposed separately via `calendar_due` for the caller to decide
    *when* to invoke this at all (e.g. once a day)."""
    reasons: list[str] = []

    tilted, tilt_log = apply_jev_tilt(target_weights, risky_symbols or {}, jev, policy)
    if tilt_log:
        reasons.append(f"jev regime tilt applied: {tilt_log}")

    breaches = band_breach(current_weights, tilted, policy)
    breaching = [s for s, hit in breaches.items() if hit]
    if not breaching:
        reasons.append("within 5/25 tolerance bands - no trade needed")
        return RebalanceDecision(False, dict(tilted), {}, reasons, 0.0, 0.0, tilt_log)

    trade_weights = {s: tilted.get(s, 0.0) - current_weights.get(s, 0.0) for s in breaching}
    trade_notional = {s: dw * equity for s, dw in trade_weights.items()}
    trade_notional = {s: n for s, n in trade_notional.items() if abs(n) >= policy.min_trade_notional}

    if not trade_notional:
        reasons.append("all breaching trades below min_trade_notional")
        return RebalanceDecision(False, dict(tilted), {}, reasons, 0.0, 0.0, tilt_log)

    cost = estimated_cost(trade_notional, instruments, fee_model, spread_bps, prices)
    benefit_fraction = expected_te_reduction(current_weights, tilted, cov)
    benefit = benefit_fraction * equity

    trade_weights = {s: trade_notional[s] / equity for s in trade_notional}

    if cost >= benefit:
        reasons.append(f"estimated cost ${cost:,.2f} >= expected benefit ${benefit:,.2f} - skipping")
        return RebalanceDecision(False, dict(tilted), {}, reasons, cost, benefit, tilt_log)

    reasons.append(f"trading {sorted(trade_notional)}: benefit ${benefit:,.2f} > cost ${cost:,.2f}")
    return RebalanceDecision(True, dict(tilted), trade_weights, reasons, cost, benefit, tilt_log)
