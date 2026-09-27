"""RiskManager: the single pre-trade gate. Implements `jevtrader.core.interfaces.RiskGate`.

Capital preservation first: every check below is one explicit, documented limit from
`RiskLimits`. When several limits would bind, the order is resized down to the tightest one;
when no size satisfies every limit, it is rejected with a human-readable reason. Reduce-only
orders (flattening) are always allowed, even while halted, because that is the only way out of
a kill-switch state.

Thread-safety: every public method takes `self._lock` (an `RLock`), so `RiskManager` can be
shared between a live event loop and a dashboard thread.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import replace
from typing import Mapping, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from jevtrader.core.interfaces import RiskDecision
from jevtrader.core.types import (
    AccountState,
    AssetClass,
    Fill,
    Instrument,
    Order,
    OrderType,
    Position,
    Quote,
    Side,
)
from jevtrader.risk.limits import RiskLimits
from jevtrader.risk.sizing import lot_round


class RiskManager:
    """Pre-trade risk gate + capital-preservation halts + audit trail.

    `instruments`: optional symbol -> Instrument overrides (lot size, shortability, ...).
    Unlisted symbols fall back to `Instrument.infer(symbol)`.
    """

    def __init__(self, limits: RiskLimits, instruments: Optional[Mapping[str, Instrument]] = None):
        self.limits = limits
        self.instruments = dict(instruments or {})
        self._lock = threading.RLock()

        # kill switch / halts
        self._killed = False
        self._kill_reason = ""
        self._daily_halted = False
        self._daily_halt_reason = ""
        self._cooldown_until: Optional[pd.Timestamp] = None

        # drawdown / daily-loss tracking
        self._hwm: Optional[float] = None
        self._day_key = None
        self._day_start_equity: Optional[float] = None

        # consecutive-loss tracking (shadow position ledger fed only by fills)
        self._consecutive_losses = 0
        self._shadow_positions: dict[str, Position] = {}

        # order-rate throttle
        self._order_times: list[pd.Timestamp] = []

        # open-orders-per-symbol bookkeeping: client_order_id -> (symbol, remaining_qty)
        self._open_orders: dict[str, tuple[str, float]] = {}

        # per-strategy capital ledger: strategy_id -> signed notional held (from fills)
        self._strategy_notional: dict[str, float] = {}

        self.audit: list[dict] = []
        self._last_now: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------------------- RiskGate

    def check(self, order: Order, account: AccountState, quote: Optional[Quote], now: pd.Timestamp) -> RiskDecision:
        with self._lock:
            self._refresh(now, account.equity)
            instrument = self._instrument_for(order.symbol)

            if order.reduce_only:
                decision = self._check_reduce_only(order, account, instrument, quote)
                self._audit(order, decision, now)
                return decision

            if order.qty <= 0:
                decision = RiskDecision(False, None, "rejected: non-positive quantity")
                self._audit(order, decision, now)
                return decision

            if self.halted:
                decision = RiskDecision(False, None, f"rejected: halted ({self.halt_reason})")
                self._audit(order, decision, now)
                return decision

            price = self._price_for(order, quote)
            if price is None or price <= 0:
                decision = RiskDecision(False, None, "rejected: no quote and no limit_price to risk-check against")
                self._audit(order, decision, now)
                return decision

            decision = self._check_new_risk(order, account, instrument, quote, price, now)
            self._audit(order, decision, now)
            return decision

    def on_fill(self, fill: Fill, account: AccountState) -> None:
        with self._lock:
            entry = self._open_orders.get(fill.client_order_id)
            if entry is not None:
                symbol, remaining = entry
                remaining -= fill.qty
                if remaining <= 1e-9:
                    self._open_orders.pop(fill.client_order_id, None)
                else:
                    self._open_orders[fill.client_order_id] = (symbol, remaining)

            self._strategy_notional[fill.strategy_id] = self._strategy_notional.get(
                fill.strategy_id, 0.0
            ) + fill.side.sign * fill.qty * fill.price

            pos = self._shadow_positions.setdefault(fill.symbol, Position(fill.symbol))
            realized_gross = pos.apply_fill(fill)
            if realized_gross != 0.0:
                net = realized_gross - fill.fee
                if net < 0:
                    self._consecutive_losses += 1
                    if self._consecutive_losses >= self.limits.max_consecutive_losses and self._cooldown_until is None:
                        anchor = fill.ts or self._last_now or pd.Timestamp.now(tz="UTC")
                        self._cooldown_until = anchor + pd.Timedelta(minutes=self.limits.consecutive_loss_cooldown_minutes)
                elif net > 0:
                    self._consecutive_losses = 0

    def on_mark(self, account: AccountState, now: pd.Timestamp) -> None:
        with self._lock:
            self._refresh(now, account.equity)
            equity = account.equity
            if self._hwm is None:
                self._hwm = equity
            else:
                self._hwm = max(self._hwm, equity)

            if self._hwm and self._hwm > 0:
                drawdown = (self._hwm - equity) / self._hwm
                if drawdown >= self.limits.max_drawdown_pct and not self._killed:
                    self._trigger_kill(
                        f"drawdown {drawdown:.1%} reached max_drawdown_pct {self.limits.max_drawdown_pct:.1%} "
                        f"(equity ${equity:,.2f} vs HWM ${self._hwm:,.2f})"
                    )

            if self._day_start_equity is None:
                self._day_start_equity = equity
            elif self._day_start_equity > 0:
                daily_pnl_pct = (equity - self._day_start_equity) / self._day_start_equity
                if daily_pnl_pct <= -self.limits.max_daily_loss_pct and not self._daily_halted:
                    self._daily_halted = True
                    self._daily_halt_reason = (
                        f"daily loss {daily_pnl_pct:.1%} reached -max_daily_loss_pct "
                        f"{-self.limits.max_daily_loss_pct:.1%}"
                    )

    @property
    def halted(self) -> bool:
        with self._lock:
            return self._killed or self._daily_halted or self._cooldown_until is not None

    # ------------------------------------------------------------------------------ extras

    @property
    def flatten(self) -> bool:
        """True once the drawdown kill switch has tripped: the caller should flatten everything."""
        with self._lock:
            return self._killed

    @property
    def halt_reason(self) -> str:
        with self._lock:
            return self._halt_reason_text()

    def kill(self, reason: str = "manual kill switch") -> None:
        """Manually trip the kill switch (halt + flatten)."""
        with self._lock:
            self._trigger_kill(reason)

    def reset(self) -> None:
        """Operator override: clear all halts (kill switch, daily loss, cooldown). Does NOT
        reset the high-water mark or the daily baseline - those are facts about the account."""
        with self._lock:
            self._killed = False
            self._kill_reason = ""
            self._daily_halted = False
            self._daily_halt_reason = ""
            self._cooldown_until = None
            self._consecutive_losses = 0

    # ------------------------------------------------------------------------------ internals

    def _refresh(self, now: pd.Timestamp, equity: Optional[float]) -> None:
        self._last_now = now
        if self._cooldown_until is not None and now >= self._cooldown_until:
            self._cooldown_until = None

        key = self._day_key_for(now)
        if self._day_key is None:
            self._day_key = key
        elif key != self._day_key:
            self._day_key = key
            self._daily_halted = False
            self._daily_halt_reason = ""
            self._day_start_equity = equity  # reset the baseline as of the new trading day

    def _day_key_for(self, now: pd.Timestamp):
        ts = now.tz_convert("UTC") if now.tzinfo else now.tz_localize("UTC")
        shifted = ts - pd.Timedelta(hours=self.limits.daily_reset_hour_utc)
        return shifted.date()

    def _trigger_kill(self, reason: str) -> None:
        self._killed = True
        self._kill_reason = reason

    def _halt_reason_text(self) -> str:
        reasons = []
        if self._killed:
            reasons.append(f"kill switch: {self._kill_reason}")
        if self._daily_halted:
            reasons.append(self._daily_halt_reason)
        if self._cooldown_until is not None:
            reasons.append(f"consecutive-loss cooldown until {self._cooldown_until}")
        return "; ".join(reasons) if reasons else ""

    def _instrument_for(self, symbol: str) -> Instrument:
        if symbol in self.instruments:
            return self.instruments[symbol]
        return Instrument.infer(symbol)

    @staticmethod
    def _price_for(order: Order, quote: Optional[Quote]) -> Optional[float]:
        if quote is not None:
            return quote.mid
        if order.limit_price is not None:
            return order.limit_price
        return None

    def _throttle_ok(self, now: pd.Timestamp) -> bool:
        window_start = now - pd.Timedelta(seconds=60)
        self._order_times = [t for t in self._order_times if t > window_start]
        if len(self._order_times) >= self.limits.max_orders_per_minute:
            return False
        self._order_times.append(now)
        return True

    def _within_trading_hours(self, now: pd.Timestamp) -> tuple[bool, str]:
        th = self.limits.trading_hours
        tz = ZoneInfo(th.tz)
        ts = now.tz_convert("UTC") if now.tzinfo else now.tz_localize("UTC")
        local = ts.tz_convert(tz)
        if not th.trade_weekends and local.weekday() >= 5:
            return False, f"outside trading hours (weekend, {th.tz})"
        if th.allow_extended_hours:
            start_s, end_s = th.extended_start, th.extended_end
        else:
            start_s, end_s = th.regular_start, th.regular_end
        start_t, end_t = dt.time.fromisoformat(start_s), dt.time.fromisoformat(end_s)
        t = local.time()
        if not (start_t <= t <= end_t):
            return False, f"outside trading hours ({start_s}-{end_s} {th.tz}, now {t.isoformat()[:5]})"
        return True, ""

    @staticmethod
    def _max_qty_within_notional(current_qty: float, sign: int, price: float, max_notional: float) -> float:
        """Max additional qty (>=0, in the direction `sign`) keeping |current_qty + sign*q|*price
        within `max_notional`. Never negative; unconstrained buys into a large short return a big
        number (i.e. no clamp), which is correct since reducing exposure is never the binding case.
        """
        if price <= 0:
            return 0.0
        max_notional = max(max_notional, 0.0)
        limit_qty = max_notional / price
        if sign > 0:
            q = limit_qty - current_qty
        else:
            q = current_qty + limit_qty
        return max(q, 0.0)

    def _check_reduce_only(
        self, order: Order, account: AccountState, instrument: Instrument, quote: Optional[Quote]
    ) -> RiskDecision:
        pos = account.positions.get(order.symbol)
        current_qty = pos.qty if pos else 0.0
        max_close_qty = abs(current_qty)
        if max_close_qty <= 0:
            return RiskDecision(False, None, "rejected: reduce_only order but no open position to reduce")

        qty = min(order.qty, max_close_qty) if order.qty > 0 else max_close_qty
        new_limit_price = order.limit_price
        note = "approved (reduce_only)" if qty == order.qty else "approved (reduce_only, resized to available position)"

        if order.type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.limit_price is not None and quote is not None and quote.mid > 0:
            band = self.limits.fat_finger_band_bps
            dev_bps = abs(order.limit_price - quote.mid) / quote.mid * 1e4
            if dev_bps > band:
                direction = 1 if order.limit_price > quote.mid else -1
                new_limit_price = quote.mid * (1 + direction * band / 1e4)
                note += "; fat-finger price clamped to band edge"

        new_order = replace(order, qty=qty, limit_price=new_limit_price)
        self._open_orders[order.client_order_id] = (order.symbol, qty)
        return RiskDecision(True, new_order, note)

    def _check_new_risk(
        self,
        order: Order,
        account: AccountState,
        instrument: Instrument,
        quote: Optional[Quote],
        price: float,
        now: pd.Timestamp,
    ) -> RiskDecision:
        equity = account.equity
        notes: list[str] = []

        # 1. trading hours (equities only; crypto is 24/7)
        if instrument.asset_class is AssetClass.EQUITY and self.limits.trading_hours.enabled:
            ok, reason = self._within_trading_hours(now)
            if not ok:
                return RiskDecision(False, None, f"rejected: {reason}")

        # 2. throttle
        if not self._throttle_ok(now):
            return RiskDecision(
                False, None, f"rejected: throttled, max {self.limits.max_orders_per_minute} orders/minute exceeded"
            )

        # 3. fat-finger band (limit-priced orders only)
        if order.type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.limit_price is not None and quote is not None and quote.mid > 0:
            band = self.limits.fat_finger_band_bps
            dev_bps = abs(order.limit_price - quote.mid) / quote.mid * 1e4
            if dev_bps > band:
                return RiskDecision(
                    False, None,
                    f"rejected: fat-finger, limit price {dev_bps:.0f}bps from mid exceeds {band:.0f}bps band",
                )

        # 4. min notional floor
        if order.qty * price < self.limits.min_notional_usd:
            return RiskDecision(
                False, None,
                f"rejected: order notional ${order.qty*price:,.2f} below min_notional_usd ${self.limits.min_notional_usd:,.2f}",
            )

        qty = order.qty

        # 5. max single-order notional
        max_qty_order = self.limits.max_order_notional_usd / price
        if qty > max_qty_order:
            qty = max_qty_order
            notes.append(f"max_order_notional_usd (${self.limits.max_order_notional_usd:,.0f})")

        pos = account.positions.get(order.symbol)
        current_qty = pos.qty if pos else 0.0

        # 6. long-only enforcement for non-shortable instruments
        if self.limits.enforce_long_only_for_non_shortable and not instrument.shortable and order.side is Side.SELL:
            max_sell = max(current_qty, 0.0)
            if qty > max_sell:
                qty = max_sell
                notes.append("long_only (non-shortable instrument)")
            if qty <= 0:
                return RiskDecision(False, None, "rejected: long-only instrument, no long position to sell")

        # 7. per-symbol position notional cap (tighter of absolute $ and % equity)
        max_pos_notional = min(self.limits.max_position_notional_usd, self.limits.max_position_pct_equity * equity)
        allowed = self._max_qty_within_notional(current_qty, order.side.sign, price, max_pos_notional)
        if allowed < qty:
            qty = max(allowed, 0.0)
            notes.append(f"max_position_notional (${max_pos_notional:,.0f})")
        if qty <= 0:
            return RiskDecision(False, None, "rejected: per-symbol position notional cap already reached")

        # 8. crypto asset-class cap
        if instrument.asset_class is AssetClass.CRYPTO:
            crypto_other = sum(
                abs(p.qty * p.avg_price)
                for sym, p in account.positions.items()
                if sym != order.symbol and self._instrument_for(sym).asset_class is AssetClass.CRYPTO
            )
            crypto_cap = self.limits.crypto_max_pct_equity * equity
            budget = max(crypto_cap - crypto_other, 0.0)
            allowed = self._max_qty_within_notional(current_qty, order.side.sign, price, budget)
            if allowed < qty:
                qty = max(allowed, 0.0)
                notes.append(f"crypto_max_pct_equity ({self.limits.crypto_max_pct_equity:.0%})")
            if qty <= 0:
                return RiskDecision(False, None, "rejected: crypto asset-class exposure cap already reached")

        # 9. gross exposure cap
        gross_other = sum(abs(p.qty * p.avg_price) for sym, p in account.positions.items() if sym != order.symbol)
        gross_cap = self.limits.max_gross_exposure_pct * equity
        budget = max(gross_cap - gross_other, 0.0)
        allowed = self._max_qty_within_notional(current_qty, order.side.sign, price, budget)
        if allowed < qty:
            qty = max(allowed, 0.0)
            notes.append(f"max_gross_exposure_pct ({self.limits.max_gross_exposure_pct:.0%})")
        if qty <= 0:
            return RiskDecision(False, None, "rejected: gross exposure cap already reached")

        # 10. net exposure cap (signed)
        net_other = sum(p.qty * p.avg_price for sym, p in account.positions.items() if sym != order.symbol)
        net_cap = self.limits.max_net_exposure_pct * equity
        base = net_other + current_qty * price
        if order.side.sign > 0:
            max_q = (net_cap - base) / price if price > 0 else 0.0
        else:
            max_q = (base + net_cap) / price if price > 0 else 0.0
        max_q = max(max_q, 0.0)
        if max_q < qty:
            qty = max_q
            notes.append(f"max_net_exposure_pct ({self.limits.max_net_exposure_pct:.0%})")
        if qty <= 0:
            return RiskDecision(False, None, "rejected: net exposure cap already reached")

        # 11. per-strategy capital allocation cap
        cap_pct = self.limits.per_strategy_capital_pct.get(order.strategy_id, self.limits.default_strategy_capital_pct)
        strategy_cap = cap_pct * equity
        current_strategy_notional = abs(self._strategy_notional.get(order.strategy_id, 0.0))
        budget = max(strategy_cap - current_strategy_notional, 0.0)
        if qty * price > budget:
            qty = budget / price if price > 0 else 0.0
            notes.append(f"per_strategy_capital_pct ({cap_pct:.0%} for {order.strategy_id!r})")
        if qty <= 0:
            return RiskDecision(False, None, f"rejected: per-strategy capital allocation cap reached for {order.strategy_id!r}")

        # 12. max open orders per symbol
        open_count = sum(1 for sym, _ in self._open_orders.values() if sym == order.symbol)
        if open_count >= self.limits.max_open_orders_per_symbol:
            return RiskDecision(
                False, None,
                f"rejected: max_open_orders_per_symbol ({self.limits.max_open_orders_per_symbol}) reached for {order.symbol}",
            )

        # finalize: lot-round and re-check the min-notional floor
        qty = lot_round(qty, instrument)
        if qty <= 0 or qty * price < self.limits.min_notional_usd:
            return RiskDecision(False, None, "rejected: resized order fell below lot size / min notional")

        final_order = replace(order, qty=qty)
        self._open_orders[order.client_order_id] = (order.symbol, qty)
        reason = "approved" if not notes else f"approved, resized down to satisfy: {'; '.join(notes)}"
        return RiskDecision(True, final_order, reason)

    def _audit(self, order: Order, decision: RiskDecision, now: pd.Timestamp) -> None:
        self.audit.append(
            {
                "ts": now,
                "symbol": order.symbol,
                "side": order.side.value,
                "strategy_id": order.strategy_id,
                "requested_qty": order.qty,
                "approved": decision.approved,
                "approved_qty": decision.order.qty if decision.order else 0.0,
                "reason": decision.reason,
            }
        )
        if len(self.audit) > 10_000:
            del self.audit[: len(self.audit) - 10_000]
