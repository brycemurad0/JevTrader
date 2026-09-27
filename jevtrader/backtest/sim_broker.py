"""SimBroker: a `Broker` implementation for the backtester with two honest, conservative fill
models, selected automatically by the kind of market data an order's symbol is receiving.

**Bar-mode** (order's symbol is driven by `Bar` events):
  * Market orders fill at the **next** bar's open (never the bar they were created on -- that
    would be lookahead), plus half-spread + fixed slippage + a sqrt participation-impact term,
    all in bps of price. Fill quantity is capped at `slippage.max_participation` of that bar's
    volume; any unfilled remainder keeps resting for the bar after that.
  * Limit orders fill only if a later bar's range trades **strictly through** the limit (`low <
    limit` for a buy, `high > limit` for a sell; set `strict_through=False` for touch-fills), at
    the limit price, as MAKER liquidity. Same no-lookahead rule: never the creation bar.

**Quote/trade-mode** (order's symbol is driven by `Quote`/`Trade`/`OrderBook` events):
  * Market or marketable limit orders (crossing the current touch) become eligible `latency_ms`
    after submission and then take liquidity at the opposite touch (TAKER), capped by the size
    resting there (partial fills rest until more liquidity appears, unless `tif` is IOC/FOK).
  * Non-marketable (resting) limit orders are MAKER: at the first eligible tick, the L1 size on
    their side of the book is snapshotted as "queue ahead". Each subsequent trade at that price
    depletes the queue; once the queue is exhausted, further trade volume at that price fills the
    order directly. A trade printing strictly through the limit is treated as instant queue
    exhaustion (the market moved past every order ahead of ours). This is a standard, simplified
    L1-only queue model -- it has no visibility into the true order queue, only the size Alpaca
    (or any L1 feed) reports at that price.
  * `post_only` orders that would already cross the book at submission are rejected immediately.

**Instrument constraints** (both modes): quantity is floored to `lot_size`; orders below
`min_notional` are rejected; `shortable=False` instruments (Alpaca crypto) reject any SELL that
would push the position negative; buying power is cash-only (no margin) unless `allow_leverage`
is set, in which case `leverage` sets max gross exposure as a multiple of equity.

Every fill goes through `FeeModel.fee(...)` with the correct `Liquidity` flag. If the fee model
is (or wraps, via `CompositeFees`) an `AlpacaCryptoFees`, its `thirty_day_volume` is kept in sync
with a rolling window of this broker's own crypto fills before every fee calculation, so tiered
fees respond to trading activity within the backtest itself.

Simplifying assumptions (so results are honest about what they are): no partial-day equity
auction modelling, no exchange-specific queue priority beyond L1 size, stop orders trigger on
bar high/low (bar-mode) or last trade/quote price (quote-mode) and then fill as if they were a
plain market/limit order referenced off the stop price rather than the next tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Union

import pandas as pd

from jevtrader.core.broker import Broker, TradingMode
from jevtrader.core.fees import AlpacaCryptoFees, CompositeFees, FeeModel
from jevtrader.core.types import (
    AccountState,
    AssetClass,
    Bar,
    Fill,
    Instrument,
    Liquidity,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
    Trade,
)

MarketEvent = Union[Bar, Quote, Trade, OrderBook]


@dataclass
class SlippageModel:
    """Bar-mode market-order execution cost model, all terms in basis points of price.

    `total_bps = half_spread_bps + slippage_bps + impact_coeff_bps * sqrt(participation)`,
    where `participation = qty / bar_volume`. `max_participation` caps how much of a single
    bar's volume one fill may consume; the remainder stays pending for later bars.
    """

    half_spread_bps: float = 1.0
    slippage_bps: float = 0.5
    impact_coeff_bps: float = 2.0
    max_participation: float = 1.0


@dataclass
class _OrderState:
    order: Order
    eligible_ts: pd.Timestamp
    created_bar_ts: Optional[pd.Timestamp] = None
    queue_ahead: Optional[float] = None
    queue_initialized: bool = False
    triggered: bool = False


class SimBroker(Broker):
    """Backtest `Broker`. See module docstring for the fill models and constraints enforced."""

    mode = TradingMode.BACKTEST

    def __init__(
        self,
        fee_model: FeeModel,
        instruments: Optional[Mapping[str, Instrument]] = None,
        initial_cash: float = 100_000.0,
        latency_ms: float = 50.0,
        slippage: Optional[SlippageModel] = None,
        allow_leverage: bool = False,
        leverage: float = 1.0,
        strict_through: bool = True,
        through_epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        self.fee_model = fee_model
        self.instruments: dict[str, Instrument] = dict(instruments or {})
        self.initial_cash = initial_cash
        self.latency_ms = latency_ms
        self.slippage = slippage if slippage is not None else SlippageModel()
        self.allow_leverage = allow_leverage
        self.leverage = leverage
        self.strict_through = strict_through
        self.through_epsilon = through_epsilon

        self._cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._pending: dict[str, list[_OrderState]] = {}
        self._last_bar: dict[str, Bar] = {}
        self._last_quote: dict[str, Quote] = {}
        self._last_trade: dict[str, Trade] = {}
        self._now: Optional[pd.Timestamp] = None
        self._crypto_fill_log: list[tuple[pd.Timestamp, float]] = []
        self.fills: list[Fill] = []

    # ------------------------------------------------------------------ instrument helpers

    def _instrument(self, symbol: str) -> Instrument:
        if symbol not in self.instruments:
            self.instruments[symbol] = Instrument.infer(symbol)
        return self.instruments[symbol]

    @staticmethod
    def _round_qty(qty: float, lot_size: float) -> float:
        if lot_size <= 0:
            return qty
        return math.floor(qty / lot_size + 1e-9) * lot_size

    def mark_price(self, symbol: str) -> Optional[float]:
        if symbol in self._last_trade:
            return self._last_trade[symbol].price
        if symbol in self._last_quote:
            return self._last_quote[symbol].mid
        if symbol in self._last_bar:
            return self._last_bar[symbol].close
        return None

    # ------------------------------------------------------------------ Broker interface

    def submit(self, order: Order) -> Order:
        order.created_ts = self._now
        order.broker_order_id = order.broker_order_id or order.client_order_id
        instrument = self._instrument(order.symbol)

        order.qty = self._round_qty(order.qty, instrument.lot_size)
        if order.qty <= 0:
            return self._reject(order, "qty_rounds_to_zero_at_lot_size")

        ref_price = self._reference_price(order)
        if ref_price is not None and order.qty * ref_price < instrument.min_notional:
            return self._reject(order, "below_min_notional")

        if not instrument.shortable and order.side is Side.SELL:
            pos = self._positions.get(order.symbol)
            available = max(pos.qty, 0.0) if pos else 0.0
            if order.qty - available > 1e-9:
                return self._reject(order, "would_short_non_shortable_instrument")

        if order.post_only and self._is_marketable(order, self._last_quote.get(order.symbol)):
            return self._reject(order, "post_only_would_cross")

        if ref_price is not None and not self._has_buying_power(order, ref_price):
            return self._reject(order, "insufficient_buying_power")

        order.status = OrderStatus.NEW
        order.reject_reason = ""
        self._orders[order.client_order_id] = order
        state = _OrderState(
            order=order,
            eligible_ts=(self._now + pd.Timedelta(milliseconds=self.latency_ms)) if self._now is not None else pd.Timestamp.min.tz_localize("UTC"),
            created_bar_ts=self._last_bar[order.symbol].ts if order.symbol in self._last_bar else None,
        )
        self._pending.setdefault(order.symbol, []).append(state)
        self._emit_order(order)
        return order

    def cancel(self, client_order_id: str) -> None:
        order = self._orders.get(client_order_id)
        if order is None or not order.status.is_open:
            return
        order.status = OrderStatus.CANCELED
        lst = self._pending.get(order.symbol, [])
        self._pending[order.symbol] = [s for s in lst if s.order.client_order_id != client_order_id]
        self._emit_order(order)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        for oid, order in list(self._orders.items()):
            if order.status.is_open and (symbol is None or order.symbol == symbol):
                self.cancel(oid)

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        return [o for o in self._orders.values() if o.status.is_open and (symbol is None or o.symbol == symbol)]

    def positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._positions.items() if abs(p.qty) > 1e-12}

    def account(self) -> AccountState:
        equity = self._cash
        for sym, pos in self._positions.items():
            if abs(pos.qty) < 1e-12:
                continue
            price = self.mark_price(sym)
            equity += pos.qty * (price if price is not None else pos.avg_price)
        buying_power = self._cash if not self.allow_leverage else max(equity, 0.0) * self.leverage
        return AccountState(cash=self._cash, equity=equity, buying_power=buying_power, positions=dict(self._positions), ts=self._now)

    # ------------------------------------------------------------------ submit-time checks

    def _reference_price(self, order: Order) -> Optional[float]:
        if order.limit_price is not None:
            return order.limit_price
        return self.mark_price(order.symbol)

    def _has_buying_power(self, order: Order, ref_price: float) -> bool:
        if self.allow_leverage:
            return True
        if order.side is Side.BUY:
            return order.qty * ref_price <= self._cash + 1e-6
        pos = self._positions.get(order.symbol)
        current = pos.qty if pos else 0.0
        opening = max(0.0, order.qty - max(current, 0.0))  # only opening/adding to a short needs collateral
        return opening * ref_price <= self._cash + 1e-6

    @staticmethod
    def _is_marketable(order: Order, quote: Optional[Quote]) -> bool:
        if order.type is OrderType.MARKET:
            return True
        if order.type not in (OrderType.LIMIT, OrderType.STOP_LIMIT) or order.limit_price is None or quote is None:
            return False
        return order.limit_price >= quote.ask if order.side is Side.BUY else order.limit_price <= quote.bid

    def _reject(self, order: Order, reason: str) -> Order:
        order.status = OrderStatus.REJECTED
        order.reject_reason = reason
        self._emit_order(order)
        return order

    # ------------------------------------------------------------------ market data intake + fills

    def on_market_event(self, event: MarketEvent) -> list[Fill]:
        """Update internal market state from `event` and attempt to fill orders that were
        already resting *before* this event (never orders a strategy submits while reacting to
        this same event -- the engine calls this before dispatching the event to strategies)."""
        self._now = event.ts
        if isinstance(event, Bar):
            self._last_bar[event.symbol] = event
            fills = self._process_bar_fills(event)
        elif isinstance(event, Quote):
            self._last_quote[event.symbol] = event
            fills = self._process_quote_fills(event)
        elif isinstance(event, Trade):
            self._last_trade[event.symbol] = event
            fills = self._process_trade_fills(event)
        elif isinstance(event, OrderBook):
            q = event.to_quote()
            fills = []
            if q is not None:
                self._last_quote[event.symbol] = q
                fills = self._process_quote_fills(q)
        else:  # pragma: no cover - defensive
            fills = []
        self.fills.extend(fills)
        return fills

    # -- stop trigger helper (shared by all three processors) -----------------------------

    def _stop_kind(self, state: _OrderState, lo: Optional[float], hi: Optional[float]) -> str:
        """Returns 'market', 'limit' or 'untriggered'/'wait' for the order's effective kind."""
        order = state.order
        if order.type is OrderType.MARKET:
            return "market"
        if order.type is OrderType.LIMIT:
            return "limit"
        # STOP / STOP_LIMIT
        if not state.triggered:
            if lo is None or hi is None or order.stop_price is None:
                return "wait"
            triggered = hi >= order.stop_price if order.side is Side.BUY else lo <= order.stop_price
            if not triggered:
                return "wait"
            state.triggered = True
        return "market" if order.type is OrderType.STOP else "limit"

    # -- bar mode ---------------------------------------------------------------------------

    def _bar_market_fill_price(self, order: Order, bar: Bar, reference: float) -> float:
        participation = min(1.0, order.remaining / bar.volume) if bar.volume > 0 else 1.0
        impact_bps = self.slippage.impact_coeff_bps * math.sqrt(max(participation, 0.0))
        total_bps = self.slippage.half_spread_bps + self.slippage.slippage_bps + impact_bps
        return reference * (1.0 + order.side.sign * total_bps / 1e4)

    def _bar_trades_through(self, order: Order, bar: Bar) -> bool:
        eps = self.through_epsilon
        if order.side is Side.BUY:
            return bar.low < order.limit_price - eps if self.strict_through else bar.low <= order.limit_price + eps
        return bar.high > order.limit_price + eps if self.strict_through else bar.high >= order.limit_price - eps

    def _process_bar_fills(self, bar: Bar) -> list[Fill]:
        fills: list[Fill] = []
        remaining: list[_OrderState] = []
        for state in self._pending.get(bar.symbol, []):
            order = state.order
            if not order.status.is_open:
                continue
            if state.created_bar_ts is not None and bar.ts <= state.created_bar_ts:
                remaining.append(state)  # no lookahead: only a strictly later bar may fill this
                continue
            kind = self._stop_kind(state, bar.low, bar.high)
            if kind == "market":
                ref = bar.open if order.type is OrderType.MARKET else order.stop_price
                price = self._bar_market_fill_price(order, bar, ref)
                cap = bar.volume * self.slippage.max_participation if bar.volume > 0 else order.remaining
                qty = min(order.remaining, max(cap, 0.0)) if bar.volume > 0 else order.remaining
                if qty > 0:
                    fills.append(self._make_fill(order, qty, price, Liquidity.TAKER, bar.ts))
            elif kind == "limit" and self._bar_trades_through(order, bar):
                fills.append(self._make_fill(order, order.remaining, order.limit_price, Liquidity.MAKER, bar.ts))
            if order.status.is_open:
                remaining.append(state)
        self._pending[bar.symbol] = remaining
        return fills

    # -- quote / trade mode -------------------------------------------------------------------

    @staticmethod
    def _is_marketable_against(order: Order, quote: Quote) -> bool:
        if order.type is OrderType.MARKET:
            return True
        if order.limit_price is None:
            return False
        return order.limit_price >= quote.ask if order.side is Side.BUY else order.limit_price <= quote.bid

    @staticmethod
    def _at_or_behind_touch(order: Order, quote: Quote) -> bool:
        if order.limit_price is None:
            return False
        return order.limit_price <= quote.bid + 1e-12 if order.side is Side.BUY else order.limit_price >= quote.ask - 1e-12

    def _process_quote_fills(self, quote: Quote) -> list[Fill]:
        fills: list[Fill] = []
        remaining: list[_OrderState] = []
        for state in self._pending.get(quote.symbol, []):
            order = state.order
            if not order.status.is_open:
                continue
            if quote.ts < state.eligible_ts:
                remaining.append(state)
                continue
            kind = self._stop_kind(state, quote.bid, quote.ask)
            if kind == "wait":
                remaining.append(state)
                continue
            marketable = kind == "market" or self._is_marketable_against(order, quote)
            if marketable:
                side_avail = quote.ask_size if order.side is Side.BUY else quote.bid_size
                if order.tif is TimeInForce.FOK and side_avail < order.remaining - 1e-9:
                    order.status = OrderStatus.CANCELED
                    self._emit_order(order)
                    continue
                price = quote.ask if order.side is Side.BUY else quote.bid
                qty = min(order.remaining, side_avail) if side_avail > 0 else order.remaining
                if qty > 0:
                    fills.append(self._make_fill(order, qty, price, Liquidity.TAKER, quote.ts))
                if order.status.is_open and order.tif in (TimeInForce.IOC, TimeInForce.FOK):
                    order.status = OrderStatus.CANCELED
                    self._emit_order(order)
                    continue
            elif not state.queue_initialized:
                state.queue_ahead = (quote.bid_size if order.side is Side.BUY else quote.ask_size) if self._at_or_behind_touch(order, quote) else 0.0
                state.queue_initialized = True
            if order.status.is_open:
                remaining.append(state)
        self._pending[quote.symbol] = remaining
        return fills

    def _process_trade_fills(self, trade: Trade) -> list[Fill]:
        fills: list[Fill] = []
        remaining: list[_OrderState] = []
        for state in self._pending.get(trade.symbol, []):
            order = state.order
            if not order.status.is_open:
                continue
            is_untriggered_stop_limit = order.type is OrderType.STOP_LIMIT and not state.triggered
            if order.type not in (OrderType.LIMIT, OrderType.STOP_LIMIT) or is_untriggered_stop_limit:
                # market-kind orders and not-yet-triggered stop-limits are handled purely off quotes
                remaining.append(state)
                continue
            if trade.ts < state.eligible_ts or not state.queue_initialized or order.limit_price is None:
                remaining.append(state)
                continue
            through = (order.side is Side.BUY and trade.price < order.limit_price) or (order.side is Side.SELL and trade.price > order.limit_price)
            at_price = abs(trade.price - order.limit_price) < 1e-9
            if through:
                fills.append(self._make_fill(order, order.remaining, order.limit_price, Liquidity.MAKER, trade.ts))
            elif at_price:
                prev_queue = state.queue_ahead or 0.0
                overflow = trade.size - max(prev_queue, 0.0)
                state.queue_ahead = prev_queue - trade.size
                if overflow > 0:
                    fills.append(self._make_fill(order, min(order.remaining, overflow), order.limit_price, Liquidity.MAKER, trade.ts))
            if order.status.is_open:
                remaining.append(state)
        self._pending[trade.symbol] = remaining
        return fills

    # ------------------------------------------------------------------ fill application

    def _crypto_fee_model(self) -> Optional[AlpacaCryptoFees]:
        fm = self.fee_model
        if isinstance(fm, AlpacaCryptoFees):
            return fm
        if isinstance(fm, CompositeFees) and isinstance(fm.crypto, AlpacaCryptoFees):
            return fm.crypto
        return None

    def _refresh_crypto_tier(self, instrument: Instrument, ts: pd.Timestamp) -> None:
        fee_model = self._crypto_fee_model()
        if fee_model is None or instrument.asset_class is not AssetClass.CRYPTO:
            return
        cutoff = ts - pd.Timedelta(days=30)
        self._crypto_fill_log = [(t, n) for (t, n) in self._crypto_fill_log if t >= cutoff]
        fee_model.thirty_day_volume = sum(n for _, n in self._crypto_fill_log)

    def _make_fill(self, order: Order, qty: float, price: float, liquidity: Liquidity, ts: pd.Timestamp) -> Fill:
        qty = max(0.0, min(qty, order.remaining))
        instrument = self._instrument(order.symbol)
        self._refresh_crypto_tier(instrument, ts)
        fee = self.fee_model.fee(instrument, order.side, qty, price, liquidity)
        fill = Fill(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            fee=fee,
            liquidity=liquidity,
            ts=ts,
            strategy_id=order.strategy_id,
        )
        new_filled = order.filled_qty + qty
        order.avg_fill_price = (order.avg_fill_price * order.filled_qty + price * qty) / new_filled if new_filled > 0 else 0.0
        order.filled_qty = new_filled
        order.status = OrderStatus.FILLED if order.remaining <= 1e-9 else OrderStatus.PARTIALLY_FILLED

        pos = self._positions.setdefault(fill.symbol, Position(fill.symbol))
        pos.apply_fill(fill)
        self._cash += -fill.side.sign * fill.qty * fill.price - fill.fee

        if instrument.asset_class is AssetClass.CRYPTO and self._crypto_fee_model() is not None:
            self._crypto_fill_log.append((ts, qty * price))

        self._emit_fill(fill)
        self._emit_order(order)
        return fill

    # ------------------------------------------------------------------ public read accessors
    # (used by BacktestContext so the engine never reaches into underscore-prefixed internals)

    def instrument(self, symbol: str) -> Instrument:
        return self._instrument(symbol)

    def position_or_flat(self, symbol: str) -> Position:
        return self._positions.get(symbol) or Position(symbol)

    def last_quote(self, symbol: str) -> Optional[Quote]:
        return self._last_quote.get(symbol)

    def last_trade(self, symbol: str) -> Optional[Trade]:
        return self._last_trade.get(symbol)

    def last_bar(self, symbol: str) -> Optional[Bar]:
        return self._last_bar.get(symbol)

    def all_orders(self) -> list[Order]:
        return list(self._orders.values())
