"""AlpacaBroker: the `Broker` implementation backed by alpaca-py's `TradingClient`.

One class serves both paper and live trading, and both stocks and crypto ("BTC/USD"-style
symbols): Alpaca's trading API already accepts crypto symbols in that slash format, so no symbol
translation is needed between core `Order.symbol` and the alpaca-py request.

Design notes (see docs/ARCHITECTURE.md and jevtrader/core/broker.py for the contract):

* The constructor takes an injected `client` for tests; without one it builds a real
  `alpaca.trading.client.TradingClient` using `settings.keys_for(mode)`.
* Constructing with `mode=TradingMode.LIVE` raises `LiveTradingNotAllowed` unless
  `settings.live_allowed()` is True AND every id in `strategy_ids` has a passing LIVE promotion
  record (`jevtrader.live.promotion.assert_live_allowed`). LIVE construction with no
  `strategy_ids` at all is refused too -- there is no such thing as "live trading nothing in
  particular".
* Alpaca has no native post-only order type. We emulate it by checking a locally supplied
  `quote_provider` callback before submission: a post-only limit that would cross the current
  quote is rejected locally and never sent to Alpaca. The same crossing check also decides the
  maker/taker liquidity hint recorded for the order, which is looked up again when a fill for it
  arrives (Alpaca's `TradeUpdate` carries no liquidity or fee field, for either asset class, so
  every fee is computed locally via the injected `FeeModel`).
* Fills only arrive through `TradingStream` trade updates (`on_trade_update` / `run_stream`),
  matching how Alpaca actually reports paper and live fills; `submit()` only returns the broker's
  synchronous acknowledgement (new/accepted/rejected), never a fill.
* `reconcile()` re-syncs local open-order/position/account state from the broker's REST API. The
  caller (normally `jevtrader.live.runner.LiveRunner`) is expected to call it once on startup and
  then periodically -- AlpacaBroker does not spawn its own timer since it has no asyncio loop of
  its own.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Mapping, Optional, Sequence

import pandas as pd

from jevtrader.core.broker import Broker, TradingMode
from jevtrader.core.fees import FeeModel, default_fees
from jevtrader.core.types import (
    AccountState,
    AssetClass,
    Fill,
    Instrument,
    Liquidity,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
)

logger = logging.getLogger(__name__)

QuoteProvider = Callable[[str], Optional[Quote]]


class LiveTradingNotAllowed(RuntimeError):
    """Raised when constructing (or using) an AlpacaBroker in LIVE mode without both an explicit
    operator confirmation (`JEV_LIVE_CONFIRM`) and a passing LIVE promotion record."""


def _val(x: Any) -> Any:
    """`x.value` for an enum (alpaca-py or core), `x` unchanged for a plain string -- lets the
    mapping helpers accept either real alpaca-py model objects or simple duck-typed test fakes."""
    return x.value if hasattr(x, "value") else x


def _f(x: Any, default: float = 0.0) -> float:
    return default if x is None else float(x)


# --------------------------------------------------------------------------------- TIF / type mapping

_CORE_TIF_TO_ALPACA_EQUITY = {
    TimeInForce.DAY: "day",
    TimeInForce.GTC: "gtc",
    TimeInForce.IOC: "ioc",
    TimeInForce.FOK: "fok",
}


def _map_tif(order: Order, instrument: Instrument) -> Any:
    from alpaca.trading.enums import TimeInForce as ATIF

    if instrument.asset_class is AssetClass.CRYPTO:
        # Alpaca crypto only supports gtc/ioc. day has no meaning for a 24/7 market -> gtc;
        # fok has no crypto equivalent -> ioc is the closest available semantics (execute what
        # you can immediately, cancel the rest, rather than requiring an all-or-nothing fill).
        if order.tif in (TimeInForce.DAY, TimeInForce.GTC):
            return ATIF.GTC
        return ATIF.IOC
    return ATIF(_CORE_TIF_TO_ALPACA_EQUITY[order.tif])


def _round_qty(qty: float, lot_size: float) -> float:
    if lot_size is None or lot_size <= 0:
        return qty
    steps = round(qty / lot_size)
    return round(steps * lot_size, 10)


def _crosses(order: Order, quote: Quote) -> bool:
    """Would this limit order take liquidity (cross the book) if sent right now?"""
    if order.limit_price is None:
        return False
    if order.side is Side.BUY:
        return order.limit_price >= quote.ask
    return order.limit_price <= quote.bid


def build_order_request(order: Order, instrument: Instrument) -> Any:
    """Map a core `Order` to the alpaca-py `*OrderRequest` for it. Pure function: no I/O, no
    broker state, so it is exercised directly in tests without a client."""
    from alpaca.trading.enums import OrderSide as ASide
    from alpaca.trading.requests import (
        LimitOrderRequest,
        MarketOrderRequest,
        StopLimitOrderRequest,
        StopOrderRequest,
    )

    side = ASide.BUY if order.side is Side.BUY else ASide.SELL
    tif = _map_tif(order, instrument)
    qty = _round_qty(order.qty, instrument.lot_size)
    common: dict[str, Any] = dict(
        symbol=order.symbol,
        qty=qty,
        side=side,
        time_in_force=tif,
        client_order_id=order.client_order_id,
    )

    is_crypto = instrument.asset_class is AssetClass.CRYPTO
    order_type = order.type

    if is_crypto and order_type is OrderType.STOP:
        # Alpaca crypto has no plain "stop" order type (market/limit/stop_limit only); the
        # standard workaround is stop_limit with the limit price pinned to the stop price, which
        # behaves like a stop-market once triggered for any reasonably liquid crypto pair.
        if order.stop_price is None:
            raise ValueError("stop order requires stop_price")
        limit_price = order.limit_price if order.limit_price is not None else order.stop_price
        return StopLimitOrderRequest(stop_price=order.stop_price, limit_price=limit_price, **common)

    if order_type is OrderType.MARKET:
        return MarketOrderRequest(**common)
    if order_type is OrderType.LIMIT:
        if order.limit_price is None:
            raise ValueError("limit order requires limit_price")
        return LimitOrderRequest(limit_price=order.limit_price, **common)
    if order_type is OrderType.STOP:
        if order.stop_price is None:
            raise ValueError("stop order requires stop_price")
        return StopOrderRequest(stop_price=order.stop_price, **common)
    if order_type is OrderType.STOP_LIMIT:
        if order.stop_price is None or order.limit_price is None:
            raise ValueError("stop_limit order requires stop_price and limit_price")
        return StopLimitOrderRequest(stop_price=order.stop_price, limit_price=order.limit_price, **common)
    raise ValueError(f"unsupported order type {order_type!r}")


# --------------------------------------------------------------------------------- alpaca -> core mapping

_ALPACA_TYPE_TO_CORE = {
    "market": OrderType.MARKET,
    "limit": OrderType.LIMIT,
    "stop": OrderType.STOP,
    "stop_limit": OrderType.STOP_LIMIT,
    "trailing_stop": OrderType.STOP,  # no core equivalent; closest analogue
}

_ALPACA_STATUS_TO_CORE = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.NEW,
    "pending_new": OrderStatus.NEW,
    "accepted_for_bidding": OrderStatus.NEW,
    "pending_replace": OrderStatus.NEW,
    "pending_cancel": OrderStatus.NEW,
    "stopped": OrderStatus.NEW,
    "calculated": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.CANCELED,
    "canceled": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
}

_CORE_TIF_VALUES = {t.value for t in TimeInForce}


def map_alpaca_order_to_core(resp: Any) -> Order:
    """Map an alpaca-py `Order` (or a duck-typed fake with the same attributes) to a fresh core
    `Order`. Used for reconciliation and for trade updates about orders we did not place
    ourselves this session."""
    side = Side.BUY if _val(resp.side) == "buy" else Side.SELL
    raw_type = _val(getattr(resp, "type", None) or getattr(resp, "order_type", None))
    otype = _ALPACA_TYPE_TO_CORE.get(raw_type, OrderType.MARKET)
    raw_tif = _val(getattr(resp, "time_in_force", None))
    tif = TimeInForce(raw_tif) if raw_tif in _CORE_TIF_VALUES else TimeInForce.GTC

    order = Order(
        symbol=resp.symbol,
        side=side,
        qty=_f(getattr(resp, "qty", None)),
        type=otype,
        limit_price=_f(resp.limit_price, None) if getattr(resp, "limit_price", None) is not None else None,
        stop_price=_f(resp.stop_price, None) if getattr(resp, "stop_price", None) is not None else None,
        tif=tif,
        client_order_id=resp.client_order_id,
    )
    _update_order_from_alpaca(order, resp)
    submitted_at = getattr(resp, "submitted_at", None)
    if submitted_at is not None:
        order.created_ts = pd.Timestamp(submitted_at)
    return order


def _update_order_from_alpaca(order: Order, resp: Any) -> None:
    """Mutate `order`'s broker-owned fields (status/fill state) in place from an alpaca-py
    response, without disturbing the object's identity (callers may hold references to it)."""
    order_id = getattr(resp, "id", None)
    if order_id is not None:
        order.broker_order_id = str(order_id)
    status_val = _val(getattr(resp, "status", None))
    if status_val in _ALPACA_STATUS_TO_CORE:
        order.status = _ALPACA_STATUS_TO_CORE[status_val]
    filled_qty = getattr(resp, "filled_qty", None)
    if filled_qty is not None:
        order.filled_qty = float(filled_qty)
    avg_price = getattr(resp, "filled_avg_price", None)
    if avg_price is not None:
        order.avg_fill_price = float(avg_price)
    if order.status is OrderStatus.REJECTED and not order.reject_reason:
        order.reject_reason = "rejected by broker"


def map_alpaca_position_to_core(resp: Any) -> Position:
    qty = abs(_f(getattr(resp, "qty", None)))
    if _val(getattr(resp, "side", "long")) == "short":
        qty = -qty
    return Position(symbol=resp.symbol, qty=qty, avg_price=_f(getattr(resp, "avg_entry_price", None)))


def map_alpaca_account_to_core(
    resp: Any, positions: Mapping[str, Position], now: Optional[pd.Timestamp] = None
) -> AccountState:
    return AccountState(
        cash=_f(getattr(resp, "cash", None)),
        equity=_f(getattr(resp, "equity", None)),
        buying_power=_f(getattr(resp, "buying_power", None)),
        positions=dict(positions),
        ts=now or pd.Timestamp.utcnow(),
    )


def _assert_live_ok(settings: Any, strategy_ids: Sequence[str]) -> None:
    if not settings.live_allowed():
        raise LiveTradingNotAllowed(
            "LIVE trading requires JEV_LIVE_CONFIRM=I_UNDERSTAND_THIS_IS_REAL_MONEY in the "
            "environment. This is a deliberate real-money speed bump; set it in .env, on your "
            "own machine, only when you mean it."
        )
    if not strategy_ids:
        raise LiveTradingNotAllowed(
            "LIVE trading requires at least one promoted strategy_id; got none. Pass the ids of "
            "the strategies this broker instance will route live orders for."
        )
    from jevtrader.live.promotion import assert_live_allowed

    for strategy_id in strategy_ids:
        assert_live_allowed(strategy_id, settings)


class AlpacaBroker(Broker):
    """`Broker` backed by `alpaca.trading.client.TradingClient`, for both PAPER and LIVE modes,
    and both equities and crypto."""

    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        *,
        client: Optional[Any] = None,
        settings: Optional[Any] = None,
        fee_model: Optional[FeeModel] = None,
        quote_provider: Optional[QuoteProvider] = None,
        trading_stream: Optional[Any] = None,
        instruments: Optional[Mapping[str, Instrument]] = None,
        strategy_ids: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if mode is TradingMode.BACKTEST:
            raise ValueError("AlpacaBroker serves PAPER and LIVE only; use SimBroker for backtests")
        self.mode = mode

        if settings is None:
            from jevtrader.config import load_settings

            settings = load_settings()
        self._settings = settings

        if mode is TradingMode.LIVE:
            _assert_live_ok(settings, strategy_ids)

        self._fee_model = fee_model or default_fees()
        self._quote_provider = quote_provider
        self._instruments: dict[str, Instrument] = dict(instruments or {})
        self._lock = threading.RLock()
        self._open_orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._account: Optional[AccountState] = None
        self._liquidity_hint: dict[str, Liquidity] = {}
        self._trading_stream = trading_stream

        if client is not None:
            self._client = client
        else:
            from alpaca.trading.client import TradingClient

            key, secret = settings.keys_for(mode)
            self._client = TradingClient(key, secret, paper=(mode is not TradingMode.LIVE))

    # ------------------------------------------------------------------------- config

    def _instrument(self, symbol: str) -> Instrument:
        return self._instruments.get(symbol) or Instrument.infer(symbol)

    def set_quote_provider(self, fn: Optional[QuoteProvider]) -> None:
        self._quote_provider = fn

    # ------------------------------------------------------------------------- Broker contract

    def submit(self, order: Order) -> Order:
        instrument = self._instrument(order.symbol)
        order.created_ts = order.created_ts or pd.Timestamp.utcnow()

        if order.post_only and order.type is not OrderType.LIMIT:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "post_only is only meaningful for limit orders"
            self._emit_order(order)
            return order

        quote = self._quote_provider(order.symbol) if self._quote_provider else None
        would_cross = quote is not None and _crosses(order, quote)

        if order.post_only and would_cross:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "post_only: order would cross the book (emulated, Alpaca has no native post-only)"
            self._emit_order(order)
            return order

        intended_maker = order.type is OrderType.LIMIT and order.limit_price is not None and not would_cross

        try:
            request = build_order_request(order, instrument)
        except ValueError as exc:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(exc)
            self._emit_order(order)
            return order

        from alpaca.common.exceptions import APIError

        try:
            resp = self._client.submit_order(request)
        except APIError as exc:
            order.status = OrderStatus.REJECTED
            order.reject_reason = getattr(exc, "message", None) or str(exc)
            self._emit_order(order)
            return order

        with self._lock:
            self._liquidity_hint[order.client_order_id] = Liquidity.MAKER if intended_maker else Liquidity.TAKER
        _update_order_from_alpaca(order, resp)
        with self._lock:
            if order.status.is_open:
                self._open_orders[order.client_order_id] = order
        self._emit_order(order)
        return order

    def cancel(self, client_order_id: str) -> None:
        with self._lock:
            order = self._open_orders.get(client_order_id)
        if order is None or not order.broker_order_id:
            return
        self._client.cancel_order_by_id(order.broker_order_id)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._client.cancel_orders()
            with self._lock:
                self._open_orders.clear()
            return
        with self._lock:
            ids = [cid for cid, o in self._open_orders.items() if o.symbol == symbol]
        for cid in ids:
            self.cancel(cid)

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        with self._lock:
            orders = list(self._open_orders.values())
        if symbol is not None:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def positions(self) -> dict[str, Position]:
        with self._lock:
            return dict(self._positions)

    def account(self) -> AccountState:
        with self._lock:
            if self._account is not None:
                return self._account
            return AccountState(cash=0.0, equity=0.0, buying_power=0.0, positions=dict(self._positions))

    # ------------------------------------------------------------------------- reconciliation

    def reconcile(self) -> None:
        """Re-sync local open-order/position/account state from Alpaca's REST API. Call once on
        startup and then periodically (the LiveRunner does both)."""
        self._reconcile_orders()
        self._reconcile_positions()
        self._reconcile_account()

    def _reconcile_orders(self) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        try:
            resp = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        except TypeError:
            resp = self._client.get_orders()
        with self._lock:
            self._open_orders = {r.client_order_id: map_alpaca_order_to_core(r) for r in resp}

    def _reconcile_positions(self) -> None:
        resp = self._client.get_all_positions()
        with self._lock:
            self._positions = {r.symbol: map_alpaca_position_to_core(r) for r in resp}

    def _reconcile_account(self) -> None:
        resp = self._client.get_account()
        with self._lock:
            self._account = map_alpaca_account_to_core(resp, self._positions)

    # ------------------------------------------------------------------------- fills (trade updates stream)

    async def on_trade_update(self, update: Any) -> None:
        """Handler for `TradingStream.subscribe_trade_updates`. Also callable directly in tests
        with a duck-typed fake shaped like `alpaca.trading.models.TradeUpdate`."""
        resp_order = update.order
        cid = getattr(resp_order, "client_order_id", None)
        with self._lock:
            local = self._open_orders.get(cid)
        if local is None:
            local = map_alpaca_order_to_core(resp_order)
        else:
            _update_order_from_alpaca(local, resp_order)

        event = _val(getattr(update, "event", ""))
        if event in ("fill", "partial_fill"):
            qty = _f(getattr(update, "qty", None))
            price = _f(getattr(update, "price", None))
            if qty > 0:
                instrument = self._instrument(local.symbol)
                with self._lock:
                    liquidity = self._liquidity_hint.get(cid, Liquidity.TAKER)
                fee = self._fee_model.fee(instrument, local.side, qty, price, liquidity)
                ts_raw = getattr(update, "timestamp", None)
                ts = pd.Timestamp(ts_raw) if ts_raw is not None else pd.Timestamp.utcnow()
                fill = Fill(
                    client_order_id=cid,
                    symbol=local.symbol,
                    side=local.side,
                    qty=qty,
                    price=price,
                    fee=fee,
                    liquidity=liquidity,
                    ts=ts,
                    strategy_id=local.strategy_id,
                )
                with self._lock:
                    pos = self._positions.setdefault(local.symbol, Position(symbol=local.symbol))
                    pos.apply_fill(fill)
                self._emit_fill(fill)

        with self._lock:
            if local.status.is_open:
                self._open_orders[cid] = local
            else:
                self._open_orders.pop(cid, None)
                self._liquidity_hint.pop(cid, None)
        self._emit_order(local)

    async def run_stream(self) -> None:
        """Subscribe to trade updates and run the websocket connection until stopped. Runs the
        (blocking) alpaca-py `TradingStream.run()` in a worker thread so it shares this object's
        asyncio loop with everything else the LiveRunner does."""
        if self._trading_stream is None:
            raise RuntimeError("no trading_stream configured on this AlpacaBroker")
        self._trading_stream.subscribe_trade_updates(self.on_trade_update)
        await asyncio.to_thread(self._trading_stream.run)

    def stop_stream(self) -> None:
        if self._trading_stream is not None:
            self._trading_stream.stop()
