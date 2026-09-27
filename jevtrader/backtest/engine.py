"""Event-driven backtest engine.

`Backtester` merges any number of already-time-sorted market data streams (bars, quotes, trades,
order books, across any number of symbols) into one global timeline, replays it through a
`SimBroker`, and dispatches events to one or more `Strategy` instances through a
`BacktestContext` that implements the same `StrategyContext` protocol the live runner uses -- so
a strategy written against `jevtrader.core.strategy.Strategy` runs unmodified here.

Processing order for every event, which is what makes the simulation lookahead-free:

1. `SimBroker.on_market_event(event)` updates the broker's view of the market and attempts to
   fill orders that were already resting *before* this event. Orders a strategy submits while
   reacting to this event are not visible to this step -- they only start resting afterward.
2. The event is appended to this symbol's bar/book history (for `ctx.bars()` / `ctx.last_book()`).
3. Every strategy subscribed to this symbol gets its `on_bar`/`on_quote`/`on_trade`/`on_book`
   handler called, via a `BacktestContext`. Any `ctx.submit(...)` here goes through the risk gate
   (if any) and then `SimBroker.submit`, becoming eligible to fill starting with the *next* event.
4. The risk gate's `on_mark` runs and the mark-to-market equity is recorded for this timestamp.

`on_day_end` fires once per UTC calendar-date boundary in the merged stream (a documented
simplification: real equity/crypto session boundaries differ, but a single UTC-day boundary is
adequate for daily bookkeeping hooks and keeps the engine asset-class-agnostic).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from jevtrader.core.broker import TradingMode
from jevtrader.core.fees import FeeModel, default_fees
from jevtrader.core.interfaces import RiskGate
from jevtrader.core.strategy import Strategy, StrategyContext
from jevtrader.core.types import (
    AccountState,
    AssetClass,
    Bar,
    Fill,
    Instrument,
    Order,
    OrderBook,
    OrderStatus,
    Position,
    Quote,
    Trade,
)

from jevtrader.backtest.sim_broker import SimBroker, SlippageModel

if TYPE_CHECKING:  # pragma: no cover
    from jevtrader.core.interfaces import JevAdvisorProtocol
    from jevtrader.jev.advisor import JevAdvisor

MarketEvent = Union[Bar, Quote, Trade, OrderBook]
DataInput = Union[Mapping[str, Iterable[MarketEvent]], Sequence[Iterable[MarketEvent]]]


@dataclass
class BacktestResult:
    """Everything a caller needs to evaluate a run: `metrics` is produced by
    `jevtrader.backtest.metrics.compute_metrics` on `equity_curve`/`fills`."""

    equity_curve: pd.Series
    fills: pd.DataFrame
    orders: pd.DataFrame
    per_strategy_pnl: dict[str, float]
    fee_total: float
    metrics: dict[str, float]
    positions: dict[str, Position] = field(default_factory=dict)
    final_account: Optional[AccountState] = None


class BacktestContext:
    """`StrategyContext` implementation backed by a running `Backtester`. One instance per
    strategy (so `ctx.log` / order attribution know which strategy they belong to)."""

    def __init__(self, engine: "Backtester", strategy: Strategy) -> None:
        self._engine = engine
        self._strategy = strategy

    @property
    def now(self) -> pd.Timestamp:
        return self._engine.now

    @property
    def jev(self) -> Optional["JevAdvisorProtocol"]:
        return self._engine.jev

    def instrument(self, symbol: str) -> Instrument:
        return self._engine.broker.instrument(symbol)

    def position(self, symbol: str) -> Position:
        return self._engine.broker.position_or_flat(symbol)

    def account(self) -> AccountState:
        return self._engine.broker.account()

    def last_quote(self, symbol: str) -> Optional[Quote]:
        return self._engine.broker.last_quote(symbol)

    def last_book(self, symbol: str) -> Optional[OrderBook]:
        return self._engine.last_book(symbol)

    def bars(self, symbol: str, n: int) -> pd.DataFrame:
        return self._engine.bars_df(symbol, n)

    def submit(self, order: Order) -> Optional[str]:
        order.strategy_id = order.strategy_id or self._strategy.id
        return self._engine.submit_order(order)

    def cancel(self, client_order_id: str) -> None:
        self._engine.broker.cancel(client_order_id)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        self._engine.broker.cancel_all(symbol)

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        return [o for o in self._engine.broker.open_orders(symbol) if o.strategy_id == self._strategy.id]

    def log(self, msg: str, **fields) -> None:
        self._engine.log(self._strategy.id, msg, fields)


class Backtester:
    """Event-driven backtester. See module docstring for exact event processing order."""

    def __init__(
        self,
        strategies: Sequence[Strategy],
        data: DataInput,
        fees: Optional[FeeModel] = None,
        risk: Optional[RiskGate] = None,
        jev: Optional["JevAdvisorProtocol"] = None,
        initial_cash: float = 100_000.0,
        latency_ms: float = 50.0,
        fill_model: Optional[SlippageModel] = None,
        instruments: Optional[Mapping[str, Instrument]] = None,
        allow_leverage: bool = False,
        leverage: float = 1.0,
        broker: Optional[SimBroker] = None,
        log_fn=None,
    ) -> None:
        self.strategies = list(strategies)
        self.data = data
        self.risk = risk
        self.jev = jev
        self.broker = broker or SimBroker(
            fee_model=fees or default_fees(),
            instruments=instruments,
            initial_cash=initial_cash,
            latency_ms=latency_ms,
            slippage=fill_model,
            allow_leverage=allow_leverage,
            leverage=leverage,
        )
        assert self.broker.mode is TradingMode.BACKTEST
        self.broker.on_fill(self._on_broker_fill)
        self.broker.on_order_update(self._on_broker_order)

        self._contexts: dict[str, BacktestContext] = {s.id: BacktestContext(self, s) for s in self.strategies}
        self._now: Optional[pd.Timestamp] = None
        self._bar_history: dict[str, list[Bar]] = {}
        self._last_book: dict[str, OrderBook] = {}
        self._equity_log: dict[pd.Timestamp, float] = {}
        self.all_fills: list[Fill] = []
        self.log_lines: list[dict] = []
        self._log_fn = log_fn
        self._ran = False

    # ------------------------------------------------------------------ context surface

    @property
    def now(self) -> pd.Timestamp:
        if self._now is None:
            raise RuntimeError("Backtester.now accessed before any event was processed")
        return self._now

    def last_book(self, symbol: str) -> Optional[OrderBook]:
        return self._last_book.get(symbol)

    def bars_df(self, symbol: str, n: int) -> pd.DataFrame:
        hist = self._bar_history.get(symbol, [])[-n:]
        if not hist:
            df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
            df.index = pd.DatetimeIndex([], name="ts", tz="UTC")
            return df
        return pd.DataFrame(
            {
                "open": [b.open for b in hist],
                "high": [b.high for b in hist],
                "low": [b.low for b in hist],
                "close": [b.close for b in hist],
                "volume": [b.volume for b in hist],
                "vwap": [b.vwap for b in hist],
            },
            index=pd.DatetimeIndex([b.ts for b in hist], name="ts"),
        )

    def submit_order(self, order: Order) -> Optional[str]:
        if self.risk is not None:
            account = self.broker.account()
            quote = self.broker.last_quote(order.symbol)
            decision = self.risk.check(order, account, quote, self.now)
            if not decision.approved or decision.order is None:
                order.status = OrderStatus.REJECTED
                order.reject_reason = decision.reason or "risk_rejected"
                order.created_ts = self.now
                return None
            order = decision.order
        result = self.broker.submit(order)
        if result.status is OrderStatus.REJECTED:
            return None
        return result.client_order_id

    def log(self, strategy_id: str, msg: str, fields: dict) -> None:
        entry = {"ts": self._now, "strategy_id": strategy_id, "msg": msg, **fields}
        self.log_lines.append(entry)
        if self._log_fn is not None:
            self._log_fn(entry)

    # ------------------------------------------------------------------ run loop

    @staticmethod
    def _normalize_streams(data: DataInput) -> list[Iterable[MarketEvent]]:
        if isinstance(data, Mapping):
            return [iter(v) for v in data.values()]
        return [iter(v) for v in data]

    def _update_history(self, event: MarketEvent) -> None:
        if isinstance(event, Bar):
            self._bar_history.setdefault(event.symbol, []).append(event)
        elif isinstance(event, OrderBook):
            self._last_book[event.symbol] = event

    def _dispatch(self, event: MarketEvent) -> None:
        for strat in self.strategies:
            if event.symbol not in strat.symbols:
                continue
            ctx = self._contexts[strat.id]
            if isinstance(event, Bar):
                strat.on_bar(event, ctx)
            elif isinstance(event, Quote):
                strat.on_quote(event, ctx)
            elif isinstance(event, Trade):
                strat.on_trade(event, ctx)
            elif isinstance(event, OrderBook):
                strat.on_book(event, ctx)

    def _on_broker_fill(self, fill: Fill) -> None:
        self.all_fills.append(fill)
        strat = self._strategy_by_id().get(fill.strategy_id)
        if strat is not None:
            strat.on_fill(fill, self._contexts[strat.id])
        if self.risk is not None:
            self.risk.on_fill(fill, self.broker.account())

    def _on_broker_order(self, order: Order) -> None:
        strat = self._strategy_by_id().get(order.strategy_id)
        if strat is not None:
            strat.on_order_update(order, self._contexts[strat.id])

    def _strategy_by_id(self) -> dict[str, Strategy]:
        return {s.id: s for s in self.strategies}

    def run(self) -> BacktestResult:
        if self._ran:
            raise RuntimeError("Backtester.run() already called; construct a new Backtester to run again")
        self._ran = True
        streams = self._normalize_streams(self.data)

        for strat in self.strategies:
            strat.on_start(self._contexts[strat.id])

        merged = heapq.merge(*streams, key=lambda e: e.ts) if streams else iter(())
        last_date = None
        for event in merged:
            d = event.ts.date()
            if last_date is not None and d != last_date:
                for strat in self.strategies:
                    strat.on_day_end(self._contexts[strat.id])
            last_date = d

            self._now = event.ts
            self.broker.on_market_event(event)
            self._update_history(event)
            self._dispatch(event)
            if self.risk is not None:
                self.risk.on_mark(self.broker.account(), self._now)
            self._equity_log[self._now] = self.broker.account().equity

        if last_date is not None:
            for strat in self.strategies:
                strat.on_day_end(self._contexts[strat.id])
        for strat in self.strategies:
            strat.on_stop(self._contexts[strat.id])

        return self._build_result()

    # ------------------------------------------------------------------ result assembly

    def _dominant_asset_class(self) -> AssetClass:
        symbols = {f.symbol for f in self.all_fills} or set(self._bar_history)
        classes = {self.broker.instrument(sym).asset_class for sym in symbols}
        if classes == {AssetClass.CRYPTO}:
            return AssetClass.CRYPTO
        return AssetClass.EQUITY

    def _per_strategy_pnl(self) -> dict[str, float]:
        by_strategy: dict[str, dict[str, Position]] = {}
        for f in self.all_fills:
            sid = f.strategy_id or "unknown"
            positions = by_strategy.setdefault(sid, {})
            pos = positions.setdefault(f.symbol, Position(f.symbol))
            pos.apply_fill(f)
        result: dict[str, float] = {}
        for sid, positions in by_strategy.items():
            total = 0.0
            for sym, pos in positions.items():
                price = self.broker.mark_price(sym)
                total += pos.realized_pnl + pos.unrealized_pnl(price if price is not None else pos.avg_price)
            result[sid] = total
        return result

    def _build_result(self) -> BacktestResult:
        equity_curve = pd.Series(self._equity_log, dtype=float).sort_index()
        equity_curve.index.name = "ts"
        equity_curve.name = "equity"

        fills_df = _fills_to_df(self.all_fills)
        orders_df = _orders_to_df(self.broker.all_orders())
        fee_total = float(fills_df["fee"].sum()) if not fills_df.empty else 0.0

        from jevtrader.backtest.metrics import compute_metrics

        metrics = compute_metrics(
            equity_curve,
            fills_df,
            asset_class=self._dominant_asset_class(),
            initial_cash=self.broker.initial_cash,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            fills=fills_df,
            orders=orders_df,
            per_strategy_pnl=self._per_strategy_pnl(),
            fee_total=fee_total,
            metrics=metrics,
            positions=self.broker.positions(),
            final_account=self.broker.account(),
        )


def _fills_to_df(fills: list[Fill]) -> pd.DataFrame:
    columns = ["ts", "symbol", "side", "qty", "price", "fee", "liquidity", "strategy_id", "client_order_id", "notional"]
    if not fills:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "ts": f.ts,
            "symbol": f.symbol,
            "side": f.side.value,
            "qty": f.qty,
            "price": f.price,
            "fee": f.fee,
            "liquidity": f.liquidity.value,
            "strategy_id": f.strategy_id,
            "client_order_id": f.client_order_id,
            "notional": f.notional,
        }
        for f in fills
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("ts").reset_index(drop=True)


def _orders_to_df(orders: list[Order]) -> pd.DataFrame:
    columns = [
        "client_order_id",
        "symbol",
        "side",
        "qty",
        "type",
        "limit_price",
        "stop_price",
        "tif",
        "status",
        "filled_qty",
        "avg_fill_price",
        "created_ts",
        "reject_reason",
        "strategy_id",
    ]
    if not orders:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "client_order_id": o.client_order_id,
            "symbol": o.symbol,
            "side": o.side.value,
            "qty": o.qty,
            "type": o.type.value,
            "limit_price": o.limit_price,
            "stop_price": o.stop_price,
            "tif": o.tif.value,
            "status": o.status.value,
            "filled_qty": o.filled_qty,
            "avg_fill_price": o.avg_fill_price,
            "created_ts": o.created_ts,
            "reject_reason": o.reject_reason,
            "strategy_id": o.strategy_id,
        }
        for o in orders
    ]
    return pd.DataFrame(rows, columns=columns)
