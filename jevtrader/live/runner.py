"""LiveRunner: the async event loop that runs Strategy objects against a live/paper Broker,
market data streams and a RiskGate. The same Strategy subclasses run here as in the backtester;
`LiveContext` is the `StrategyContext` implementation backing them.

Responsibilities (see docs/ARCHITECTURE.md):
* Consume normalized market events from each stream and dispatch them to the strategies
  subscribed to that symbol, maintaining rolling bar history / last quote / last book.
* Route `ctx.submit(order)` through `risk.check(...)` before it ever reaches the broker.
* Route fills and order updates back to `strategy.on_fill` / `on_order_update` and to
  `risk.on_fill`.
* Mark-to-market periodically via `risk.on_mark`; if the risk gate trips (`risk.halted`), flatten
  everything (`broker.close_all()`) when `RunConfig.kill_on_halt` is set.
* Heartbeat + stale-data watchdog: no data for `stale_data_seconds` pauses new order submission
  until fresh data arrives.
* Periodically reconcile the broker's local state (if it exposes `.reconcile()`).
* Write a JSONL event journal (orders, fills, risk decisions, jev decisions, equity marks) under
  `runs_dir/<run_id>/events.jsonl` -- this is the paper-trading forward-test record the
  promotion gate (`jevtrader.live.promotion`) reads.
* Optionally push everything to a `jevtrader.dashboard.state.DashboardState` -- decoupled: the
  runner works identically with or without a dashboard attached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

import pandas as pd

from jevtrader.core.interfaces import JevView, RiskGate
from jevtrader.core.strategy import Strategy, StrategyContext
from jevtrader.core.types import (
    AccountState,
    Bar,
    Fill,
    Instrument,
    MarketEvent,
    Order,
    OrderBook,
    OrderType,
    Position,
    Quote,
    Side,
    Trade,
)

if TYPE_CHECKING:  # pragma: no cover
    from jevtrader.core.broker import Broker
    from jevtrader.jev.advisor import JevAdvisor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------- config

@dataclass
class RunConfig:
    run_id: str = field(default_factory=lambda: pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%fZ"))
    stale_data_seconds: float = 30.0
    heartbeat_seconds: float = 5.0
    equity_snapshot_seconds: float = 60.0
    reconcile_seconds: float = 60.0
    cancel_open_orders_on_stop: bool = True
    kill_on_halt: bool = True


# --------------------------------------------------------------------------------- journal

class Journal:
    """Append-only JSONL event log under `run_dir/events.jsonl`: orders, fills, risk decisions,
    jev decisions and periodic equity snapshots -- the paper-trading forward-test record used by
    the promotion gate."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", buffering=1)

    def write(self, kind: str, **fields: Any) -> None:
        record = {"ts": pd.Timestamp.utcnow().isoformat(), "kind": kind, **fields}
        line = json.dumps(record, default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass


def _order_dict(order: Order) -> dict:
    return {
        "client_order_id": order.client_order_id,
        "broker_order_id": order.broker_order_id,
        "symbol": order.symbol,
        "side": order.side.value,
        "qty": order.qty,
        "type": order.type.value,
        "limit_price": order.limit_price,
        "stop_price": order.stop_price,
        "tif": order.tif.value,
        "post_only": order.post_only,
        "status": order.status.value,
        "filled_qty": order.filled_qty,
        "avg_fill_price": order.avg_fill_price,
        "strategy_id": order.strategy_id,
        "reject_reason": order.reject_reason,
    }


def _fill_dict(fill: Fill) -> dict:
    return {
        "client_order_id": fill.client_order_id,
        "symbol": fill.symbol,
        "side": fill.side.value,
        "qty": fill.qty,
        "price": fill.price,
        "fee": fill.fee,
        "liquidity": fill.liquidity.value,
        "strategy_id": fill.strategy_id,
        "ts": str(fill.ts),
    }


# --------------------------------------------------------------------------------- bar aggregation

class BarBuilder:
    """Aggregates trades (or, failing that, quote mid-prices) into fixed-width bars for symbols
    that only stream ticks -- e.g. a crypto book/trade feed with no native minute bars. Bars that
    already arrive from a native bar feed bypass this and are emitted as-is."""

    def __init__(self, width: pd.Timedelta = pd.Timedelta(seconds=60)) -> None:
        self.width = width
        self._open: dict[str, dict[str, Any]] = {}

    def _bucket(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.floor(self.width)

    def on_trade(self, trade: Trade) -> list[Bar]:
        return self._on_tick(trade.symbol, trade.ts, trade.price, trade.size)

    def on_quote(self, quote: Quote) -> list[Bar]:
        return self._on_tick(quote.symbol, quote.ts, quote.mid, 0.0)

    def _on_tick(self, symbol: str, ts: pd.Timestamp, price: float, size: float) -> list[Bar]:
        bucket = self._bucket(ts)
        state = self._open.get(symbol)
        closed: list[Bar] = []
        if state is not None and state["bucket"] != bucket:
            closed.append(self._to_bar(symbol, state))
            state = None
        if state is None:
            state = {"bucket": bucket, "open": price, "high": price, "low": price, "close": price, "volume": 0.0}
            self._open[symbol] = state
        state["high"] = max(state["high"], price)
        state["low"] = min(state["low"], price)
        state["close"] = price
        state["volume"] += size
        return closed

    def _to_bar(self, symbol: str, state: dict) -> Bar:
        return Bar(
            symbol=symbol,
            ts=state["bucket"],
            open=state["open"],
            high=state["high"],
            low=state["low"],
            close=state["close"],
            volume=state["volume"],
        )


# --------------------------------------------------------------------------------- null dashboard

class _NullDashboard:
    """No-op stand-in for a `DashboardState` so `LiveRunner` never has to branch on whether one
    was attached."""

    def __getattr__(self, name: str) -> Callable[..., None]:
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop


# --------------------------------------------------------------------------------- jev logging proxy

class _JevProxy:
    """Wraps a `JevAdvisorProtocol` implementation so every decision it returns is also logged to
    the run journal and (if attached) pushed to the dashboard's Jev feed -- without changing the
    advisor's contract, so strategies calling `ctx.jev.direction(...)` etc. see no difference."""

    def __init__(self, advisor: Any, on_view: Callable[[str, str, Optional[JevView]], None]) -> None:
        self._advisor = advisor
        self._on_view = on_view

    def direction(self, symbol: str, features: Mapping[str, Any], horizon: str) -> Optional[JevView]:
        view = self._advisor.direction(symbol, features, horizon)
        self._on_view("direction", symbol, view)
        return view

    def regime(self, symbol: str, features: Mapping[str, Any]) -> Optional[JevView]:
        view = self._advisor.regime(symbol, features)
        self._on_view("regime", symbol, view)
        return view

    def ask(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        view = self._advisor.ask(state, questions)
        self._on_view("ask", str(state.get("symbol", "")), view)
        return view


# --------------------------------------------------------------------------------- context

class LiveContext:
    """`StrategyContext` implementation backing the live/paper runner."""

    def __init__(self, runner: "LiveRunner", strategy_id: str) -> None:
        self._runner = runner
        self.strategy_id = strategy_id

    @property
    def now(self) -> pd.Timestamp:
        return self._runner.now

    @property
    def jev(self) -> Optional["JevAdvisor"]:
        return self._runner.jev

    def instrument(self, symbol: str) -> Instrument:
        return self._runner.instrument(symbol)

    def position(self, symbol: str) -> Position:
        return self._runner.broker.positions().get(symbol, Position(symbol=symbol))

    def account(self) -> AccountState:
        return self._runner.broker.account()

    def last_quote(self, symbol: str) -> Optional[Quote]:
        return self._runner.last_quote.get(symbol)

    def last_book(self, symbol: str) -> Optional[OrderBook]:
        return self._runner.last_book.get(symbol)

    def bars(self, symbol: str, n: int) -> pd.DataFrame:
        return self._runner.bar_history(symbol, n)

    def submit(self, order: Order) -> Optional[str]:
        order.strategy_id = order.strategy_id or self.strategy_id
        return self._runner.submit(order)

    def cancel(self, client_order_id: str) -> None:
        self._runner.broker.cancel(client_order_id)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        self._runner.broker.cancel_all(symbol)

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        return self._runner.broker.open_orders(symbol)

    def log(self, msg: str, **fields: Any) -> None:
        self._runner.journal.write("log", strategy_id=self.strategy_id, msg=msg, **fields)


# --------------------------------------------------------------------------------- runner

class LiveRunner:
    """Drives `strategies` against `broker` and `streams` under `risk`'s supervision.

    `streams` is a sequence of adapters (see `jevtrader.live.streams`), each with `.queue`
    (an `asyncio.Queue` of core `MarketEvent`s), `async .run()` and `.stop()`.
    """

    def __init__(
        self,
        strategies: Sequence[Strategy],
        broker: "Broker",
        streams: Sequence[Any],
        risk: RiskGate,
        jev: Optional[Any] = None,
        bar_builder: Optional[BarBuilder] = None,
        config: Optional[RunConfig] = None,
        runs_dir: Optional[Any] = None,
        instruments: Optional[Mapping[str, Instrument]] = None,
        dashboard_state: Optional[Any] = None,
    ) -> None:
        self.strategies = list(strategies)
        self.broker = broker
        self.streams = list(streams)
        self.risk = risk
        self.jev = _JevProxy(jev, self._on_jev_view) if jev is not None else None
        self.bar_builder = bar_builder or BarBuilder()
        self.config = config or RunConfig()
        self._instruments: dict[str, Instrument] = dict(instruments or {})
        self.dashboard = dashboard_state if dashboard_state is not None else _NullDashboard()

        self._contexts: dict[str, LiveContext] = {s.id: LiveContext(self, s.id) for s in self.strategies}
        self._bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self.last_quote: dict[str, Quote] = {}
        self.last_book: dict[str, OrderBook] = {}
        self.now: pd.Timestamp = pd.Timestamp.utcnow()
        self._last_data_ts: Optional[pd.Timestamp] = None
        self._paused = False
        self._stop = asyncio.Event()

        settings_runs_dir = getattr(getattr(broker, "_settings", None), "runs_dir", None)
        base = Path(runs_dir) if runs_dir is not None else Path(settings_runs_dir or "runs")
        self.run_dir = base / self.config.run_id
        self.journal = Journal(self.run_dir)

        mode = getattr(broker, "mode", None)
        self.dashboard.set_mode(getattr(mode, "value", str(mode)).upper() if mode is not None else "PAPER")
        for s in self.strategies:
            self.dashboard.update_strategy(s.id, status="starting", pnl=0.0, position=0.0)

        broker.on_fill(self._on_fill)
        broker.on_order_update(self._on_order_update)

    # ------------------------------------------------------------------------- lookups

    def instrument(self, symbol: str) -> Instrument:
        return self._instruments.get(symbol) or Instrument.infer(symbol)

    def bar_history(self, symbol: str, n: int) -> pd.DataFrame:
        bars = list(self._bars.get(symbol, ()))[-n:]
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        idx = pd.DatetimeIndex([b.ts for b in bars])
        return pd.DataFrame(
            {
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            },
            index=idx,
        )

    def _strategies_for(self, symbol: str) -> list[Strategy]:
        return [s for s in self.strategies if symbol in s.symbols]

    def _strategy_for_id(self, strategy_id: str) -> Optional[Strategy]:
        for s in self.strategies:
            if s.id == strategy_id:
                return s
        return None

    # ------------------------------------------------------------------------- order flow

    def submit(self, order: Order) -> Optional[str]:
        if self._paused:
            self.journal.write(
                "order_blocked", reason="stale_data_watchdog", client_order_id=order.client_order_id
            )
            return None
        quote = self.last_quote.get(order.symbol)
        decision = self.risk.check(order, self.broker.account(), quote, self.now)
        self.journal.write(
            "risk_decision",
            approved=decision.approved,
            reason=decision.reason,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
        )
        if not decision.approved or decision.order is None:
            return None
        result = self.broker.submit(decision.order)
        self.journal.write("order", **_order_dict(result))
        self.dashboard.upsert_order(_order_dict(result))
        return None if result.reject_reason else result.client_order_id

    def submit_manual(self, payload: Mapping[str, Any]) -> dict:
        """Build a core `Order` from a dashboard manual-ticket payload and route it through
        `submit()` -- i.e. through the same `RiskGate` every strategy order goes through."""
        try:
            order = Order(
                symbol=str(payload["symbol"]),
                side=Side(payload["side"]),
                qty=float(payload["qty"]),
                type=OrderType(payload.get("type", "market")),
                limit_price=float(payload["limit_price"]) if payload.get("limit_price") is not None else None,
                stop_price=float(payload["stop_price"]) if payload.get("stop_price") is not None else None,
                post_only=bool(payload.get("post_only", False)),
                strategy_id="manual",
            )
        except (KeyError, ValueError, TypeError) as exc:
            return {"ok": False, "reason": f"invalid order payload: {exc}"}
        cid = self.submit(order)
        if cid is None:
            return {"ok": False, "reason": "rejected by risk gate", "client_order_id": order.client_order_id}
        return {"ok": True, "client_order_id": cid}

    def kill_switch(self, reason: str = "manual") -> None:
        self.journal.write("kill_switch", reason=reason)
        self.dashboard.update_risk(halted=True)
        self.broker.close_all()

    # ------------------------------------------------------------------------- callbacks (broker)

    def _on_fill(self, fill: Fill) -> None:
        self.journal.write("fill", **_fill_dict(fill))
        try:
            self.risk.on_fill(fill, self.broker.account())
        except Exception:
            logger.exception("risk.on_fill raised")
        pos = self.broker.positions().get(fill.symbol)
        if pos is not None:
            self.dashboard.update_position(fill.symbol, qty=pos.qty, avg_price=pos.avg_price, realized_pnl=pos.realized_pnl)
        if fill.strategy_id:
            self.dashboard.update_strategy(fill.strategy_id, last_fill=_fill_dict(fill))
        strategy = self._strategy_for_id(fill.strategy_id)
        if strategy is not None:
            strategy.on_fill(fill, self._contexts[strategy.id])

    def _on_order_update(self, order: Order) -> None:
        self.journal.write("order_update", **_order_dict(order))
        self.dashboard.upsert_order(_order_dict(order))
        strategy = self._strategy_for_id(order.strategy_id)
        if strategy is not None:
            strategy.on_order_update(order, self._contexts[strategy.id])

    def _on_jev_view(self, question: str, symbol: str, view: Optional[JevView]) -> None:
        if view is None:
            return
        record = {
            "question": question,
            "symbol": symbol,
            "top": dict(view.top),
            "latency_ms": view.latency_ms,
            "late": view.late,
            "model": view.model,
        }
        self.journal.write("jev_decision", **record)
        self.dashboard.push_jev_decision(record)

    # ------------------------------------------------------------------------- market data dispatch

    async def _dispatch(self, event: MarketEvent) -> None:
        self.now = event.ts
        self._last_data_ts = event.ts
        if self._paused:
            self._paused = False
            self.journal.write("watchdog_resume")
        symbol = event.symbol

        if isinstance(event, Quote):
            self.last_quote[symbol] = event
            for s in self._strategies_for(symbol):
                s.on_quote(event, self._contexts[s.id])
            if symbol not in self.last_book:
                self.dashboard.update_book(
                    symbol,
                    [{"price": event.bid, "size": event.bid_size}],
                    [{"price": event.ask, "size": event.ask_size}],
                )
            for bar in self.bar_builder.on_quote(event):
                await self._emit_bar(bar)
        elif isinstance(event, Trade):
            for s in self._strategies_for(symbol):
                s.on_trade(event, self._contexts[s.id])
            self.dashboard.push_trade(symbol, price=event.price, size=event.size, side=getattr(event.aggressor, "value", None))
            for bar in self.bar_builder.on_trade(event):
                await self._emit_bar(bar)
        elif isinstance(event, OrderBook):
            self.last_book[symbol] = event
            for s in self._strategies_for(symbol):
                s.on_book(event, self._contexts[s.id])
            self.dashboard.update_book(
                symbol,
                [{"price": lvl.price, "size": lvl.size} for lvl in event.bids],
                [{"price": lvl.price, "size": lvl.size} for lvl in event.asks],
            )
        elif isinstance(event, Bar):
            await self._emit_bar(event)

    async def _emit_bar(self, bar: Bar) -> None:
        self._bars[bar.symbol].append(bar)
        for s in self._strategies_for(bar.symbol):
            s.on_bar(bar, self._contexts[s.id])

    # ------------------------------------------------------------------------- background loops

    def _is_stale(self) -> bool:
        if self._last_data_ts is None:
            return False
        age = (pd.Timestamp.utcnow() - self._last_data_ts).total_seconds()
        return age > self.config.stale_data_seconds

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.config.heartbeat_seconds)
            if self._is_stale() and not self._paused:
                self._paused = True
                self.journal.write("watchdog_pause", reason="stale_data")
            self.journal.write("heartbeat", paused=self._paused)

    def _on_equity_mark(self, account: AccountState) -> None:
        """One iteration of the equity-mark/kill-switch check; exposed separately from the loop
        so it can be exercised deterministically in tests."""
        try:
            self.risk.on_mark(account, self.now)
        except Exception:
            logger.exception("risk.on_mark raised")
        self.dashboard.update_account(cash=account.cash, equity=account.equity, buying_power=account.buying_power)
        self.dashboard.update_risk(halted=self.risk.halted)
        self.journal.write("equity", cash=account.cash, equity=account.equity, ts=str(self.now))
        if self.risk.halted and self.config.kill_on_halt:
            self.journal.write("kill_switch", reason="risk_halted")
            try:
                self.broker.close_all()
            except Exception:
                logger.exception("close_all failed on kill switch")

    async def _equity_snapshot_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.config.equity_snapshot_seconds)
            try:
                account = self.broker.account()
            except Exception:
                logger.exception("broker.account() failed for equity snapshot")
                continue
            self._on_equity_mark(account)

    async def _reconcile_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.config.reconcile_seconds)
            self._reconcile_safely()

    def _reconcile_safely(self) -> None:
        reconcile = getattr(self.broker, "reconcile", None)
        if callable(reconcile):
            try:
                reconcile()
            except Exception:
                logger.exception("broker.reconcile() failed")

    async def _consume(self, stream: Any) -> None:
        while not self._stop.is_set():
            try:
                event = await asyncio.wait_for(stream.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self._dispatch(event)

    async def _run_broker_stream(self) -> None:
        run_stream = getattr(self.broker, "run_stream", None)
        if not callable(run_stream):
            return
        try:
            await run_stream()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("broker trade-update stream failed")

    async def _wait_idle(self, poll: float = 0.02) -> None:
        while True:
            await asyncio.sleep(poll)
            done = all(getattr(s, "done", None) is None or s.done.is_set() for s in self.streams)
            empty = all(s.queue.empty() for s in self.streams)
            if done and empty:
                await asyncio.sleep(poll)
                return

    # ------------------------------------------------------------------------- lifecycle

    def stop(self) -> None:
        self._stop.set()

    async def run(self, until_idle: bool = False) -> None:
        self.journal.write("run_start", run_id=self.config.run_id, mode=str(getattr(self.broker, "mode", "")))
        self._reconcile_safely()
        for s in self.strategies:
            s.on_start(self._contexts[s.id])
            self.dashboard.update_strategy(s.id, status="running")

        tasks: list[asyncio.Task] = []
        for stream in self.streams:
            tasks.append(asyncio.create_task(stream.run()))
            tasks.append(asyncio.create_task(self._consume(stream)))
        tasks.append(asyncio.create_task(self._heartbeat_loop()))
        tasks.append(asyncio.create_task(self._equity_snapshot_loop()))
        tasks.append(asyncio.create_task(self._reconcile_loop()))
        tasks.append(asyncio.create_task(self._run_broker_stream()))

        try:
            if until_idle:
                await self._wait_idle()
                self._stop.set()
            else:
                await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._shutdown()

    async def _shutdown(self) -> None:
        for s in self.strategies:
            try:
                s.on_stop(self._contexts[s.id])
                self.dashboard.update_strategy(s.id, status="stopped")
            except Exception:
                logger.exception("on_stop failed for %s", s.id)
        if self.config.cancel_open_orders_on_stop:
            try:
                self.broker.cancel_all()
            except Exception:
                logger.exception("cancel_all failed on shutdown")
        self.journal.write("run_stop", run_id=self.config.run_id)
        self.journal.close()
