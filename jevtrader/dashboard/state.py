"""`DashboardState`: the shared, thread-safe, plain-data object the dashboard backend reads and
that a `LiveRunner` (or `jevtrader.dashboard.demo`) writes to. Deliberately has no FastAPI, no
asyncio and no knowledge of `Broker`/`Strategy`/alpaca-py -- it is just a snapshot the frontend can
render and a couple of action hooks the frontend can call, wired up by whoever owns the broker.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional


def _now_ms() -> float:
    return time.time() * 1000.0


@dataclass
class DashboardState:
    mode: str = "PAPER"
    max_book_depth: int = 25
    max_trades: int = 200
    max_jev_feed: int = 100

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self.account: Dict[str, Any] = {
            "cash": 0.0,
            "equity": 0.0,
            "buying_power": 0.0,
            "peak_equity": 0.0,
            "drawdown": 0.0,
        }
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.books: Dict[str, Dict[str, Any]] = {}
        self.trades: Dict[str, Deque[Dict[str, Any]]] = {}
        self.jev_feed: Deque[Dict[str, Any]] = deque(maxlen=self.max_jev_feed)
        self.risk: Dict[str, Any] = {"halted": False, "utilization": {}}
        self.kill_switch_engaged: bool = False
        self._on_submit_order: Optional[Callable[[dict], dict]] = None
        self._on_cancel_order: Optional[Callable[[str], Any]] = None
        self._on_kill: Optional[Callable[[], Any]] = None

    # ------------------------------------------------------------------ wiring

    def bind_actions(
        self,
        *,
        submit_order: Optional[Callable[[dict], dict]] = None,
        cancel_order: Optional[Callable[[str], Any]] = None,
        kill_switch: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Wire the dashboard's manual-ticket / cancel / kill-switch buttons to real handlers
        (normally `LiveRunner.submit_manual`, `broker.cancel`, `LiveRunner.kill_switch`). Leaving
        a hook unset means that action is unavailable (e.g. demo mode has no kill switch worth
        wiring, or wires it to a no-op)."""
        with self._lock:
            if submit_order is not None:
                self._on_submit_order = submit_order
            if cancel_order is not None:
                self._on_cancel_order = cancel_order
            if kill_switch is not None:
                self._on_kill = kill_switch

    # ------------------------------------------------------------------ writers

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self.mode = mode

    def update_account(self, *, cash: float, equity: float, buying_power: float = 0.0) -> None:
        with self._lock:
            peak = max(self.account.get("peak_equity", 0.0), equity)
            drawdown = 0.0 if peak <= 0 else max(0.0, (peak - equity) / peak)
            self.account.update(cash=cash, equity=equity, buying_power=buying_power, peak_equity=peak, drawdown=drawdown)

    def update_strategy(self, strategy_id: str, **fields: Any) -> None:
        with self._lock:
            self.strategies.setdefault(strategy_id, {"strategy_id": strategy_id}).update(fields)

    def update_position(self, symbol: str, **fields: Any) -> None:
        with self._lock:
            self.positions.setdefault(symbol, {"symbol": symbol}).update(fields)

    def upsert_order(self, order: Dict[str, Any]) -> None:
        with self._lock:
            cid = order["client_order_id"]
            if order.get("status") in ("filled", "canceled", "rejected", "expired"):
                self.open_orders.pop(cid, None)
            else:
                self.open_orders[cid] = dict(order)

    def update_book(self, symbol: str, bids: List[Dict[str, float]], asks: List[Dict[str, float]], ts: Optional[float] = None) -> None:
        with self._lock:
            self.books[symbol] = {
                "symbol": symbol,
                "ts": ts if ts is not None else _now_ms(),
                "bids": list(bids[: self.max_book_depth]),
                "asks": list(asks[: self.max_book_depth]),
            }

    def push_trade(self, symbol: str, price: float, size: float, side: Optional[str] = None, ts: Optional[float] = None) -> None:
        with self._lock:
            dq = self.trades.setdefault(symbol, deque(maxlen=self.max_trades))
            dq.appendleft({"symbol": symbol, "price": price, "size": size, "side": side, "ts": ts if ts is not None else _now_ms()})

    def push_jev_decision(self, record: Dict[str, Any]) -> None:
        with self._lock:
            rec = dict(record)
            rec.setdefault("ts", _now_ms())
            self.jev_feed.appendleft(rec)

    def update_risk(self, **fields: Any) -> None:
        with self._lock:
            self.risk.update(fields)

    # ------------------------------------------------------------------ actions (dashboard -> broker)

    def submit_manual_order(self, order: dict) -> dict:
        with self._lock:
            handler = self._on_submit_order
        if handler is None:
            return {"ok": False, "reason": "no broker wired to this dashboard (demo mode?)"}
        return handler(order)

    def cancel_order(self, client_order_id: str) -> dict:
        with self._lock:
            handler = self._on_cancel_order
        if handler is None:
            return {"ok": False, "reason": "no broker wired to this dashboard (demo mode?)"}
        handler(client_order_id)
        return {"ok": True}

    def kill(self) -> dict:
        with self._lock:
            handler = self._on_kill
            self.kill_switch_engaged = True
        if handler is None:
            return {"ok": False, "reason": "no kill switch wired to this dashboard (demo mode?)"}
        handler()
        return {"ok": True}

    # ------------------------------------------------------------------ reader

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "account": dict(self.account),
                "positions": {k: dict(v) for k, v in self.positions.items()},
                "strategies": {k: dict(v) for k, v in self.strategies.items()},
                "open_orders": {k: dict(v) for k, v in self.open_orders.items()},
                "books": {k: dict(v) for k, v in self.books.items()},
                "trades": {k: list(v) for k, v in self.trades.items()},
                "jev_feed": list(self.jev_feed),
                "risk": dict(self.risk),
                "kill_switch_engaged": self.kill_switch_engaged,
            }
