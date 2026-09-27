"""Broker contract, implemented by SimBroker (backtest), AlpacaBroker(paper) and AlpacaBroker(live)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional

from jevtrader.core.types import AccountState, Fill, Order, Position


class TradingMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


FillCallback = Callable[[Fill], None]
OrderCallback = Callable[[Order], None]


class Broker(ABC):
    mode: TradingMode

    def __init__(self) -> None:
        self._fill_cbs: list[FillCallback] = []
        self._order_cbs: list[OrderCallback] = []

    def on_fill(self, cb: FillCallback) -> None:
        self._fill_cbs.append(cb)

    def on_order_update(self, cb: OrderCallback) -> None:
        self._order_cbs.append(cb)

    def _emit_fill(self, fill: Fill) -> None:
        for cb in self._fill_cbs:
            cb(fill)

    def _emit_order(self, order: Order) -> None:
        for cb in self._order_cbs:
            cb(order)

    @abstractmethod
    def submit(self, order: Order) -> Order:
        """Send order; returns the order with broker_order_id/status set (may be REJECTED)."""

    @abstractmethod
    def cancel(self, client_order_id: str) -> None: ...

    @abstractmethod
    def cancel_all(self, symbol: Optional[str] = None) -> None: ...

    @abstractmethod
    def open_orders(self, symbol: Optional[str] = None) -> list[Order]: ...

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def account(self) -> AccountState: ...

    def close_all(self) -> None:
        """Flatten everything (kill switch). Default: cancel all then market-out each position."""
        from jevtrader.core.types import OrderType, Side

        self.cancel_all()
        for sym, pos in self.positions().items():
            if abs(pos.qty) > 0:
                side = Side.SELL if pos.qty > 0 else Side.BUY
                self.submit(Order(sym, side, abs(pos.qty), OrderType.MARKET, reduce_only=True, tag="kill_switch"))
