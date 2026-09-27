"""Kill switch: LiveRunner.kill_switch() flattens via broker.close_all(), and a halted RiskGate
triggers the same flattening automatically on the next equity mark."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import pytest

from jevtrader.core.broker import Broker, TradingMode
from jevtrader.core.types import AccountState, Fill, Order, OrderStatus, Position, Side
from jevtrader.live.runner import LiveRunner, RunConfig


class FlattenTrackingBroker(Broker):
    """Positions to flatten + records whether close_all() (the kill switch) was invoked."""

    def __init__(self, positions: Optional[Dict[str, Position]] = None) -> None:
        super().__init__()
        self.mode = TradingMode.PAPER
        self._positions = positions or {}
        self.close_all_called = 0
        self.cancel_all_called = 0
        self.flatten_orders: List[Order] = []

    def submit(self, order: Order) -> Order:
        order.status = OrderStatus.FILLED
        if order.tag == "kill_switch":
            self.flatten_orders.append(order)
            self._positions.pop(order.symbol, None)
        return order

    def cancel(self, client_order_id: str) -> None:
        pass

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        self.cancel_all_called += 1

    def open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return []

    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def account(self) -> AccountState:
        return AccountState(cash=10_000.0, equity=10_000.0, buying_power=10_000.0)

    def close_all(self) -> None:
        self.close_all_called += 1
        super().close_all()  # exercises the real cancel_all + flatten-each-position default


class NeverApproveRisk:
    def __init__(self, halted: bool = False) -> None:
        self._halted = halted
        self.marks = 0

    def check(self, order, account, quote, now):
        from jevtrader.core.interfaces import RiskDecision
        return RiskDecision(approved=True, order=order)

    def on_fill(self, fill, account) -> None:
        pass

    def on_mark(self, account, now) -> None:
        self.marks += 1

    @property
    def halted(self) -> bool:
        return self._halted


def test_manual_kill_switch_flattens_via_broker(tmp_path):
    broker = FlattenTrackingBroker({"AAPL": Position(symbol="AAPL", qty=10, avg_price=100.0)})
    risk = NeverApproveRisk(halted=False)
    runner = LiveRunner([], broker, [], risk, runs_dir=tmp_path)

    runner.kill_switch(reason="test")

    assert broker.close_all_called == 1
    assert broker.cancel_all_called == 1
    assert len(broker.flatten_orders) == 1
    assert broker.flatten_orders[0].symbol == "AAPL"
    assert broker.flatten_orders[0].side is Side.SELL  # was long, so flatten sells
    assert broker.positions() == {}
    runner.journal.close()


def test_risk_halt_triggers_kill_switch_on_equity_mark(tmp_path):
    broker = FlattenTrackingBroker({"AAPL": Position(symbol="AAPL", qty=5, avg_price=50.0)})
    risk = NeverApproveRisk(halted=True)
    runner = LiveRunner([], broker, [], risk, runs_dir=tmp_path, config=RunConfig(kill_on_halt=True))

    runner._on_equity_mark(broker.account())

    assert risk.marks == 1
    assert broker.close_all_called == 1
    assert broker.positions() == {}
    runner.journal.close()


def test_risk_halt_does_not_kill_when_disabled(tmp_path):
    broker = FlattenTrackingBroker({"AAPL": Position(symbol="AAPL", qty=5, avg_price=50.0)})
    risk = NeverApproveRisk(halted=True)
    runner = LiveRunner([], broker, [], risk, runs_dir=tmp_path, config=RunConfig(kill_on_halt=False))

    runner._on_equity_mark(broker.account())

    assert broker.close_all_called == 0
    runner.journal.close()


def test_no_kill_when_not_halted(tmp_path):
    broker = FlattenTrackingBroker({"AAPL": Position(symbol="AAPL", qty=5, avg_price=50.0)})
    risk = NeverApproveRisk(halted=False)
    runner = LiveRunner([], broker, [], risk, runs_dir=tmp_path)

    runner._on_equity_mark(broker.account())

    assert broker.close_all_called == 0
    runner.journal.close()
