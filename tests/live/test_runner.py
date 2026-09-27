"""LiveRunner end-to-end: SimulatedStream + an in-memory fake broker + a trivial inline strategy
and RiskGate. Fakes are defined here per the task's offline-testing instructions."""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional

import pandas as pd
import pytest

from jevtrader.core.broker import Broker, TradingMode
from jevtrader.core.interfaces import RiskDecision
from jevtrader.core.strategy import Strategy, StrategyContext, StrategySpec
from jevtrader.core.types import AccountState, Bar, Fill, Liquidity, Order, OrderStatus, OrderType, Position, Side
from jevtrader.live.runner import LiveRunner, RunConfig
from jevtrader.live.streams import SimulatedStream


class FakeBroker(Broker):
    """Fills every order immediately at the bar close price it was submitted against."""

    def __init__(self) -> None:
        super().__init__()
        self.mode = TradingMode.PAPER
        self.positions_: Dict[str, Position] = {}
        self.submitted: List[Order] = []
        self.canceled_all_count = 0
        self.last_price: Dict[str, float] = {}
        self._cash = 100_000.0

    def submit(self, order: Order) -> Order:
        self.submitted.append(order)
        price = self.last_price.get(order.symbol, 100.0)
        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.avg_fill_price = price
        fill = Fill(order.client_order_id, order.symbol, order.side, order.qty, price, 0.1, Liquidity.TAKER,
                    pd.Timestamp.now("UTC"), order.strategy_id)
        pos = self.positions_.setdefault(order.symbol, Position(symbol=order.symbol))
        pos.apply_fill(fill)
        self._cash -= order.side.sign * order.qty * price
        self._emit_order(order)
        self._emit_fill(fill)
        return order

    def cancel(self, client_order_id: str) -> None:
        pass

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        self.canceled_all_count += 1

    def open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return []

    def positions(self) -> Dict[str, Position]:
        return dict(self.positions_)

    def account(self) -> AccountState:
        equity = self._cash + sum(p.market_value(self.last_price.get(s, p.avg_price)) for s, p in self.positions_.items())
        return AccountState(cash=self._cash, equity=equity, buying_power=equity)


class AlwaysApproveRisk:
    """Trivial RiskGate: approves everything unchanged, tracks calls."""

    def __init__(self) -> None:
        self.checks: List[Order] = []
        self.fills: List[Fill] = []
        self.marks: List[AccountState] = []
        self._halted = False

    def check(self, order, account, quote, now) -> RiskDecision:
        self.checks.append(order)
        return RiskDecision(approved=True, order=order)

    def on_fill(self, fill, account) -> None:
        self.fills.append(fill)

    def on_mark(self, account, now) -> None:
        self.marks.append(account)

    @property
    def halted(self) -> bool:
        return self._halted


class RejectAllRisk(AlwaysApproveRisk):
    def check(self, order, account, quote, now) -> RiskDecision:
        self.checks.append(order)
        return RiskDecision(approved=False, order=None, reason="nope")


class BuyOnceStrategy(Strategy):
    spec = StrategySpec(name="buy_once", description="test", asset_classes=("equity",), frequency="1min", style="momentum")

    def __init__(self, symbols):
        super().__init__(symbols)
        self.bars_seen: List[Bar] = []
        self.fills_seen: List[Fill] = []
        self._bought = False

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        self.bars_seen.append(bar)
        if not self._bought:
            self._bought = True
            ctx.submit(Order(symbol=bar.symbol, side=Side.BUY, qty=1, type=OrderType.MARKET))

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        self.fills_seen.append(fill)


def _bars(symbol, n=3):
    return [
        Bar(symbol=symbol, ts=pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=i),
            open=100 + i, high=100 + i, low=100 + i, close=100 + i, volume=10)
        for i in range(n)
    ]


def _runner(tmp_path, strategy, broker, risk, streams=None):
    queue = asyncio.Queue()
    bars = _bars("AAPL")
    stream = SimulatedStream(bars, queue, speed=0.0)
    for b in bars:
        broker.last_price[b.symbol] = b.close
    runner = LiveRunner(
        [strategy], broker, [stream], risk, runs_dir=tmp_path,
        config=RunConfig(heartbeat_seconds=1000, equity_snapshot_seconds=1000, reconcile_seconds=1000),
    )
    return runner


def test_runner_dispatches_bars_and_routes_orders_through_risk_and_broker(tmp_path):
    strategy = BuyOnceStrategy(["AAPL"])
    broker = FakeBroker()
    risk = AlwaysApproveRisk()
    runner = _runner(tmp_path, strategy, broker, risk)

    asyncio.run(runner.run(until_idle=True))

    assert len(strategy.bars_seen) == 3
    assert len(risk.checks) == 1
    assert len(broker.submitted) == 1
    assert len(strategy.fills_seen) == 1
    assert broker.positions()["AAPL"].qty == 1.0

    # journal was written
    lines = runner.journal.path.read_text().strip().splitlines()
    kinds = [json.loads(l)["kind"] for l in lines]
    assert "run_start" in kinds and "order" in kinds and "fill" in kinds and "run_stop" in kinds


def test_runner_blocks_orders_rejected_by_risk_gate(tmp_path):
    strategy = BuyOnceStrategy(["AAPL"])
    broker = FakeBroker()
    risk = RejectAllRisk()
    runner = _runner(tmp_path, strategy, broker, risk)

    asyncio.run(runner.run(until_idle=True))

    assert len(risk.checks) == 1
    assert broker.submitted == []
    assert strategy.fills_seen == []


def test_runner_cancels_open_orders_on_shutdown(tmp_path):
    strategy = BuyOnceStrategy(["AAPL"])
    broker = FakeBroker()
    risk = AlwaysApproveRisk()
    runner = _runner(tmp_path, strategy, broker, risk)
    asyncio.run(runner.run(until_idle=True))
    assert broker.canceled_all_count == 1


def test_runner_submit_manual_goes_through_risk_gate(tmp_path):
    strategy = BuyOnceStrategy(["AAPL"])
    broker = FakeBroker()
    risk = AlwaysApproveRisk()
    runner = _runner(tmp_path, strategy, broker, risk)
    result = runner.submit_manual({"symbol": "AAPL", "side": "buy", "qty": 2, "type": "market"})
    assert result["ok"] is True
    assert len(risk.checks) == 1
    assert broker.submitted[-1].strategy_id == "manual"


def test_runner_submit_manual_rejected_by_risk_gate(tmp_path):
    strategy = BuyOnceStrategy(["AAPL"])
    broker = FakeBroker()
    risk = RejectAllRisk()
    runner = _runner(tmp_path, strategy, broker, risk)
    result = runner.submit_manual({"symbol": "AAPL", "side": "buy", "qty": 2, "type": "market"})
    assert result["ok"] is False
    assert broker.submitted == []


def test_runner_submit_manual_invalid_payload(tmp_path):
    broker = FakeBroker()
    risk = AlwaysApproveRisk()
    runner = LiveRunner([], broker, [], risk, runs_dir=tmp_path)
    result = runner.submit_manual({"symbol": "AAPL"})  # missing side/qty
    assert result["ok"] is False
    runner.journal.close()
