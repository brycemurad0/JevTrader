"""Demo mode: drives the dashboard from synthetic data and a tiny in-memory paper broker, so the
UI can be explored with no Alpaca keys and no other JevTrader subsystem (data/backtest/risk/jev)
running. Self-contained on purpose -- it does not import `jevtrader.data`, `jevtrader.risk` or
`jevtrader.live.runner`, so it works standalone regardless of those packages' state.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import threading
from typing import Dict, List, Optional

import pandas as pd

from jevtrader.core.broker import Broker, TradingMode
from jevtrader.core.types import AccountState, Fill, Liquidity, Order, OrderStatus, Position, Side
from jevtrader.dashboard.app import run_dashboard
from jevtrader.dashboard.state import DashboardState

logger = logging.getLogger(__name__)


class _DemoBroker(Broker):
    """Fills every order immediately at the current synthetic mid price. Good enough to drive
    the dashboard's manual ticket and kill switch in demo mode; not a real fill simulator."""

    def __init__(self, mid_price: Dict[str, float]) -> None:
        super().__init__()
        self.mode = TradingMode.PAPER
        self._mid = mid_price
        self._positions: Dict[str, Position] = {}
        self._cash = 100_000.0

    def submit(self, order: Order) -> Order:
        price = self._mid.get(order.symbol, order.limit_price or 100.0)
        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.avg_fill_price = price
        fee = abs(order.qty * price) * 0.0005
        fill = Fill(order.client_order_id, order.symbol, order.side, order.qty, price, fee, Liquidity.TAKER, pd.Timestamp.utcnow(), order.strategy_id)
        pos = self._positions.setdefault(order.symbol, Position(symbol=order.symbol))
        pos.apply_fill(fill)
        self._cash -= order.side.sign * order.qty * price + fee
        self._emit_order(order)
        self._emit_fill(fill)
        return order

    def cancel(self, client_order_id: str) -> None:
        return None

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        return None

    def open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return []

    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def account(self) -> AccountState:
        equity = self._cash + sum(p.market_value(self._mid.get(s, p.avg_price)) for s, p in self._positions.items())
        return AccountState(cash=self._cash, equity=equity, buying_power=equity)


def _default_mid(symbols: List[str]) -> Dict[str, float]:
    return {s: (30_000.0 if "/" in s else 100.0) for s in symbols}


def run_demo(symbols: Optional[List[str]] = None, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the dashboard fed by synthetic data and `_DemoBroker`. Blocks until interrupted."""
    symbols = symbols or ["AAPL", "BTC/USD"]
    state = DashboardState(mode="PAPER")
    mid = _default_mid(symbols)
    broker = _DemoBroker(mid)

    def _submit(payload: dict) -> dict:
        try:
            order = Order(
                symbol=str(payload["symbol"]),
                side=Side(payload["side"]),
                qty=float(payload["qty"]),
                strategy_id="manual",
            )
        except (KeyError, ValueError, TypeError) as exc:
            return {"ok": False, "reason": f"invalid order: {exc}"}
        broker.submit(order)
        return {"ok": True, "client_order_id": order.client_order_id}

    state.bind_actions(submit_order=_submit, cancel_order=broker.cancel, kill_switch=broker.close_all)
    for s in symbols:
        state.update_strategy(f"demo:{s}", status="running", pnl=0.0, position=0.0)

    stop_event = threading.Event()

    def _feed() -> None:
        rng = random.Random(7)
        while not stop_event.is_set():
            for s in symbols:
                mid[s] *= math.exp(rng.gauss(0, 0.0006))
                p = mid[s]
                spread = max(p * 0.0003, 0.01)
                bids = [{"price": round(p - spread / 2 - i * spread, 4), "size": round(rng.uniform(0.5, 5), 3)} for i in range(10)]
                asks = [{"price": round(p + spread / 2 + i * spread, 4), "size": round(rng.uniform(0.5, 5), 3)} for i in range(10)]
                state.update_book(s, bids, asks)
                state.push_trade(s, price=p, size=round(rng.uniform(0.01, 1.0), 3), side=rng.choice(["buy", "sell"]))
                pos = broker.positions().get(s)
                if pos is not None and abs(pos.qty) > 0:
                    state.update_position(s, qty=pos.qty, avg_price=pos.avg_price, unrealized_pnl=pos.unrealized_pnl(p))
            account = broker.account()
            state.update_account(cash=account.cash, equity=account.equity, buying_power=account.buying_power)
            state.update_risk(halted=False, utilization={"gross_exposure": 0.1, "daily_loss": 0.02})
            stop_event.wait(0.5)

    thread = threading.Thread(target=_feed, daemon=True)
    thread.start()
    try:
        run_dashboard(state, runner=None, host=host, port=port)
    finally:
        stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JevTrader dashboard in demo mode (no Alpaca keys needed).")
    parser.add_argument("--symbols", nargs="*", default=["AAPL", "BTC/USD"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_demo(args.symbols, args.host, args.port)


if __name__ == "__main__":
    main()
