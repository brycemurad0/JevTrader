"""Fakes for AlpacaBroker tests: duck-typed stand-ins for alpaca-py's model/client shapes.

AlpacaBroker only ever accesses attributes on these objects (never isinstance-checks against
alpaca-py classes), so a `SimpleNamespace` with the right attribute names behaves identically to
a real `alpaca.trading.models.Order`/`Position`/`TradeAccount`/`TradeUpdate` for its purposes.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd

from jevtrader.config import LIVE_CONFIRM_PHRASE, Settings


def make_settings(tmp_path: Path, *, live_confirm: str = "") -> Settings:
    """A `Settings` instance rooted under a pytest tmp_path, so promotion records and run
    journals never touch the real repo's `state/`/`runs/` directories."""
    return Settings(
        alpaca_key="test-key",
        alpaca_secret="test-secret",
        alpaca_paper=True,
        alpaca_live_key="",
        alpaca_live_secret="",
        alpaca_data_feed="iex",
        typesafe_api_key="",
        jev_model="jev-latest",
        jev_latency_budget_ms=400,
        live_confirm=live_confirm,
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
    )


def make_live_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path, live_confirm=LIVE_CONFIRM_PHRASE)


def alpaca_order(**overrides: Any) -> SimpleNamespace:
    defaults = dict(
        id=str(uuid.uuid4()),
        client_order_id="jt-1",
        symbol="AAPL",
        side="buy",
        type="market",
        order_type="market",
        time_in_force="day",
        status="accepted",
        qty="1",
        filled_qty="0",
        filled_avg_price=None,
        limit_price=None,
        stop_price=None,
        submitted_at=pd.Timestamp.utcnow(),
        order_class="simple",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def alpaca_position(**overrides: Any) -> SimpleNamespace:
    defaults = dict(symbol="AAPL", qty="10", side="long", avg_entry_price="150.0")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def alpaca_account(**overrides: Any) -> SimpleNamespace:
    defaults = dict(cash="50000", equity="60000", buying_power="100000")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def alpaca_trade_update(**overrides: Any) -> SimpleNamespace:
    defaults = dict(
        event="fill",
        execution_id=str(uuid.uuid4()),
        order=alpaca_order(),
        timestamp=pd.Timestamp.utcnow(),
        position_qty=1.0,
        price=150.25,
        qty=1.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class ApiError(Exception):
    """Stand-in for alpaca.common.exceptions.APIError good enough for AlpacaBroker's `except`."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class FakeTradingClient:
    """Minimal stand-in for `alpaca.trading.client.TradingClient`: no network, records what was
    submitted/canceled, and lets tests seed positions/account/open orders for reconciliation."""

    def __init__(self) -> None:
        self.submitted: List[Any] = []
        self.canceled_ids: List[str] = []
        self.canceled_all_count = 0
        self._orders: Dict[str, SimpleNamespace] = {}
        self._positions: List[SimpleNamespace] = []
        self._account = alpaca_account()
        self.fail_next_submit: Optional[Exception] = None

    def submit_order(self, order_data: Any) -> SimpleNamespace:
        if self.fail_next_submit is not None:
            exc, self.fail_next_submit = self.fail_next_submit, None
            raise exc
        self.submitted.append(order_data)

        def _v(x: Any) -> Any:
            return x.value if hasattr(x, "value") else x

        resp = alpaca_order(
            id=str(uuid.uuid4()),
            client_order_id=order_data.client_order_id,
            symbol=order_data.symbol,
            side=_v(order_data.side),
            type=_v(order_data.type),
            time_in_force=_v(order_data.time_in_force),
            qty=str(order_data.qty) if getattr(order_data, "qty", None) is not None else None,
            limit_price=getattr(order_data, "limit_price", None),
            stop_price=getattr(order_data, "stop_price", None),
            status="accepted",
        )
        self._orders[resp.client_order_id] = resp
        return resp

    def cancel_order_by_id(self, order_id: Any) -> None:
        self.canceled_ids.append(order_id)

    def cancel_orders(self) -> list:
        self.canceled_all_count += 1
        return []

    def get_orders(self, filter: Any = None) -> List[SimpleNamespace]:
        return list(self._orders.values())

    def get_all_positions(self) -> List[SimpleNamespace]:
        return list(self._positions)

    def get_account(self) -> SimpleNamespace:
        return self._account

    def set_positions(self, positions: List[SimpleNamespace]) -> None:
        self._positions = positions

    def set_account(self, account: SimpleNamespace) -> None:
        self._account = account


class FakeTradingStream:
    """Stand-in for `alpaca.trading.stream.TradingStream`: records the subscribed handler and lets
    a test drive it without ever opening a websocket."""

    def __init__(self) -> None:
        self.handler = None
        self.ran = False
        self.stopped = False

    def subscribe_trade_updates(self, handler) -> None:
        self.handler = handler

    def run(self) -> None:
        self.ran = True

    def stop(self) -> None:
        self.stopped = True
