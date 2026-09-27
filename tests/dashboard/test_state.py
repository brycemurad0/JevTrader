"""DashboardState: writer/reader methods and action wiring (bind_actions)."""

from __future__ import annotations

from jevtrader.dashboard.state import DashboardState


def test_initial_snapshot_shape():
    state = DashboardState()
    snap = state.snapshot()
    assert snap["mode"] == "PAPER"
    assert snap["account"]["equity"] == 0.0
    assert snap["positions"] == {}
    assert snap["open_orders"] == {}
    assert snap["kill_switch_engaged"] is False


def test_update_account_computes_drawdown():
    state = DashboardState()
    state.update_account(cash=1000, equity=10_000, buying_power=20_000)
    state.update_account(cash=1000, equity=9_000, buying_power=20_000)
    snap = state.snapshot()
    assert snap["account"]["equity"] == 9_000
    assert snap["account"]["peak_equity"] == 10_000
    assert abs(snap["account"]["drawdown"] - 0.10) < 1e-9


def test_update_strategy_and_position():
    state = DashboardState()
    state.update_strategy("meanrev:AAPL", status="running", pnl=12.5)
    state.update_position("AAPL", qty=10, avg_price=100.0)
    snap = state.snapshot()
    assert snap["strategies"]["meanrev:AAPL"]["status"] == "running"
    assert snap["positions"]["AAPL"]["qty"] == 10


def test_upsert_order_and_removal_on_terminal_status():
    state = DashboardState()
    state.upsert_order({"client_order_id": "c1", "status": "new", "symbol": "AAPL"})
    assert "c1" in state.snapshot()["open_orders"]
    state.upsert_order({"client_order_id": "c1", "status": "filled", "symbol": "AAPL"})
    assert "c1" not in state.snapshot()["open_orders"]


def test_update_book_truncates_to_max_depth():
    state = DashboardState(max_book_depth=2)
    bids = [{"price": 100 - i, "size": 1} for i in range(5)]
    asks = [{"price": 101 + i, "size": 1} for i in range(5)]
    state.update_book("AAPL", bids, asks)
    book = state.snapshot()["books"]["AAPL"]
    assert len(book["bids"]) == 2
    assert len(book["asks"]) == 2


def test_push_trade_keeps_recent_first():
    state = DashboardState()
    state.push_trade("AAPL", price=100.0, size=1.0)
    state.push_trade("AAPL", price=101.0, size=1.0)
    trades = state.snapshot()["trades"]["AAPL"]
    assert trades[0]["price"] == 101.0  # most recent first


def test_push_jev_decision():
    state = DashboardState()
    state.push_jev_decision({"question": "direction", "symbol": "AAPL", "top": {"direction": "up"}, "latency_ms": 50, "late": False})
    feed = state.snapshot()["jev_feed"]
    assert feed[0]["symbol"] == "AAPL"


def test_submit_manual_order_without_binding_reports_unavailable():
    state = DashboardState()
    result = state.submit_manual_order({"symbol": "AAPL", "side": "buy", "qty": 1})
    assert result["ok"] is False


def test_bind_actions_wires_manual_order_cancel_and_kill():
    state = DashboardState()
    submitted = []
    canceled = []
    killed = []
    state.bind_actions(
        submit_order=lambda o: (submitted.append(o), {"ok": True, "client_order_id": "c1"})[1],
        cancel_order=lambda cid: canceled.append(cid),
        kill_switch=lambda: killed.append(True),
    )
    result = state.submit_manual_order({"symbol": "AAPL", "side": "buy", "qty": 1})
    assert result["ok"] is True
    assert submitted == [{"symbol": "AAPL", "side": "buy", "qty": 1}]

    state.cancel_order("c1")
    assert canceled == ["c1"]

    kill_result = state.kill()
    assert kill_result["ok"] is True
    assert killed == [True]
    assert state.snapshot()["kill_switch_engaged"] is True


def test_kill_without_binding_reports_unavailable_but_marks_engaged():
    state = DashboardState()
    result = state.kill()
    assert result["ok"] is False
    assert state.snapshot()["kill_switch_engaged"] is True
