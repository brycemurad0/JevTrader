"""FastAPI dashboard endpoints via TestClient. Requires httpx (FastAPI's TestClient dependency);
skipped cleanly if it isn't installed, per the task's instructions."""

from __future__ import annotations

import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from jevtrader.dashboard.app import _LOOPBACK_HOSTS, create_app, run_dashboard  # noqa: E402
from jevtrader.dashboard.state import DashboardState  # noqa: E402


def _client(state=None):
    state = state or DashboardState()
    app = create_app(state)
    return TestClient(app), state


def test_index_serves_html():
    client, _ = _client()
    r = client.get("/")
    assert r.status_code == 200
    assert "JevTrader" in r.text or "JEVTRADER" in r.text
    assert "text/html" in r.headers["content-type"]


def test_health():
    client, state = _client()
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "mode": "PAPER"}


def test_snapshot_endpoint_matches_state():
    state = DashboardState()
    state.update_strategy("s1", status="running")
    client, _ = _client(state)
    r = client.get("/api/snapshot")
    assert r.status_code == 200
    assert r.json()["strategies"]["s1"]["status"] == "running"


def test_manual_order_success():
    state = DashboardState()
    received = {}

    def _submit(order):
        received.update(order)
        return {"ok": True, "client_order_id": "c1"}

    state.bind_actions(submit_order=_submit)
    client, _ = _client(state)
    r = client.post("/api/orders", json={"symbol": "AAPL", "side": "buy", "qty": 1, "type": "market"})
    assert r.status_code == 200
    assert r.json()["client_order_id"] == "c1"
    assert received["symbol"] == "AAPL"


def test_manual_order_rejected_returns_400():
    state = DashboardState()
    state.bind_actions(submit_order=lambda o: {"ok": False, "reason": "risk gate rejected"})
    client, _ = _client(state)
    r = client.post("/api/orders", json={"symbol": "AAPL", "side": "buy", "qty": 1})
    assert r.status_code == 400
    assert "risk gate rejected" in r.json()["detail"]


def test_manual_order_no_broker_wired_returns_400():
    client, _ = _client()
    r = client.post("/api/orders", json={"symbol": "AAPL", "side": "buy", "qty": 1})
    assert r.status_code == 400


def test_cancel_order():
    state = DashboardState()
    canceled = []
    state.bind_actions(cancel_order=lambda cid: canceled.append(cid))
    client, _ = _client(state)
    r = client.delete("/api/orders/c1")
    assert r.status_code == 200
    assert canceled == ["c1"]


def test_kill_switch_success():
    state = DashboardState()
    killed = []
    state.bind_actions(kill_switch=lambda: killed.append(True))
    client, _ = _client(state)
    r = client.post("/api/kill")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert killed == [True]


def test_kill_switch_unavailable_returns_400():
    client, _ = _client()
    r = client.post("/api/kill")
    assert r.status_code == 400


def test_websocket_pushes_snapshot():
    state = DashboardState()
    state.set_mode("LIVE")
    client, _ = _client(state)
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert data["mode"] == "LIVE"


def test_run_dashboard_refuses_non_loopback_host_by_default():
    state = DashboardState()
    with pytest.raises(ValueError, match="allow_remote"):
        run_dashboard(state, host="0.0.0.0")


def test_loopback_hosts_constant_covers_localhost_variants():
    assert {"127.0.0.1", "localhost", "::1"} <= _LOOPBACK_HOSTS
