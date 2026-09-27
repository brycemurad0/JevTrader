"""Local-only FastAPI dashboard: live L2 order books, strategy list, open orders, a manual order
ticket (routed through the same `RiskGate` as every strategy order), account/risk panel, a Jev
decision feed, and a kill switch.

Binds to 127.0.0.1 by default; `run_dashboard` refuses any other host unless the caller
explicitly opts in with `allow_remote=True` -- this is a local trading terminal for the machine
it runs on, not a public service.

`create_app(state, runner=None)` takes a `DashboardState` (see `jevtrader.dashboard.state`) that
some other code -- a `LiveRunner`, or `jevtrader.dashboard.demo` -- writes into; the FastAPI layer
here only reads/relays it, so the same app works identically wired to a live paper-trading run or
to nothing at all (an idle dashboard with an empty snapshot).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from jevtrader.dashboard.state import DashboardState
from jevtrader.dashboard.templates import INDEX_HTML

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ManualOrderRequest(BaseModel):
    symbol: str
    side: str
    qty: float
    type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    post_only: bool = False


def create_app(state: DashboardState, runner: Optional[Any] = None) -> FastAPI:
    app = FastAPI(title="JevTrader Dashboard")
    app.state.dashboard_state = state
    app.state.runner = runner

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "mode": state.mode}

    @app.get("/api/snapshot")
    async def snapshot() -> dict:
        return state.snapshot()

    @app.post("/api/orders")
    async def submit_order(req: ManualOrderRequest) -> dict:
        result = state.submit_manual_order(req.model_dump())
        if not result.get("ok", False):
            raise HTTPException(status_code=400, detail=result.get("reason", "order rejected"))
        return result

    @app.delete("/api/orders/{client_order_id}")
    async def cancel_order(client_order_id: str) -> dict:
        return state.cancel_order(client_order_id)

    @app.post("/api/kill")
    async def kill_switch() -> dict:
        result = state.kill()
        if not result.get("ok", False):
            raise HTTPException(status_code=400, detail=result.get("reason", "kill switch unavailable"))
        return result

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(state.snapshot())
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("dashboard websocket error")

    return app


def run_dashboard(
    state: DashboardState,
    runner: Optional[Any] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    allow_remote: bool = False,
) -> None:
    """Build the app and serve it with uvicorn (blocking). Refuses to bind anywhere but
    localhost/loopback unless `allow_remote=True`."""
    if host not in _LOOPBACK_HOSTS and not allow_remote:
        raise ValueError(
            f"refusing to bind the JevTrader dashboard to {host!r}: this is a local-only trading "
            "UI with a manual order ticket and a kill switch. Pass allow_remote=True if you really "
            "mean to expose it beyond this machine (e.g. behind your own auth/reverse proxy)."
        )
    import uvicorn

    app = create_app(state, runner)
    uvicorn.run(app, host=host, port=port)
