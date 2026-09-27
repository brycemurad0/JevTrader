"""One real, minimal call to a decision backend, for a human to sanity-check connectivity and
latency on their own machine (this container has no network access to api.typesafe.ai, and no
GPU to serve Kev/Laya on, so this cannot be exercised end-to-end in CI/tests here -- it's meant
to be run locally, e.g. via a `jev ping` CLI command, against hosted Jev or a local Kev/Laya
server started per `docs/DECISION_MODELS.md`).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd

from jevtrader.jev.backends import DecisionBackend, make_client


def ping(settings: Any, backend: Optional[Any] = None) -> dict[str, Any]:
    """Ask one trivial Noul question of `backend` (default: `$DECISION_BACKEND`, else `"jev"`)
    and report latency + answer.

    Returns `{"ok", "backend", "latency_ms", "model", "answer", "error"}`. Never raises: a
    missing API key, an unreachable local server, or any other SDK/network failure is reported
    in `error` with `ok=False`.
    """
    resolved_backend = DecisionBackend.coerce(backend) if backend is not None else DecisionBackend.from_env(default=DecisionBackend.JEV)

    if resolved_backend is DecisionBackend.OFFLINE:
        return {"ok": False, "backend": resolved_backend.value, "latency_ms": None, "model": None, "answer": None, "error": "backend=offline has no endpoint to ping (it's OfflineJevAdvisor, in-process)"}
    if resolved_backend is DecisionBackend.JEV and not getattr(settings, "has_jev", False):
        return {"ok": False, "backend": resolved_backend.value, "latency_ms": None, "model": None, "answer": None, "error": "TYPESAFE_API_KEY not set"}

    from typesafe_sdk import Noul

    timeout_s = max(1.0, getattr(settings, "jev_latency_budget_ms", 10_000) / 1000.0)
    client: Optional[Any] = None
    t0 = time.perf_counter()
    try:
        client = make_client(resolved_backend, settings=settings, timeout=timeout_s)
        resp = client.system_one(
            state={"ping": "jevtrader connectivity check", "ts": pd.Timestamp.now(tz="UTC").isoformat()},
            questions={"reachable": Noul(instructions="Respond true; this call only checks connectivity.", criteria={"true": "Always.", "false": "Never applicable."})},
        )
    except Exception as exc:  # never raise: this is a diagnostic, not a trading-loop call
        return {
            "ok": False,
            "backend": resolved_backend.value,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model": None,
            "answer": None,
            "error": str(exc),
        }
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "ok": True,
        "backend": resolved_backend.value,
        "latency_ms": latency_ms,
        "model": resp.model,
        "answer": {"reachable": resp.nouls["reachable"].noul},
        "error": None,
    }
