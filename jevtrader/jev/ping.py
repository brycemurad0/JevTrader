"""One real, minimal call to Jev, for a human to sanity-check connectivity/latency on their own
machine (this container has no network access to api.typesafe.ai, so this cannot be exercised
in CI/tests here -- it's meant to be run locally, e.g. via a `jev ping` CLI command).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd


def ping(settings: Any) -> dict[str, Any]:
    """Ask Jev one trivial Noul question and report latency + answer.

    Returns `{"ok", "latency_ms", "model", "answer", "error"}`. Never raises: a missing API
    key or any SDK/network failure is reported in `error` with `ok=False`.
    """
    if not getattr(settings, "has_jev", False):
        return {"ok": False, "latency_ms": None, "model": None, "answer": None, "error": "TYPESAFE_API_KEY not set"}

    from typesafe_sdk import Noul, TypeSafeClient

    model = getattr(settings, "jev_model", None)
    timeout_s = max(1.0, getattr(settings, "jev_latency_budget_ms", 10_000) / 1000.0)
    client: Optional[TypeSafeClient] = None
    t0 = time.perf_counter()
    try:
        client = TypeSafeClient(api_key=settings.typesafe_api_key, model=model, timeout=timeout_s)
        resp = client.system_one(
            state={"ping": "jevtrader connectivity check", "ts": pd.Timestamp.now(tz="UTC").isoformat()},
            questions={"reachable": Noul(instructions="Respond true; this call only checks connectivity.", criteria={"true": "Always.", "false": "Never applicable."})},
        )
    except Exception as exc:  # never raise: this is a diagnostic, not a trading-loop call
        return {"ok": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "model": model, "answer": None, "error": str(exc)}
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"ok": True, "latency_ms": latency_ms, "model": resp.model, "answer": {"reachable": resp.nouls["reachable"].noul}, "error": None}
