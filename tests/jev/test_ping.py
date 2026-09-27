from __future__ import annotations

import json

import pytest
from typesafe_sdk import SystemOneResponse

from jevtrader.jev.backends import DEFAULT_KEV_BASE_URL, DecisionBackend
from jevtrader.jev.ping import ping

from conftest import make_settings


def test_ping_without_api_key_reports_missing_key_and_never_raises():
    settings = make_settings(typesafe_api_key="")
    result = ping(settings)
    assert result["ok"] is False
    assert result["latency_ms"] is None
    assert "TYPESAFE_API_KEY" in result["error"]


def test_ping_success(monkeypatch):
    payload = {
        "model": "jev-latest",
        "usage": {"input_tokens": 5, "output_tokens": 1},
        "answers": {"reachable": {"type": "noul", "noul": 0.99}},
    }
    resp = SystemOneResponse.model_validate_json(json.dumps(payload))

    class FakeTypeSafeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def system_one(self, *, state, questions, **kw):
            return resp

        def close(self):
            pass

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", FakeTypeSafeClient)

    settings = make_settings(typesafe_api_key="secret")
    result = ping(settings)
    assert result["ok"] is True
    assert result["model"] == "jev-latest"
    assert result["answer"]["reachable"] == pytest.approx(0.99)
    assert result["latency_ms"] >= 0
    assert result["error"] is None


def test_ping_handles_client_error_gracefully(monkeypatch):
    class FailingClient:
        def __init__(self, **kwargs):
            pass

        def system_one(self, *, state, questions, **kw):
            raise RuntimeError("network unreachable")

        def close(self):
            pass

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", FailingClient)
    settings = make_settings(typesafe_api_key="secret")
    result = ping(settings)
    assert result["ok"] is False
    assert "network unreachable" in result["error"]


def test_ping_offline_backend_reports_no_endpoint():
    settings = make_settings(typesafe_api_key="")
    result = ping(settings, backend=DecisionBackend.OFFLINE)
    assert result["ok"] is False
    assert result["backend"] == "offline"
    assert "no endpoint" in result["error"]


def test_ping_kev_backend_needs_no_api_key(monkeypatch):
    payload = {"model": "kev-4b", "usage": {"input_tokens": 3, "output_tokens": 1}, "answers": {"reachable": {"type": "noul", "noul": 1.0}}}
    resp = SystemOneResponse.model_validate_json(json.dumps(payload))

    class FakeKevClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def system_one(self, *, state, questions, **kw):
            return resp

        def close(self):
            pass

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", FakeKevClient)
    settings = make_settings(typesafe_api_key="")  # no Jev key at all
    result = ping(settings, backend="kev")
    assert result["ok"] is True
    assert result["backend"] == "kev"
    assert result["model"] == "kev-4b"


def test_ping_respects_decision_backend_env(monkeypatch):
    monkeypatch.setenv("DECISION_BACKEND", "laya")
    monkeypatch.delenv("LAYA_BASE_URL", raising=False)

    class UnreachableClient:
        def __init__(self, **kwargs):
            pass

        def system_one(self, *, state, questions, **kw):
            raise ConnectionError("no laya server running")

        def close(self):
            pass

    monkeypatch.setattr("typesafe_sdk.TypeSafeClient", UnreachableClient)
    settings = make_settings(typesafe_api_key="")
    result = ping(settings)
    assert result["backend"] == "laya"
    assert result["ok"] is False
    assert "no laya server running" in result["error"]
