from __future__ import annotations

import pytest

from jevtrader.jev.backends import (
    DEFAULT_KEV_BASE_URL,
    DEFAULT_LAYA_BASE_URL,
    DecisionBackend,
    base_url_for,
    make_client,
)

from conftest import make_settings


def test_from_env_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("DECISION_BACKEND", raising=False)
    assert DecisionBackend.from_env() is DecisionBackend.OFFLINE
    assert DecisionBackend.from_env(default=DecisionBackend.JEV) is DecisionBackend.JEV


def test_from_env_reads_env_var(monkeypatch):
    monkeypatch.setenv("DECISION_BACKEND", "kev")
    assert DecisionBackend.from_env() is DecisionBackend.KEV
    monkeypatch.setenv("DECISION_BACKEND", "LAYA")  # case-insensitive
    assert DecisionBackend.from_env() is DecisionBackend.LAYA


def test_from_env_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("DECISION_BACKEND", "nonsense")
    with pytest.raises(ValueError):
        DecisionBackend.from_env()


def test_coerce_accepts_enum_or_string():
    assert DecisionBackend.coerce("kev") is DecisionBackend.KEV
    assert DecisionBackend.coerce(DecisionBackend.LAYA) is DecisionBackend.LAYA


def test_base_url_jev_is_none_by_default(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    assert base_url_for("jev") is None


def test_base_url_kev_and_laya_defaults(monkeypatch):
    monkeypatch.delenv("KEV_BASE_URL", raising=False)
    monkeypatch.delenv("LAYA_BASE_URL", raising=False)
    assert base_url_for("kev") == DEFAULT_KEV_BASE_URL
    assert base_url_for("laya") == DEFAULT_LAYA_BASE_URL


def test_base_url_respects_env_override(monkeypatch):
    monkeypatch.setenv("KEV_BASE_URL", "http://gpu-box:9000")
    assert base_url_for("kev") == "http://gpu-box:9000"


def test_base_url_offline_raises():
    with pytest.raises(ValueError):
        base_url_for("offline")


def test_make_client_offline_raises():
    with pytest.raises(ValueError):
        make_client("offline")


def test_make_client_kev_uses_local_base_url_and_placeholder_key(monkeypatch):
    monkeypatch.delenv("KEV_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_API_KEY", raising=False)
    client = make_client("kev")
    try:
        assert client._config.base_url == DEFAULT_KEV_BASE_URL.rstrip("/")
        assert client._config.api_key  # non-empty placeholder, SDK would otherwise reject it
        assert client._config.default_model == "kev-latest"
    finally:
        client.close()


def test_make_client_laya_respects_local_model_api_key_env(monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "my-laya-key")
    client = make_client("laya")
    try:
        assert client._config.api_key == "my-laya-key"
        assert client._config.default_model == "laya-latest"
    finally:
        client.close()


def test_make_client_jev_uses_settings_key_and_sdk_default_base_url():
    settings = make_settings(typesafe_api_key="secret-key")
    client = make_client("jev", settings=settings)
    try:
        assert client._config.api_key == "secret-key"
        assert client._config.default_model  # settings.jev_model default ("jev-latest"), constructed without error
    finally:
        client.close()


def test_make_client_explicit_model_and_timeout_override():
    client = make_client("kev", model="kev-4b-custom", timeout=5.0)
    try:
        assert client._config.default_model == "kev-4b-custom"
        assert client._config.timeout == 5.0
    finally:
        client.close()


def test_make_client_async_variant():
    from typesafe_sdk import AsyncTypeSafeClient

    client = make_client("laya", is_async=True)
    try:
        assert isinstance(client, AsyncTypeSafeClient)
    finally:
        import asyncio

        asyncio.run(client.aclose())
