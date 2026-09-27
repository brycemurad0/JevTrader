from __future__ import annotations

import time

import pytest
from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPITimeoutError

from jevtrader.core.broker import TradingMode
from jevtrader.jev.advisor import JevAdvisor, OfflineJevAdvisor, make_advisor
from jevtrader.jev.log import DecisionLog, read_jsonl

from conftest import FakeClient, choice_answer, make_response, make_settings, noul_answer, score_answer


def test_response_to_jevview_conversion_choice_score_noul():
    resp = make_response(
        {
            "direction": choice_answer("up", {"up": 0.6, "down": 0.2, "flat": 0.2}, confidence=0.8),
            "conviction": score_answer(2.1, {0: 0.05, 1: 0.15, 2: 0.6, 3: 0.2}, confidence=0.7),
            "risk_off": noul_answer(0.9),
        }
    )
    client = FakeClient([resp])
    adv = JevAdvisor(client, latency_budget_ms=1000)
    view = adv.ask({"px": 100}, {"direction": object(), "conviction": object(), "risk_off": object()})

    assert view is not None
    assert view.probs["direction"] == {"up": 0.6, "down": 0.2, "flat": 0.2}
    assert view.top["direction"] == "up"
    assert view.confidence["direction"] == pytest.approx(0.8)

    assert view.probs["conviction"] == {"0": 0.05, "1": 0.15, "2": 0.6, "3": 0.2}
    assert view.raw["conviction"]["expected"] == pytest.approx(2.1)
    assert "legend" in view.raw["conviction"]

    assert view.probs["risk_off"]["yes"] == pytest.approx(0.9)
    assert view.probs["risk_off"]["no"] == pytest.approx(0.1)
    assert view.top["risk_off"] == "yes"
    assert view.late is False
    assert view.model == "jev-test"


def test_latency_budget_marks_late_but_still_returns_answer():
    resp = make_response({"direction": choice_answer("flat", {"up": 0.3, "down": 0.3, "flat": 0.4})})
    slow_client = FakeClient([resp], sleep_s=0.05)
    adv = JevAdvisor(slow_client, latency_budget_ms=10)
    view = adv.direction("AAPL", {"cost_bps": 5.0}, "5min")
    assert view is not None
    assert view.late is True
    assert view.latency_ms >= 40


def test_within_budget_is_not_late():
    resp = make_response({"direction": choice_answer("flat", {"up": 0.3, "down": 0.3, "flat": 0.4})})
    client = FakeClient([resp])
    adv = JevAdvisor(client, latency_budget_ms=5000)
    view = adv.direction("AAPL", {"cost_bps": 5.0}, "5min")
    assert view.late is False


def test_cache_avoids_duplicate_calls_for_same_state():
    resp = make_response({"direction": choice_answer("up", {"up": 0.5, "down": 0.3, "flat": 0.2})})
    client = FakeClient([resp])
    adv = JevAdvisor(client, cache_ttl_s=10.0)
    features = {"cost_bps": 5.0, "px": 100.0}
    v1 = adv.direction("AAPL", features, "5min")
    v2 = adv.direction("AAPL", dict(features), "5min")  # equal but distinct dict object
    assert len(client.calls) == 1
    assert v1 == v2


def test_cache_distinguishes_symbols_questions_and_state():
    resp = make_response({"direction": choice_answer("up", {"up": 0.5, "down": 0.3, "flat": 0.2})})
    client = FakeClient([resp, resp, resp])
    adv = JevAdvisor(client, cache_ttl_s=10.0)
    adv.direction("AAPL", {"cost_bps": 5.0}, "5min")
    adv.direction("MSFT", {"cost_bps": 5.0}, "5min")  # different symbol
    adv.direction("AAPL", {"cost_bps": 5.0}, "1h")  # different horizon -> different question_key
    adv.direction("AAPL", {"cost_bps": 6.0}, "5min")  # different state
    assert len(client.calls) == 4


def test_cache_ttl_expires():
    resp = make_response({"direction": choice_answer("up", {"up": 0.5, "down": 0.3, "flat": 0.2})})
    client = FakeClient([resp, resp])
    adv = JevAdvisor(client, cache_ttl_s=0.01)
    features = {"cost_bps": 5.0}
    adv.direction("AAPL", features, "5min")
    time.sleep(0.03)
    adv.direction("AAPL", features, "5min")
    assert len(client.calls) == 2


def test_rate_limiter_skips_calls_once_bucket_is_empty():
    resp = make_response({"direction": choice_answer("up", {"up": 0.5, "down": 0.3, "flat": 0.2})})
    client = FakeClient([resp] * 5)
    adv = JevAdvisor(client, rate_limit_per_sec=0.0, rate_limit_burst=1.0, cache_ttl_s=0.0)
    v1 = adv.direction("AAPL", {"cost_bps": 1.0}, "5min")
    v2 = adv.direction("AAPL", {"cost_bps": 2.0}, "5min")  # different state -> bypasses cache, hits limiter
    assert v1 is not None
    assert v2 is None
    assert len(client.calls) == 1


def test_circuit_breaker_opens_after_consecutive_errors_and_returns_none():
    client = FakeClient(error=TypeSafeAPIConnectionError("boom"))
    adv = JevAdvisor(client, circuit_max_errors=3, circuit_cooldown_s=100.0, cache_ttl_s=0.0)
    results = []
    for i in range(5):
        results.append(adv.direction("AAPL", {"cost_bps": float(i)}, "5min"))
    assert all(r is None for r in results)
    # after 3 consecutive failures the breaker opens; the 4th and 5th distinct-state calls
    # should be skipped without even reaching the (still-failing) client.
    assert len(client.calls) == 3


def test_circuit_breaker_half_opens_after_cooldown():
    client = FakeClient(error=TypeSafeAPIConnectionError("boom"))
    adv = JevAdvisor(client, circuit_max_errors=1, circuit_cooldown_s=0.01, cache_ttl_s=0.0)
    assert adv.direction("AAPL", {"cost_bps": 1.0}, "5min") is None
    assert len(client.calls) == 1
    # immediately retrying should be skipped (breaker open)
    assert adv.direction("AAPL", {"cost_bps": 2.0}, "5min") is None
    assert len(client.calls) == 1
    time.sleep(0.02)
    # after cooldown, a trial call is allowed through (and fails again here)
    assert adv.direction("AAPL", {"cost_bps": 3.0}, "5min") is None
    assert len(client.calls) == 2


def test_circuit_breaker_closes_again_after_a_successful_trial():
    err = TypeSafeAPIConnectionError("boom")
    ok = make_response({"direction": choice_answer("up", {"up": 0.5, "down": 0.3, "flat": 0.2})})
    client = FakeClient(sequence=[err, ok, ok])
    adv = JevAdvisor(client, circuit_max_errors=1, circuit_cooldown_s=0.01, cache_ttl_s=0.0)
    assert adv.direction("AAPL", {"cost_bps": 1.0}, "5min") is None  # fails, opens breaker
    time.sleep(0.02)
    v = adv.direction("AAPL", {"cost_bps": 2.0}, "5min")  # half-open trial succeeds
    assert v is not None
    # breaker is closed again: an immediate further call goes straight through, no skip
    v2 = adv.direction("AAPL", {"cost_bps": 3.0}, "5min")
    assert v2 is not None
    assert len(client.calls) == 3


def test_never_raises_on_unexpected_exception():
    client = FakeClient(error=RuntimeError("totally unexpected"))
    adv = JevAdvisor(client, cache_ttl_s=0.0)
    view = adv.direction("AAPL", {"cost_bps": 1.0}, "5min")
    assert view is None  # swallowed, not raised


def test_never_raises_on_timeout_error():
    client = FakeClient(error=TypeSafeAPITimeoutError(10.0))
    adv = JevAdvisor(client, cache_ttl_s=0.0)
    view = adv.direction("AAPL", {"cost_bps": 1.0}, "5min")
    assert view is None


def test_decision_log_records_calls(tmp_log_path):
    resp = make_response({"direction": choice_answer("up", {"up": 0.6, "down": 0.2, "flat": 0.2})})
    client = FakeClient([resp])
    log = DecisionLog(tmp_log_path)
    adv = JevAdvisor(client, decision_log=log)
    adv.direction("AAPL", {"cost_bps": 4.0}, "5min")
    entries = list(read_jsonl(tmp_log_path))
    assert len(entries) == 1
    e = entries[0]
    assert e["symbol"] == "AAPL"
    assert e["question_key"] == "direction:5min"
    assert e["answers"]["probs"]["direction"]["up"] == pytest.approx(0.6)
    assert e["late"] is False
    assert e["model"] == "jev-test"


def test_decision_log_not_written_on_cache_hit(tmp_log_path):
    resp = make_response({"direction": choice_answer("up", {"up": 0.6, "down": 0.2, "flat": 0.2})})
    client = FakeClient([resp])
    log = DecisionLog(tmp_log_path)
    adv = JevAdvisor(client, decision_log=log, cache_ttl_s=10.0)
    features = {"cost_bps": 4.0}
    adv.direction("AAPL", features, "5min")
    adv.direction("AAPL", dict(features), "5min")
    entries = list(read_jsonl(tmp_log_path))
    assert len(entries) == 1  # only the real call was logged


def test_make_advisor_offline_without_key():
    settings = make_settings(typesafe_api_key="")
    adv = make_advisor(settings, TradingMode.PAPER)
    assert isinstance(adv, OfflineJevAdvisor)


def test_make_advisor_offline_in_backtest_even_with_key():
    settings = make_settings(typesafe_api_key="secret")
    adv = make_advisor(settings, TradingMode.BACKTEST)
    assert isinstance(adv, OfflineJevAdvisor)


def test_make_advisor_real_advisor_when_key_and_paper_mode():
    settings = make_settings(typesafe_api_key="secret")
    resp = make_response({"direction": choice_answer("up", {"up": 0.5, "down": 0.3, "flat": 0.2})})
    fake_client = FakeClient([resp])
    adv = make_advisor(settings, "paper", client=fake_client)
    assert isinstance(adv, JevAdvisor)
    view = adv.direction("AAPL", {"cost_bps": 1.0}, "5min")
    assert view is not None
    assert len(fake_client.calls) == 1


def test_make_advisor_accepts_string_or_enum_mode():
    settings = make_settings(typesafe_api_key="secret")
    fake_client = FakeClient()
    assert isinstance(make_advisor(settings, "live", client=fake_client), JevAdvisor)
    assert isinstance(make_advisor(settings, TradingMode.LIVE, client=fake_client), JevAdvisor)
