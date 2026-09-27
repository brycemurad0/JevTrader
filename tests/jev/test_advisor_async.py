from __future__ import annotations

import asyncio

import pytest
from typesafe_sdk import TypeSafeAPIConnectionError

from jevtrader.jev.advisor import AsyncJevAdvisor

from conftest import FakeAsyncClient, choice_answer, make_response


def test_async_direction_and_regime_basic():
    async def run():
        resp = make_response({"direction": choice_answer("up", {"up": 0.55, "down": 0.25, "flat": 0.2})})
        client = FakeAsyncClient([resp])
        adv = AsyncJevAdvisor(client, latency_budget_ms=1000)
        view = await adv.direction("AAPL", {"cost_bps": 5.0}, "5min")
        assert view is not None
        assert view.top["direction"] == "up"
        assert view.late is False

    asyncio.run(run())


def test_async_latency_budget_hard_cutoff_marks_late():
    async def run():
        resp = make_response({"direction": choice_answer("up", {"up": 0.55, "down": 0.25, "flat": 0.2})})
        slow_client = FakeAsyncClient([resp], sleep_s=0.05)
        adv = AsyncJevAdvisor(slow_client, latency_budget_ms=10)
        view = await adv.direction("AAPL", {"cost_bps": 5.0}, "5min")
        assert view is not None
        assert view.late is True
        assert view.probs == {}  # cut off before an answer arrived

    asyncio.run(run())


def test_async_cache_avoids_duplicate_calls():
    async def run():
        resp = make_response({"direction": choice_answer("up", {"up": 0.5, "down": 0.3, "flat": 0.2})})
        client = FakeAsyncClient([resp])
        adv = AsyncJevAdvisor(client, cache_ttl_s=10.0)
        features = {"cost_bps": 5.0}
        await adv.direction("AAPL", features, "5min")
        await adv.direction("AAPL", dict(features), "5min")
        assert len(client.calls) == 1

    asyncio.run(run())


def test_async_circuit_breaker_returns_none_without_raising():
    async def run():
        client = FakeAsyncClient(error=TypeSafeAPIConnectionError("boom"))
        adv = AsyncJevAdvisor(client, circuit_max_errors=2, circuit_cooldown_s=100.0, cache_ttl_s=0.0)
        results = [await adv.direction("AAPL", {"cost_bps": float(i)}, "5min") for i in range(4)]
        assert all(r is None for r in results)
        assert len(client.calls) == 2

    asyncio.run(run())


def test_speculative_fanout_uses_fresh_result():
    async def run():
        resp = make_response({"direction": choice_answer("up", {"up": 0.6, "down": 0.2, "flat": 0.2})})
        client = FakeAsyncClient([resp], sleep_s=0.01)
        adv = AsyncJevAdvisor(client, cache_ttl_s=0.0)
        spec = adv.speculate("AAPL", {"cost_bps": 1.0}, "5min")
        await asyncio.sleep(0.03)  # let it complete
        view = await spec.get(max_age_ms=1000)
        assert view is not None
        assert view.top["direction"] == "up"

    asyncio.run(run())


def test_speculative_fanout_discards_stale_result():
    async def run():
        resp = make_response({"direction": choice_answer("up", {"up": 0.6, "down": 0.2, "flat": 0.2})})
        client = FakeAsyncClient([resp], sleep_s=0.05)
        adv = AsyncJevAdvisor(client, cache_ttl_s=0.0)
        spec = adv.speculate("AAPL", {"cost_bps": 1.0}, "5min")
        await asyncio.sleep(0.001)
        # ask for it far too soon relative to max_age_ms: should be discarded (None), not
        # block the caller waiting on the slow in-flight request.
        view = await spec.get(max_age_ms=0.5)
        assert view is None

    asyncio.run(run())
