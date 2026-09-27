from __future__ import annotations

import time

from jevtrader.core.interfaces import JevView
from jevtrader.jev.advisor import OfflineJevAdvisor, ShadowAdvisor
from jevtrader.jev.log import DecisionLog, read_jsonl


class _SlowShadow:
    """A shadow whose answer is only available after `delay_s`, to prove ShadowAdvisor never
    blocks the caller on it."""

    def __init__(self, view: JevView, delay_s: float = 0.05):
        self.view = view
        self.delay_s = delay_s
        self.calls = 0

    def direction(self, symbol, features, horizon):
        self.calls += 1
        time.sleep(self.delay_s)
        return self.view

    def regime(self, symbol, features):
        self.calls += 1
        time.sleep(self.delay_s)
        return self.view

    def ask(self, state, questions):
        self.calls += 1
        time.sleep(self.delay_s)
        return self.view


class _BrokenShadow:
    def direction(self, symbol, features, horizon):
        raise RuntimeError("shadow exploded")

    def regime(self, symbol, features):
        raise RuntimeError("shadow exploded")

    def ask(self, state, questions):
        raise RuntimeError("shadow exploded")


class _NoneShadow:
    def direction(self, symbol, features, horizon):
        return None

    def regime(self, symbol, features):
        return None

    def ask(self, state, questions):
        return None


def _view(label="up"):
    return JevView(
        probs={"direction": {"up": 0.5, "down": 0.3, "flat": 0.2}},
        top={"direction": label},
        confidence={"direction": 0.6},
        latency_ms=1.0,
        late=False,
        model="shadow-model",
    )


def test_shadow_advisor_returns_primary_result_immediately():
    primary = OfflineJevAdvisor()
    slow = _SlowShadow(_view(), delay_s=0.2)
    shadow_adv = ShadowAdvisor(primary, {"slow": slow})
    features = {"cost_bps": 5.0, "ret_bps": {"5": 1, "15": 1, "30": 1, "60": 1}}

    t0 = time.monotonic()
    result = shadow_adv.direction("AAPL", features, "5min")
    elapsed = time.monotonic() - t0

    assert result is not None
    assert elapsed < 0.1, "ShadowAdvisor must not block on a slow shadow"
    shadow_adv.wait(timeout=1.0)
    assert slow.calls == 1
    shadow_adv.close()


def test_shadow_advisor_logs_shadow_answers_tagged_by_backend(tmp_log_path):
    primary = OfflineJevAdvisor()
    shadow_view = _view(label="down")
    log = DecisionLog(tmp_log_path)
    shadow_adv = ShadowAdvisor(primary, {"kev": _SlowShadow(shadow_view, delay_s=0.0)}, decision_log=log)

    shadow_adv.direction("AAPL", {"cost_bps": 5.0}, "5min")
    shadow_adv.wait(timeout=1.0)

    entries = list(read_jsonl(tmp_log_path))
    assert len(entries) == 1
    e = entries[0]
    assert e["symbol"] == "AAPL"
    assert e["question_key"] == "direction:5min"
    assert e["extra"]["backend"] == "kev"
    assert e["extra"]["shadow"] is True
    assert e["answers"]["top"]["direction"] == "down"
    shadow_adv.close()


def test_shadow_advisor_multiple_shadows_all_logged(tmp_log_path):
    primary = OfflineJevAdvisor()
    log = DecisionLog(tmp_log_path)
    shadow_adv = ShadowAdvisor(
        primary,
        {"kev": _SlowShadow(_view("up"), delay_s=0.0), "laya": _SlowShadow(_view("down"), delay_s=0.0)},
        decision_log=log,
    )
    shadow_adv.regime("AAPL", {"vol_bps": 10.0})
    shadow_adv.wait(timeout=1.0)
    entries = list(read_jsonl(tmp_log_path))
    assert {e["extra"]["backend"] for e in entries} == {"kev", "laya"}
    assert all(e["question_key"] == "regime" for e in entries)
    shadow_adv.close()


def test_shadow_advisor_swallows_broken_shadow_without_raising(tmp_log_path):
    primary = OfflineJevAdvisor()
    log = DecisionLog(tmp_log_path)
    shadow_adv = ShadowAdvisor(primary, {"broken": _BrokenShadow()}, decision_log=log)
    result = shadow_adv.direction("AAPL", {"cost_bps": 1.0}, "5min")
    assert result is not None  # primary's answer, unaffected
    shadow_adv.wait(timeout=1.0)
    assert list(read_jsonl(tmp_log_path)) == []  # nothing logged for a failed shadow
    shadow_adv.close()


def test_shadow_advisor_none_result_not_logged(tmp_log_path):
    primary = OfflineJevAdvisor()
    log = DecisionLog(tmp_log_path)
    shadow_adv = ShadowAdvisor(primary, {"quiet": _NoneShadow()}, decision_log=log)
    shadow_adv.ask({"px": 1}, {})
    shadow_adv.wait(timeout=1.0)
    assert list(read_jsonl(tmp_log_path)) == []
    shadow_adv.close()


def test_shadow_advisor_with_no_shadows_is_a_noop_passthrough():
    primary = OfflineJevAdvisor()
    shadow_adv = ShadowAdvisor(primary, {})
    result = shadow_adv.direction("AAPL", {"cost_bps": 1.0}, "5min")
    assert result is not None
    shadow_adv.wait()
    shadow_adv.close()


def test_shadow_advisor_works_without_a_decision_log():
    primary = OfflineJevAdvisor()
    shadow_adv = ShadowAdvisor(primary, {"kev": _SlowShadow(_view(), delay_s=0.0)})
    result = shadow_adv.regime("AAPL", {"vol_bps": 1.0})
    assert result is not None
    shadow_adv.wait(timeout=1.0)  # should not raise even with no decision_log attached
    shadow_adv.close()
