from __future__ import annotations

from jevtrader.core.interfaces import JevView
from jevtrader.jev.advisor import ReplayJevAdvisor
from jevtrader.jev.log import DecisionLog


def _view(top_label: str) -> JevView:
    return JevView(
        probs={"direction": {"up": 0.6, "down": 0.2, "flat": 0.2}},
        top={"direction": top_label},
        confidence={"direction": 0.8},
        latency_ms=10.0,
        late=False,
        model="jev-test",
    )


def test_replay_serves_entries_in_recorded_order(tmp_log_path):
    log = DecisionLog(tmp_log_path)
    log.record(symbol="AAPL", state={"i": 0}, questions={}, view=_view("up"), question_key="direction:5min")
    log.record(symbol="AAPL", state={"i": 1}, questions={}, view=_view("down"), question_key="direction:5min")
    log.record(symbol="AAPL", state={"i": 2}, questions={}, view=_view("flat"), question_key="direction:5min")

    replay = ReplayJevAdvisor.from_log(tmp_log_path)
    v1 = replay.direction("AAPL", {}, "5min")
    v2 = replay.direction("AAPL", {}, "5min")
    v3 = replay.direction("AAPL", {}, "5min")
    assert [v1.top["direction"], v2.top["direction"], v3.top["direction"]] == ["up", "down", "flat"]


def test_replay_exhausted_queue_returns_none(tmp_log_path):
    log = DecisionLog(tmp_log_path)
    log.record(symbol="AAPL", state={}, questions={}, view=_view("up"), question_key="direction:5min")
    replay = ReplayJevAdvisor.from_log(tmp_log_path)
    assert replay.direction("AAPL", {}, "5min") is not None
    assert replay.direction("AAPL", {}, "5min") is None


def test_replay_keys_by_symbol_and_question_key_independently(tmp_log_path):
    log = DecisionLog(tmp_log_path)
    log.record(symbol="AAPL", state={}, questions={}, view=_view("up"), question_key="direction:5min")
    log.record(symbol="AAPL", state={}, questions={}, view=_view("up"), question_key="regime")
    log.record(symbol="MSFT", state={}, questions={}, view=_view("down"), question_key="direction:5min")

    replay = ReplayJevAdvisor.from_log(tmp_log_path)
    assert replay.direction("AAPL", {}, "5min").top["direction"] == "up"
    assert replay.regime("AAPL", {}) is not None
    assert replay.direction("MSFT", {}, "5min").top["direction"] == "down"
    # AAPL's direction:5min queue is now empty even though other queues still have entries
    assert replay.direction("AAPL", {}, "5min") is None


def test_replay_from_in_memory_entries():
    entries = [
        {"symbol": "AAPL", "question_key": "direction:1h", "answers": {"probs": {"direction": {"up": 0.7, "down": 0.1, "flat": 0.2}}, "top": {"direction": "up"}, "confidence": {"direction": 0.9}}, "latency_ms": 5.0, "late": False, "model": "jev-test"}
    ]
    replay = ReplayJevAdvisor(entries)
    v = replay.direction("AAPL", {}, "1h")
    assert v is not None
    assert v.top["direction"] == "up"
    assert v.probs["direction"]["up"] == 0.7


def test_replay_ask_uses_sorted_question_names_as_key():
    entries = [
        {"symbol": "_", "question_key": "a,b", "answers": {"probs": {"x": {"y": 1.0}}, "top": {}, "confidence": {}}, "latency_ms": 0.0, "late": False, "model": "m"}
    ]
    replay = ReplayJevAdvisor(entries)
    v = replay.ask({}, {"a": object(), "b": object()})
    assert v is not None
    assert v.probs["x"]["y"] == 1.0
