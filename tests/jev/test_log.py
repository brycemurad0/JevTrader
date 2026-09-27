from __future__ import annotations

import pandas as pd
import pytest
from typesafe_sdk import Choice

from jevtrader.core.interfaces import JevView
from jevtrader.jev.log import DecisionLog, annotate_outcomes, read_jsonl


def _view(**overrides) -> JevView:
    base = dict(
        probs={"direction": {"up": 0.6, "down": 0.2, "flat": 0.2}},
        top={"direction": "up"},
        confidence={"direction": 0.8},
        latency_ms=42.0,
        late=False,
        model="jev-test",
    )
    base.update(overrides)
    return JevView(**base)


def test_record_then_read_round_trips(tmp_log_path):
    log = DecisionLog(tmp_log_path)
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    questions = {"direction": Choice(instructions="q", criteria={"up": None, "down": None, "flat": None})}
    log.record(symbol="AAPL", state={"px": 100.0, "cost_bps": 5.0}, questions=questions, view=_view(), question_key="direction:5min", ts=ts)

    entries = list(read_jsonl(tmp_log_path))
    assert len(entries) == 1
    e = entries[0]
    assert e["symbol"] == "AAPL"
    assert e["ts"] == ts.isoformat()
    assert e["question_key"] == "direction:5min"
    assert e["state"]["cost_bps"] == 5.0
    assert e["questions"]["direction"]["type"] == "choice"
    assert e["answers"]["probs"]["direction"]["up"] == 0.6
    assert e["answers"]["top"]["direction"] == "up"
    assert e["latency_ms"] == 42.0
    assert e["late"] is False
    assert e["model"] == "jev-test"


def test_append_only_multiple_entries(tmp_log_path):
    log = DecisionLog(tmp_log_path)
    for i in range(3):
        log.record(symbol="AAPL", state={"i": i}, questions={}, view=_view(), question_key="direction:5min")
    entries = list(read_jsonl(tmp_log_path))
    assert len(entries) == 3
    assert [e["state"]["i"] for e in entries] == [0, 1, 2]


def test_read_jsonl_on_missing_file_yields_nothing(tmp_path):
    entries = list(read_jsonl(tmp_path / "does_not_exist.jsonl"))
    assert entries == []


def test_creates_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "decisions.jsonl"
    log = DecisionLog(nested)
    log.record(symbol="AAPL", state={}, questions={}, view=_view(), question_key="direction:5min")
    assert nested.exists()


def test_annotate_outcomes_computes_forward_return():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=10, freq="1min")
    prices = pd.Series([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9], index=idx)
    entries = [
        {"symbol": "AAPL", "ts": idx[0].isoformat(), "question_key": "direction:5min", "state": {"cost_bps": 1.0}},
    ]
    annotated = annotate_outcomes(entries, {"AAPL": prices})
    assert len(annotated) == 1
    e = annotated[0]
    entry_px, exit_px = 100.0, 100.5
    expected_bps = round(1e4 * (exit_px / entry_px - 1.0), 3)
    assert e["fwd_ret_bps"] == pytest.approx(expected_bps)
    assert e["fwd_ts"] == idx[5].isoformat()


def test_annotate_outcomes_none_when_horizon_runs_past_series_end():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min")
    prices = pd.Series([100.0, 100.1, 100.2], index=idx)
    entries = [{"symbol": "AAPL", "ts": idx[0].isoformat(), "question_key": "direction:1h", "state": {}}]
    annotated = annotate_outcomes(entries, {"AAPL": prices})
    assert annotated[0]["fwd_ret_bps"] is None


def test_annotate_outcomes_none_when_symbol_missing_from_prices():
    entries = [{"symbol": "UNKNOWN", "ts": "2026-01-01T00:00:00+00:00", "question_key": "direction:5min", "state": {}}]
    annotated = annotate_outcomes(entries, {})
    assert annotated[0]["fwd_ret_bps"] is None


def test_annotate_outcomes_default_horizon_used_for_non_direction_entries():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=10, freq="1min")
    prices = pd.Series(range(10), index=idx, dtype=float) + 100.0
    entries = [{"symbol": "AAPL", "ts": idx[0].isoformat(), "question_key": "regime", "state": {}}]
    annotated = annotate_outcomes(entries, {"AAPL": prices}, default_horizon="3min")
    assert annotated[0]["fwd_ts"] == idx[3].isoformat()
    assert annotated[0]["fwd_ret_bps"] is not None


def test_annotate_outcomes_does_not_mutate_input():
    entries = [{"symbol": "AAPL", "ts": "2026-01-01T00:00:00+00:00", "question_key": "direction:5min", "state": {}}]
    original = dict(entries[0])
    annotate_outcomes(entries, {})
    assert entries[0] == original
