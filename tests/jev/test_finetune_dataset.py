from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from jevtrader.jev.finetune.dataset import (
    DatasetConfig,
    balance_classes,
    build_dataset,
    build_examples,
    dedupe_examples,
    from_decision_log,
    load_examples_jsonl,
    split_chronological,
    write_jsonl,
    write_records_jsonl,
)


def _trending_bars(n=400, drift=0.05, seed=0, freq="1min"):
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(drift, 0.02, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.01, "low": close - 0.01, "close": close, "volume": rng.uniform(50, 150, n)},
        index=idx,
    )


def test_build_examples_produces_valid_labels():
    bars = _trending_bars(n=300, drift=0.1)
    cfg = DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, stride=1, dedupe=False)
    examples = build_examples("BTC/USD", bars, cfg)
    assert len(examples) > 0
    for ex in examples:
        assert ex["labels"]["direction"] in ("up", "down", "flat")
        assert ex["labels"]["regime"] in ("trending_up", "trending_down", "mean_reverting", "volatile_chop", "quiet")
        assert isinstance(ex["labels"]["risk_off"], bool)
        assert 0 <= ex["labels"]["entry_quality"] <= 4
        assert isinstance(ex["state"], dict)
        assert "direction" in ex["questions"]
        assert ex["outcomes"]["fwd_ret_bps"] is not None


def test_no_lookahead_in_features():
    """The state for example i must be computable from bars up to and including i only --
    changing bars strictly AFTER position i must not change example i's state."""
    bars = _trending_bars(n=300, seed=1)
    cfg = DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, stride=1, dedupe=False, include_regime=False, include_entry_quality=False)
    examples_a = build_examples("BTC/USD", bars, cfg)

    mutated = bars.copy()
    # Blow up the tail (after every example's decision point plus its horizon) -- should not
    # change any example's state (though it may change labels for examples near the tail, so
    # compare only the state of the first half of examples, safely before the mutated region).
    mutated_close = mutated["close"].to_numpy().copy()
    tail_start = int(len(mutated) * 0.8)
    mutated_close[tail_start:] *= 5.0
    mutated["close"] = mutated_close
    mutated["high"] = mutated_close + 0.01
    mutated["low"] = mutated_close - 0.01
    examples_b = build_examples("BTC/USD", mutated, cfg)

    n_check = min(len(examples_a), len(examples_b)) // 2  # well before tail_start
    for a, b in zip(examples_a[:n_check], examples_b[:n_check]):
        assert a["state"] == b["state"]


def test_direction_label_respects_cost_threshold():
    bars = _trending_bars(n=200, drift=0.0, seed=2)
    cfg_cheap = DatasetConfig(horizon="5min", cost_bps=0.0, lookback_bars=50, dedupe=False, include_regime=False, include_entry_quality=False)
    cfg_expensive = DatasetConfig(horizon="5min", cost_bps=10_000.0, lookback_bars=50, dedupe=False, include_regime=False, include_entry_quality=False)
    cheap = build_examples("BTC/USD", bars, cfg_cheap)
    expensive = build_examples("BTC/USD", bars, cfg_expensive)
    # an impossibly high cost means nothing ever clears it -> everything is "flat"
    assert all(ex["labels"]["direction"] == "flat" for ex in expensive)
    # a zero cost means "flat" only for an exact zero return, vanishingly likely on random data
    assert any(ex["labels"]["direction"] != "flat" for ex in cheap)


def test_risk_off_label_from_drawdown_threshold():
    bars = _trending_bars(n=200, drift=0.0, seed=3)
    lenient = build_examples("BTC/USD", bars, DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, dedupe=False, risk_off_drawdown_bps=100000.0, include_entry_quality=False))
    strict = build_examples("BTC/USD", bars, DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, dedupe=False, risk_off_drawdown_bps=-1_000_000.0, include_entry_quality=False))
    assert not any(ex["labels"]["risk_off"] for ex in lenient)
    assert all(ex["labels"]["risk_off"] for ex in strict)  # an absurdly low (negative) threshold: every window "exceeds" it


def test_entry_quality_bucketed_0_to_4():
    bars = _trending_bars(n=200, seed=4)
    examples = build_examples("BTC/USD", bars, DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, dedupe=False, include_regime=False))
    levels = {ex["labels"]["entry_quality"] for ex in examples}
    assert levels.issubset({0, 1, 2, 3, 4})


def test_feature_fn_receives_only_past_data_and_is_merged():
    bars = _trending_bars(n=200, seed=5)
    seen_max_ts = []

    def feature_fn(symbol, window):
        seen_max_ts.append(window.index[-1])
        return {"tfm_p_up_gt_cost": 0.7}

    cfg = DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, dedupe=False, include_regime=False, include_entry_quality=False)
    examples = build_examples("BTC/USD", bars, cfg, feature_fn=feature_fn)
    assert examples
    for ex, max_ts in zip(examples, seen_max_ts):
        assert pd.Timestamp(ex["ts"]) == max_ts  # window ends exactly at the decision bar
        assert ex["state"]["forecast"]["tfm_p_up_gt_cost"] == pytest.approx(0.7)


def test_dedupe_never_increases_count_and_leaves_unique_states():
    # Note: `state` includes time-of-day features (minute_of_day, dow), which are unique for
    # every bar within a short span, so a short flat-price series has no exact duplicates to
    # remove -- this only checks dedupe is safe (<=) and leaves no duplicates, not that it
    # necessarily shrinks a short series. `test_dedupe_examples_helper_directly` below tests the
    # actual collapsing behavior directly.
    flat_bars = pd.DataFrame(
        {"open": [100.0] * 200, "high": [100.0] * 200, "low": [100.0] * 200, "close": [100.0] * 200, "volume": [1.0] * 200},
        index=pd.date_range("2026-01-01", periods=200, freq="1min", tz="UTC"),
    )
    cfg_no_dedupe = DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, dedupe=False, include_regime=False, include_entry_quality=False)
    cfg_dedupe = DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, dedupe=True, include_regime=False, include_entry_quality=False)
    without = build_examples("BTC/USD", flat_bars, cfg_no_dedupe)
    with_dedupe = build_examples("BTC/USD", flat_bars, cfg_dedupe)
    assert len(with_dedupe) <= len(without)
    states = [json.dumps(ex["state"], sort_keys=True) for ex in with_dedupe]
    assert len(states) == len(set(states))  # no duplicates remain


def test_dedupe_examples_helper_directly():
    examples = [
        {"state": {"a": 1}, "ts": "t1"},
        {"state": {"a": 1}, "ts": "t2"},  # identical state, dropped
        {"state": {"a": 2}, "ts": "t3"},
    ]
    deduped = dedupe_examples(examples)
    assert [e["ts"] for e in deduped] == ["t1", "t3"]


def test_max_examples_caps_and_subsamples_deterministically():
    bars = _trending_bars(n=500, seed=6)
    cfg = DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, dedupe=False, include_regime=False, include_entry_quality=False, max_examples=10)
    examples = build_examples("BTC/USD", bars, cfg)
    assert len(examples) == 10
    examples2 = build_examples("BTC/USD", bars, cfg)
    assert [e["ts"] for e in examples] == [e["ts"] for e in examples2]


def test_split_chronological_respects_fractions_and_order():
    examples = [{"ts": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i), "labels": {}} for i in range(100)]
    splits = split_chronological(examples, embargo=0)
    assert len(splits.train) == 70
    assert len(splits.calibration) == 10
    assert len(splits.dev) == 10
    assert len(splits.test) == 10
    # chronological, non-overlapping, in order
    all_ts = [e["ts"] for e in splits.train] + [e["ts"] for e in splits.calibration] + [e["ts"] for e in splits.dev] + [e["ts"] for e in splits.test]
    assert all_ts == sorted(all_ts)


def test_split_chronological_embargo_drops_trailing_examples():
    examples = [{"ts": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i), "labels": {}} for i in range(100)]
    splits = split_chronological(examples, embargo=5)
    assert len(splits.train) == 65  # 70 - 5 embargo
    assert len(splits.calibration) == 5  # 10 - 5 embargo
    assert len(splits.dev) == 5
    assert len(splits.test) == 10  # last split keeps everything, no embargo needed after it


def test_split_chronological_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_chronological([], train_frac=0.5, calibration_frac=0.5, dev_frac=0.5, test_frac=0.5)


def test_build_dataset_embargo_prevents_forward_leakage_across_boundary():
    """No train example's forward-return window should reach into the calibration split's time
    range (that would leak calibration-period price action into a training label)."""
    bars = _trending_bars(n=600, seed=7)
    cfg = DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, stride=1, dedupe=False)
    splits = build_dataset("BTC/USD", bars, cfg)
    assert splits.calibration, "test setup should produce a non-empty calibration split"
    calibration_start = pd.Timestamp(splits.calibration[0]["ts"])
    horizon_td = pd.Timedelta(cfg.horizon)
    for ex in splits.train:
        assert pd.Timestamp(ex["ts"]) + horizon_td <= calibration_start


def test_balance_classes_equalizes_counts():
    examples = [{"ts": i, "labels": {"direction": "up" if i < 80 else "down"}} for i in range(100)]
    balanced = balance_classes(examples, label_key="direction", seed=0)
    counts = {}
    for e in balanced:
        counts[e["labels"]["direction"]] = counts.get(e["labels"]["direction"], 0) + 1
    assert counts["up"] == counts["down"] == 20


def test_balance_classes_is_deterministic():
    examples = [{"ts": i, "labels": {"direction": "up" if i % 3 == 0 else "down"}} for i in range(90)]
    a = balance_classes(examples, seed=42)
    b = balance_classes(examples, seed=42)
    assert [e["ts"] for e in a] == [e["ts"] for e in b]


def test_write_jsonl_schema_is_kev_compatible(tmp_path):
    bars = _trending_bars(n=200, seed=8)
    cfg = DatasetConfig(horizon="5min", cost_bps=2.0, lookback_bars=50, dedupe=False)
    examples = build_examples("BTC/USD", bars, cfg)
    out = tmp_path / "train.jsonl"
    n = write_jsonl(examples, out, source="test")
    assert n == len(examples)
    lines = out.read_text().splitlines()
    assert len(lines) == n
    for line in lines:
        rec = json.loads(line)
        assert "state" in rec and "questions" in rec and rec["questions"]
        for qname, q in rec["questions"].items():
            assert q["type"] in ("choice", "noul", "score")
            assert "label" in q
            if q["type"] == "choice":
                assert q["label"] in q["criteria"]
                assert isinstance(q["label"], str)
            elif q["type"] == "noul":
                assert isinstance(q["label"], bool)
            elif q["type"] == "score":
                assert isinstance(q["label"], int)
                assert 0 <= q["label"] < len(q["criteria"])


def test_write_jsonl_meta_round_trips_through_load_examples_jsonl(tmp_path):
    bars = _trending_bars(n=200, seed=9)
    cfg = DatasetConfig(horizon="5min", cost_bps=3.0, lookback_bars=50, dedupe=False)
    examples = build_examples("BTC/USD", bars, cfg)
    out = tmp_path / "test.jsonl"
    write_jsonl(examples, out)
    reloaded = load_examples_jsonl(out)
    assert len(reloaded) == len(examples)
    for orig, back in zip(examples, reloaded):
        assert back["state"] == orig["state"]
        assert back["labels"] == orig["labels"]
        assert back["outcomes"] == orig["outcomes"]
        assert back["cost_bps"] == orig["cost_bps"]
        assert back["symbol"] == orig["symbol"]


def test_write_jsonl_without_meta_omits_meta_key(tmp_path):
    bars = _trending_bars(n=200, seed=10)
    cfg = DatasetConfig(horizon="5min", cost_bps=1.0, lookback_bars=50, dedupe=False)
    examples = build_examples("BTC/USD", bars, cfg)[:5]
    out = tmp_path / "clean.jsonl"
    write_jsonl(examples, out, with_meta=False)
    with pytest.raises(ValueError):
        load_examples_jsonl(out)
    rec = json.loads(out.read_text().splitlines()[0])
    assert "_meta" not in rec


def test_write_records_jsonl_skips_empty_question_records(tmp_path):
    out = tmp_path / "distill.jsonl"
    n = write_records_jsonl([{"state": {}, "questions": {}}, {"state": {"a": 1}, "questions": {"x": {"type": "noul", "label": True}}}], out)
    assert n == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 1


def test_from_decision_log_produces_valid_labels_and_soft_targets():
    entries = [
        {
            "symbol": "AAPL",
            "ts": "2026-01-01T00:00:00+00:00",
            "question_key": "direction:5min",
            "late": False,
            "state": {"px": 100.0},
            "questions": {"direction": {"type": "choice", "instructions": "q", "criteria": {"up": None, "down": None, "flat": None}}},
            "answers": {"probs": {"direction": {"up": 0.6, "down": 0.2, "flat": 0.2}}, "top": {"direction": "up"}, "confidence": {"direction": 0.7}},
        }
    ]
    records = from_decision_log(entries)
    assert len(records) == 1
    q = records[0]["questions"]["direction"]
    assert q["label"] == "up"
    assert q["target"]["up"] == pytest.approx(0.6)
    assert sum(q["target"].values()) == pytest.approx(1.0)


def test_from_decision_log_translates_noul_target_keys():
    entries = [
        {
            "symbol": "AAPL",
            "ts": "2026-01-01T00:00:00+00:00",
            "late": False,
            "state": {},
            "questions": {"risk_off": {"type": "noul", "instructions": "q", "criteria": {"true": None, "false": None}}},
            "answers": {"probs": {"risk_off": {"yes": 0.8, "no": 0.2}}, "top": {"risk_off": "yes"}, "confidence": {}},
        }
    ]
    records = from_decision_log(entries)
    q = records[0]["questions"]["risk_off"]
    assert q["label"] is True
    assert q["target"] == {"true": 0.8, "false": 0.2}


def test_from_decision_log_skips_late_entries_by_default():
    entries = [
        {
            "symbol": "AAPL",
            "ts": "2026-01-01T00:00:00+00:00",
            "late": True,
            "state": {},
            "questions": {"direction": {"type": "choice", "criteria": {"up": None, "down": None, "flat": None}}},
            "answers": {"probs": {"direction": {"up": 0.9, "down": 0.05, "flat": 0.05}}, "top": {"direction": "up"}, "confidence": {}},
        }
    ]
    assert from_decision_log(entries) == []
    assert from_decision_log(entries, skip_late=False) != []


def test_from_decision_log_filters_by_backend():
    entries = [
        {"symbol": "AAPL", "ts": "t1", "late": False, "extra": {"backend": "kev"}, "state": {}, "questions": {"direction": {"type": "choice", "criteria": {"up": None, "down": None, "flat": None}}}, "answers": {"probs": {"direction": {"up": 0.5, "down": 0.3, "flat": 0.2}}, "top": {"direction": "up"}, "confidence": {}}},
        {"symbol": "AAPL", "ts": "t2", "late": False, "extra": {"backend": "laya"}, "state": {}, "questions": {"direction": {"type": "choice", "criteria": {"up": None, "down": None, "flat": None}}}, "answers": {"probs": {"direction": {"up": 0.4, "down": 0.4, "flat": 0.2}}, "top": {"direction": "up"}, "confidence": {}}},
    ]
    kev_only = from_decision_log(entries, only_backend="kev")
    assert len(kev_only) == 1
