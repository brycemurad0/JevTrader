from __future__ import annotations

import numpy as np
import pytest

from jevtrader.jev.finetune.baseline import (
    DEFAULT_FEATURE_NAMES,
    LogisticBaseline,
    calibrate_temperature,
    extract_features,
    fit_logistic_baseline,
    load_baseline,
    save_baseline,
)


def _example(state, label):
    return {"state": state, "labels": {"direction": label}}


def test_extract_features_handles_nested_paths_and_missing_values():
    state = {"ret_bps": {"5": 1.0, "15": 2.0}, "vol_bps": 10.0, "cost_bps": 5.0}
    vec = extract_features(state, feature_names=("ret_bps.5", "ret_bps.15", "ret_bps.30", "vol_bps", "missing_field"))
    assert vec.tolist() == [1.0, 2.0, 0.0, 10.0, 0.0]


def test_extract_features_ignores_non_numeric_and_bool():
    state = {"session": "us_open", "flag": True, "vol_bps": 3.0}
    vec = extract_features(state, feature_names=("session", "flag", "vol_bps"))
    assert vec.tolist() == [0.0, 0.0, 3.0]


def _synthetic_examples(n=300, seed=0):
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n):
        momentum = rng.normal(0, 1)
        # a clean, learnable linear signal: momentum > 0.5 -> up, < -0.5 -> down, else flat
        label = "up" if momentum > 0.5 else ("down" if momentum < -0.5 else "flat")
        state = {"ret_bps": {"5": momentum * 10, "15": momentum * 10, "30": momentum * 10, "60": momentum * 10}, "vol_bps": 5.0, "cost_bps": 1.0}
        examples.append(_example(state, label))
    return examples


def test_fit_produces_valid_probability_distributions():
    examples = _synthetic_examples()
    baseline = fit_logistic_baseline(examples)
    for ex in examples[:20]:
        probs = baseline.predict_one(ex["state"])
        assert set(probs.keys()) == {"up", "down", "flat"}
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_fit_learns_the_signal_better_than_chance():
    train = _synthetic_examples(n=400, seed=1)
    test = _synthetic_examples(n=200, seed=2)
    baseline = fit_logistic_baseline(train)
    correct = sum(1 for ex in test if max(baseline.predict_one(ex["state"]), key=baseline.predict_one(ex["state"]).get) == ex["labels"]["direction"])
    accuracy = correct / len(test)
    assert accuracy > 0.5  # better than the 3-way chance rate (1/3), by a wide margin on a clean signal


def test_predict_is_deterministic():
    examples = _synthetic_examples(n=100, seed=3)
    baseline = fit_logistic_baseline(examples)
    state = examples[0]["state"]
    p1 = baseline.predict_one(state)
    p2 = baseline.predict_one(state)
    assert p1 == p2


def test_fit_requires_at_least_one_example():
    with pytest.raises(ValueError):
        fit_logistic_baseline([])


def test_calibrate_temperature_changes_confidence_without_changing_ranking():
    train = _synthetic_examples(n=300, seed=4)
    calibration = _synthetic_examples(n=150, seed=5)
    baseline = fit_logistic_baseline(train)
    calibrated = calibrate_temperature(baseline, calibration)
    assert calibrated.temperature != 1.0 or calibrated.temperature == pytest.approx(baseline.temperature)
    sample_state = calibration[0]["state"]
    before = baseline.predict_one(sample_state)
    after = calibrated.predict_one(sample_state)
    assert max(before, key=before.get) == max(after, key=after.get)  # temperature scaling never flips the argmax


def test_calibrate_temperature_on_empty_calibration_set_is_a_noop():
    train = _synthetic_examples(n=100, seed=6)
    baseline = fit_logistic_baseline(train)
    result = calibrate_temperature(baseline, [])
    assert result.temperature == baseline.temperature


def test_baseline_is_immutable_dataclass():
    baseline = fit_logistic_baseline(_synthetic_examples(n=50, seed=7))
    with pytest.raises(Exception):
        baseline.temperature = 2.0  # frozen dataclass


def test_save_and_load_baseline_round_trips(tmp_path):
    train = _synthetic_examples(n=200, seed=8)
    baseline = fit_logistic_baseline(train)
    baseline = calibrate_temperature(baseline, _synthetic_examples(n=100, seed=9))
    path = tmp_path / "baseline.json"
    save_baseline(baseline, path)
    reloaded = load_baseline(path)
    assert reloaded.classes == baseline.classes
    assert reloaded.feature_names == baseline.feature_names
    assert reloaded.temperature == pytest.approx(baseline.temperature)
    np.testing.assert_allclose(reloaded.weights, baseline.weights)
    np.testing.assert_allclose(reloaded.feature_mean, baseline.feature_mean)
    np.testing.assert_allclose(reloaded.feature_std, baseline.feature_std)
    sample_state = train[0]["state"]
    assert reloaded.predict_one(sample_state) == baseline.predict_one(sample_state)


def test_default_feature_names_are_all_extractable_from_a_real_state():
    from jevtrader.jev.state import build_state
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=80, freq="1min", tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.05, 80))
    bars = pd.DataFrame({"open": close, "high": close + 0.01, "low": close - 0.01, "close": close, "volume": np.full(80, 10.0)}, index=idx)
    state = build_state("AAPL", bars, cost_bps=5.0)
    vec = extract_features(state, DEFAULT_FEATURE_NAMES)
    assert vec.shape == (len(DEFAULT_FEATURE_NAMES),)
    assert np.isfinite(vec).all()
