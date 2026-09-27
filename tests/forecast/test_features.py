from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from jevtrader.forecast.baselines import RandomWalkForecaster
from jevtrader.forecast.features import FEATURE_KEYS, clear_cache, forecast_features


def _bars(n=200, seed=0, freq="1min"):
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": volume}, index=idx)


def setup_function(_):
    clear_cache()


def test_forecast_features_shape_has_all_keys():
    df = _bars()
    out = forecast_features("BTC/USD", df, RandomWalkForecaster(), horizon=5, cost_bps=5.0)
    assert set(out.keys()) == set(FEATURE_KEYS)


def test_forecast_features_deterministic_given_same_inputs():
    df = _bars()
    out1 = forecast_features("BTC/USD", df, RandomWalkForecaster(), horizon=5, cost_bps=5.0, cache={})
    out2 = forecast_features("BTC/USD", df, RandomWalkForecaster(), horizon=5, cost_bps=5.0, cache={})
    assert out1 == out2


def test_forecast_features_json_serializable_and_compact():
    df = _bars()
    out = forecast_features("BTC/USD", df, RandomWalkForecaster(), horizon=5, cost_bps=5.0)
    encoded = json.dumps(out, sort_keys=True, separators=(",", ":"))
    assert json.loads(encoded) == out
    assert len(encoded) < 400, f"forecast features should be token-cheap, got {len(encoded)} bytes: {encoded}"


def test_forecast_features_handles_too_little_history():
    idx = pd.date_range("2026-01-01", periods=2, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": [100.0, 100.5], "high": [100.1, 100.6], "low": [99.9, 100.4], "close": [100.0, 100.5], "volume": [10.0, 12.0]}, index=idx)
    out = forecast_features("AAPL", df, RandomWalkForecaster(), horizon=5, cost_bps=1.0)
    assert all(v is None for v in out.values())


def test_forecast_features_empty_bars_raises():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    with pytest.raises(ValueError):
        forecast_features("AAPL", df, RandomWalkForecaster(), horizon=5)


def test_forecast_features_caches_per_symbol_and_last_bar_ts():
    df = _bars()
    calls = {"n": 0}

    class CountingForecaster:
        def forecast(self, series, horizon):
            calls["n"] += 1
            return RandomWalkForecaster().forecast(series, horizon)

    cache = {}
    # keep strong references to every forecaster instance for the whole test: a short-lived
    # instance can be garbage-collected and have its `id()` reused by the next one, which would
    # produce a false cache hit here (the cache key includes `id(forecaster)`).
    f1, f2 = CountingForecaster(), CountingForecaster()
    forecast_features("BTC/USD", df, f1, horizon=5, cost_bps=5.0, cache=cache)
    forecast_features("BTC/USD", df, f2, horizon=5, cost_bps=5.0, cache=cache)  # different instance -> no cache hit
    assert calls["n"] == 2

    forecaster = CountingForecaster()
    forecast_features("BTC/USD", df, forecaster, horizon=5, cost_bps=5.0, cache=cache)
    forecast_features("BTC/USD", df, forecaster, horizon=5, cost_bps=5.0, cache=cache)
    assert calls["n"] == 3  # second call with the SAME forecaster + same last bar ts hits cache


def test_forecast_features_direction_probs_in_unit_interval():
    df = _bars(seed=3)
    out = forecast_features("BTC/USD", df, RandomWalkForecaster(), horizon=10, cost_bps=2.0)
    assert 0.0 <= out["tfm_p_up_gt_cost"] <= 1.0
    assert 0.0 <= out["tfm_p_down_gt_cost"] <= 1.0
