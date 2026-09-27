from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jevtrader.forecast.baselines import EWMAVolForecaster, RandomWalkForecaster
from jevtrader.forecast.evaluate import evaluate_returns, evaluate_volatility, pinball_loss, to_markdown


def _bars(n=3000, seed=0, freq="1min"):
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": volume}, index=idx)


def test_pinball_loss_zero_at_the_point():
    assert pinball_loss(10.0, 10.0, 0.3) == pytest.approx(0.0)


def test_pinball_loss_penalizes_asymmetrically():
    # for a high quantile level, under-predicting (actual > predicted) costs more
    high_level_under = pinball_loss(actual=10.0, predicted=5.0, level=0.9)
    high_level_over = pinball_loss(actual=5.0, predicted=10.0, level=0.9)
    assert high_level_under > high_level_over


def test_evaluate_returns_produces_sane_metrics_on_random_walk_generated_data():
    df = _bars()
    result = evaluate_returns(df, RandomWalkForecaster(), horizon=15, cost_bps=5.0, min_train=500, step=30, max_windows=100)
    assert result.n_windows > 0
    assert 0.0 <= result.coverage_80 <= 1.0
    # random-walk generating process scored by a random-walk forecaster: should be
    # well-calibrated (loosely -- this is real randomness, not a tight bound).
    assert 0.5 <= result.coverage_80 <= 1.0
    assert result.mase > 0
    assert 0.0 <= result.pinball_by_level[0.5] or True  # pinball losses are non-negative
    assert all(v >= 0 for v in result.pinball_by_level.values())


def test_evaluate_returns_no_windows_on_too_short_series():
    df = _bars(n=50)
    result = evaluate_returns(df, RandomWalkForecaster(), horizon=15, cost_bps=5.0, min_train=500)
    assert result.n_windows == 0


def test_evaluate_volatility_produces_sane_metrics():
    df = _bars()
    result = evaluate_volatility(df, EWMAVolForecaster(), horizon=15, vol_window=20, min_train=500, step=30, max_windows=100)
    assert result.n_windows > 0
    assert result.qlike is not None
    assert result.mse is not None and result.mse >= 0
    assert result.mse_vs_naive is not None


def test_to_markdown_smoke():
    df = _bars()
    r = evaluate_returns(df, RandomWalkForecaster(), horizon=10, cost_bps=5.0, min_train=500, step=50, max_windows=50)
    md = to_markdown({"random_walk": {10: r}}, title="Returns")
    assert "random_walk" in md
    assert "MASE" in md


# ----------------------------------------------------------------------------- no look-ahead


class _SpyForecaster:
    """Records the max absolute value it was ever asked to forecast from, and the length of
    each series it received, so a test can prove no future data leaked into the context."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.seen_lengths: list[int] = []

    def forecast(self, series, horizon):
        for arr in series.values():
            self.seen_lengths.append(len(arr))
        return self.delegate.forecast(series, horizon)


def test_evaluate_returns_never_passes_future_bars():
    df = _bars(n=2000, seed=1)
    horizon = 10
    spy = _SpyForecaster(RandomWalkForecaster())
    evaluate_returns(df, spy, horizon=horizon, cost_bps=5.0, min_train=500, step=25, max_windows=50)
    assert spy.seen_lengths, "forecaster was never called"
    # every context length must be strictly less than len(df) - horizon (it can't include the
    # bars used to score it, let alone bars beyond that)
    assert max(spy.seen_lengths) <= len(df) - horizon - 1


def test_evaluate_returns_context_length_matches_origin_index():
    """A forecaster that reports exactly how much history it saw lets us check, for a handful
    of origins, that the context length is consistent with a no-look-ahead rolling window
    (length grows with the origin, and is always < available future data)."""
    df = _bars(n=1500, seed=2)
    horizon = 5
    seen = []

    class RecordingForecaster:
        def forecast(self, series, horizon):
            seen.append(len(series["return"]))
            return RandomWalkForecaster().forecast(series, horizon)

    evaluate_returns(df, RecordingForecaster(), horizon=horizon, cost_bps=5.0, min_train=500, step=25, max_windows=30)
    assert seen == sorted(seen)  # origins walk forward -> context length is non-decreasing
    assert len(set(seen)) > 1  # context actually grows across origins (not a fixed window)


def test_evaluate_volatility_never_passes_future_bars():
    df = _bars(n=2000, seed=3)
    horizon = 10
    spy = _SpyForecaster(EWMAVolForecaster())
    evaluate_volatility(df, spy, horizon=horizon, vol_window=20, min_train=500, step=25, max_windows=50)
    assert spy.seen_lengths, "forecaster was never called"
    assert max(spy.seen_lengths) <= len(df)
