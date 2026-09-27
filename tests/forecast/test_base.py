from __future__ import annotations

import numpy as np
import pytest

from jevtrader.forecast.base import QUANTILE_LEVELS, QuantileForecast, constant_quantile_forecast


def _simple_forecast(h=1) -> QuantileForecast:
    point = np.zeros(h)
    quantiles = {level: np.full(h, (level - 0.5) * 200.0) for level in QUANTILE_LEVELS}  # -80..+80 bps
    return QuantileForecast(point=point, quantiles=quantiles)


def test_quantile_forecast_rejects_missing_levels():
    with pytest.raises(ValueError):
        QuantileForecast(point=np.zeros(3), quantiles={0.1: np.zeros(3)})


def test_quantile_forecast_rejects_length_mismatch():
    with pytest.raises(ValueError):
        QuantileForecast(point=np.zeros(3), quantiles={level: np.zeros(2) for level in QUANTILE_LEVELS})


def test_prob_above_monotonic_in_threshold():
    qf = _simple_forecast()
    thresholds = np.linspace(-200, 200, 41)
    probs = [qf.prob_above(t) for t in thresholds]
    # non-increasing as threshold rises
    assert all(a >= b - 1e-9 for a, b in zip(probs, probs[1:]))


def test_prob_below_monotonic_in_threshold():
    qf = _simple_forecast()
    thresholds = np.linspace(-200, 200, 41)
    probs = [qf.prob_below(t) for t in thresholds]
    assert all(a <= b + 1e-9 for a, b in zip(probs, probs[1:]))


def test_prob_above_plus_prob_below_bracket_one():
    # P(X>t) + P(X<t) should be close to 1 (they overlap exactly at t, so this is an
    # approximate identity away from clamped tails).
    qf = _simple_forecast()
    for t in (-50.0, 0.0, 37.0):
        total = qf.prob_above(t) + qf.prob_below(t)
        assert 0.9 <= total <= 1.1


def test_tails_are_clamped_never_zero_or_one():
    qf = _simple_forecast()
    assert 0.0 < qf.prob_above(1e6) < 1e-2
    assert 0.0 < qf.prob_below(-1e6) < 1e-2
    assert qf.prob_above(1e6) > 0.0
    assert qf.prob_below(1e6) < 1.0


def test_median_prob_above_near_half():
    qf = _simple_forecast()
    median = qf.quantile_at(0.5)
    assert abs(qf.prob_above(median) - 0.5) < 0.05


def test_scaled_multiplies_point_and_quantiles():
    qf = _simple_forecast(h=2)
    scaled = qf.scaled(2.0)
    assert np.allclose(scaled.point, qf.point * 2.0)
    for level in QUANTILE_LEVELS:
        assert np.allclose(scaled.quantiles[level], qf.quantiles[level] * 2.0)


def test_constant_quantile_forecast_builder():
    point = [1.0, 2.0, 3.0]
    spread = {level: [level * 10.0] * 3 for level in QUANTILE_LEVELS}
    qf = constant_quantile_forecast(point, spread)
    assert qf.horizon == 3
    assert qf.quantile_at(0.3, h=0) == pytest.approx(3.0)


def test_forecast_protocol_runtime_check():
    from jevtrader.forecast.base import Forecaster

    class Dummy:
        def forecast(self, series, horizon):
            return {}

    assert isinstance(Dummy(), Forecaster)
    assert not isinstance(object(), Forecaster)
