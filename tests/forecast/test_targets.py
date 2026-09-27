from __future__ import annotations

import math

import numpy as np
import pytest

from jevtrader.forecast.base import QUANTILE_LEVELS, QuantileForecast
from jevtrader.forecast.targets import (
    cumulative_log_return_bps,
    exp_levels,
    last_valid,
    log_return_series,
    log_volume_series,
    realized_vol_series,
    spread_bps_series,
    to_bps,
)


def test_log_return_series_basic():
    closes = np.array([100.0, 101.0, 100.0, 102.0])
    rets = log_return_series(closes)
    assert rets.shape == (3,)
    assert rets[0] == pytest.approx(math.log(101.0 / 100.0))


def test_log_return_series_rejects_nonpositive_prices():
    with pytest.raises(ValueError):
        log_return_series(np.array([100.0, 0.0, 50.0]))


def test_realized_vol_series_matches_manual_std():
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 50)))
    window = 10
    rv = realized_vol_series(closes, window)
    rets = log_return_series(closes)
    # spot-check one fully-populated window
    manual = np.std(rets[5 : 5 + window], ddof=0)
    assert rv[5 + window - 1] == pytest.approx(manual, rel=1e-9)
    assert np.all(np.isnan(rv[: window - 1]))


def test_log_volume_series_floors_zero():
    vols = np.array([0.0, 1.0, 100.0])
    out = log_volume_series(vols)
    assert np.all(np.isfinite(out))
    assert out[1] == pytest.approx(0.0)


def test_spread_bps_series():
    bid = np.array([100.0, 200.0])
    ask = np.array([100.1, 200.4])
    out = spread_bps_series(bid, ask)
    assert out[0] == pytest.approx(1e4 * 0.1 / 100.05, rel=1e-6)


def test_last_valid_skips_nans():
    arr = np.array([np.nan, np.nan, 1.5, 2.5, np.nan])
    assert last_valid(arr) == pytest.approx(2.5)
    assert last_valid(np.array([np.nan, np.nan])) is None


def test_cumulative_log_return_bps_zero_step_is_zero():
    h = 3
    point = np.zeros(h)
    quantiles = {level: np.zeros(h) for level in QUANTILE_LEVELS}
    qf = QuantileForecast(point=point, quantiles=quantiles)
    cum = cumulative_log_return_bps(qf)
    assert cum.horizon == 1
    assert cum.point[0] == pytest.approx(0.0)
    for level in QUANTILE_LEVELS:
        assert cum.quantile_at(level) == pytest.approx(0.0, abs=1e-9)


def test_cumulative_log_return_bps_is_monotonic_across_levels():
    h = 4
    rng = np.random.default_rng(1)
    point = rng.normal(0, 0.001, h)
    quantiles = {}
    for level in QUANTILE_LEVELS:
        # widen away from point as level moves away from 0.5, deterministic per-step std
        z = level - 0.5
        quantiles[level] = point + z * 0.01
    qf = QuantileForecast(point=point, quantiles=quantiles)
    cum = cumulative_log_return_bps(qf)
    values = [cum.quantile_at(level) for level in QUANTILE_LEVELS]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))


def test_cumulative_log_return_bps_point_matches_compounded_sum():
    h = 3
    point = np.array([0.001, -0.0005, 0.0002])
    quantiles = {level: point.copy() for level in QUANTILE_LEVELS}  # degenerate: no spread
    qf = QuantileForecast(point=point, quantiles=quantiles)
    cum = cumulative_log_return_bps(qf)
    expected = 1e4 * (math.exp(point.sum()) - 1.0)
    assert cum.point[0] == pytest.approx(expected, rel=1e-6)


def test_to_bps_scales_by_1e4():
    qf = QuantileForecast(point=np.array([0.01]), quantiles={level: np.array([0.01]) for level in QUANTILE_LEVELS})
    scaled = to_bps(qf)
    assert scaled.point[0] == pytest.approx(100.0)


def test_exp_levels_inverts_log():
    qf = QuantileForecast(point=np.array([math.log(50.0)]), quantiles={level: np.array([math.log(50.0)]) for level in QUANTILE_LEVELS})
    levels = exp_levels(qf)
    assert levels.point[0] == pytest.approx(50.0)
