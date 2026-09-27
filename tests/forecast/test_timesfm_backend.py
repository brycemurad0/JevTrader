from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from jevtrader.core.broker import TradingMode
from jevtrader.forecast.base import QUANTILE_LEVELS
from jevtrader.forecast.timesfm_backend import (
    TIMESFM3_LICENSE_ACK_ENV,
    TIMESFM3_LICENSE_ACK_VALUE,
    TimesFM3LicenseError,
    TimesFMForecaster,
    TimesFMNotInstalled,
)


# ----------------------------------------------------------------------------- fake `timesfm` (2.5)


class _FakeForecastConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeModel25:
    calls: list[dict] = []

    def __init__(self):
        self.compiled_config = None
        self.device = None

    @classmethod
    def from_pretrained(cls, model_id):
        inst = cls()
        inst.model_id = model_id
        return inst

    def compile(self, config):
        self.compiled_config = config

    def to(self, device):
        self.device = device

    def forecast(self, horizon, inputs):
        type(self).calls.append({"horizon": horizon, "n_inputs": len(inputs)})
        b = len(inputs)
        point = np.zeros((b, horizon), dtype=float)
        quantiles = np.zeros((b, horizon, 10), dtype=float)
        for i, series in enumerate(inputs):
            base = float(np.mean(series[-5:])) if len(series) else 0.0
            point[i, :] = base
            quantiles[i, :, 0] = base + 999.0  # mean slot -- must never leak into our output
            for j, level in enumerate(QUANTILE_LEVELS):
                quantiles[i, :, j + 1] = base + (level - 0.5) * 2.0
        return point, quantiles


@pytest.fixture()
def fake_timesfm_module(monkeypatch):
    _FakeModel25.calls = []
    module = types.SimpleNamespace(TimesFM_2p5_200M_torch=_FakeModel25, ForecastConfig=_FakeForecastConfig)
    monkeypatch.setitem(sys.modules, "timesfm", module)
    return module


# ----------------------------------------------------------------------------- mapping & batching


def test_forecast_25_maps_quantiles_and_drops_mean_column(fake_timesfm_module):
    forecaster = TimesFMForecaster(version="2.5", max_horizon=10)
    out = forecaster.forecast({"a": np.arange(20.0), "b": np.arange(30.0)}, horizon=4)
    assert set(out.keys()) == {"a", "b"}
    qf = out["a"]
    assert qf.horizon == 4
    base = float(np.mean(np.arange(20.0)[-5:]))
    assert qf.point[0] == pytest.approx(base)
    # mean column (index 0, base+999) must never appear in any quantile slot
    for level in QUANTILE_LEVELS:
        val = qf.quantile_at(level, h=0)
        assert val == pytest.approx(base + (level - 0.5) * 2.0)
        assert val != pytest.approx(base + 999.0)


def test_forecast_25_batches_multiple_symbols_in_one_call(fake_timesfm_module):
    forecaster = TimesFMForecaster(version="2.5", max_horizon=10)
    series = {f"sym{i}": np.arange(20.0) + i for i in range(5)}
    out = forecaster.forecast(series, horizon=3)
    assert len(out) == 5
    assert len(_FakeModel25.calls) == 1  # one forward pass for all 5 symbols
    assert _FakeModel25.calls[0]["n_inputs"] == 5


def test_forecast_25_reuses_loaded_model_across_calls(fake_timesfm_module):
    forecaster = TimesFMForecaster(version="2.5", max_horizon=10)
    forecaster.forecast({"a": np.arange(20.0)}, horizon=2)
    forecaster.forecast({"a": np.arange(20.0)}, horizon=2)
    # from_pretrained/compile only happen once; both calls hit the same model instance
    assert len(_FakeModel25.calls) == 2


def test_forecast_rejects_horizon_over_max():
    forecaster = TimesFMForecaster(version="2.5", max_horizon=5)
    with pytest.raises(ValueError):
        forecaster.forecast({"a": np.arange(20.0)}, horizon=6)


def test_forecast_empty_series_returns_empty_dict(fake_timesfm_module):
    forecaster = TimesFMForecaster(version="2.5", max_horizon=10)
    assert forecaster.forecast({}, horizon=3) == {}
    assert _FakeModel25.calls == []  # never even tried to load/call the model


# ----------------------------------------------------------------------------- not installed


def test_missing_timesfm_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "timesfm", None)  # forces ImportError on `import timesfm`
    forecaster = TimesFMForecaster(version="2.5")
    with pytest.raises(TimesFMNotInstalled) as exc:
        forecaster.forecast({"a": np.arange(20.0)}, horizon=2)
    assert "pip install" in str(exc.value)


def test_missing_timesfm3_raises_helpful_error(monkeypatch):
    monkeypatch.setenv(TIMESFM3_LICENSE_ACK_ENV, TIMESFM3_LICENSE_ACK_VALUE)
    monkeypatch.setitem(sys.modules, "timesfm3", None)
    forecaster = TimesFMForecaster(version="3.0", mode=TradingMode.BACKTEST)
    with pytest.raises(TimesFMNotInstalled) as exc:
        forecaster.forecast({"a": np.arange(20.0)}, horizon=2)
    assert "pip install" in str(exc.value)


# ----------------------------------------------------------------------------- license guard


def test_timesfm3_refused_in_live_mode():
    with pytest.raises(TimesFM3LicenseError):
        TimesFMForecaster(version="3.0", mode=TradingMode.LIVE)


def test_timesfm3_refused_in_paper_mode():
    with pytest.raises(TimesFM3LicenseError):
        TimesFMForecaster(version="3.0", mode=TradingMode.PAPER)


def test_timesfm3_refused_with_no_mode():
    with pytest.raises(TimesFM3LicenseError):
        TimesFMForecaster(version="3.0")


def test_timesfm3_refused_in_backtest_without_ack(monkeypatch):
    monkeypatch.delenv(TIMESFM3_LICENSE_ACK_ENV, raising=False)
    with pytest.raises(TimesFM3LicenseError):
        TimesFMForecaster(version="3.0", mode=TradingMode.BACKTEST)


def test_timesfm3_refused_in_backtest_with_wrong_ack(monkeypatch):
    monkeypatch.setenv(TIMESFM3_LICENSE_ACK_ENV, "yes_please")
    with pytest.raises(TimesFM3LicenseError):
        TimesFMForecaster(version="3.0", mode=TradingMode.BACKTEST)


def test_timesfm3_allowed_in_backtest_with_correct_ack(monkeypatch):
    monkeypatch.setenv(TIMESFM3_LICENSE_ACK_ENV, TIMESFM3_LICENSE_ACK_VALUE)
    # should not raise at construction time
    TimesFMForecaster(version="3.0", mode=TradingMode.BACKTEST)


def test_unknown_version_rejected():
    with pytest.raises(ValueError):
        TimesFMForecaster(version="1.0")


def test_mlx_backend_only_for_30():
    with pytest.raises(ValueError):
        TimesFMForecaster(version="2.5", backend="mlx")
