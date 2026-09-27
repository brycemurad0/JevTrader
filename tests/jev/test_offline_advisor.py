from __future__ import annotations

import pytest
from typesafe_sdk import Choice, Noul, Score

from jevtrader.jev.advisor import OfflineJevAdvisor, OfflineWeights


def _features(**overrides):
    base = {
        "ret_bps": {"5": 5.0, "15": 10.0, "30": 15.0, "60": 20.0},
        "z_vwap": 0.0,
        "book_imbalance": 0.0,
        "trend_slope_bps": 0.0,
        "vol_bps": 10.0,
        "cost_bps": 3.0,
    }
    base.update(overrides)
    return base


def test_direction_is_deterministic():
    adv = OfflineJevAdvisor()
    features = _features()
    v1 = adv.direction("AAPL", features, "5min")
    v2 = adv.direction("AAPL", dict(features), "5min")
    assert v1.probs == v2.probs
    assert v1.top == v2.top


def test_direction_probabilities_sum_to_one():
    adv = OfflineJevAdvisor()
    v = adv.direction("AAPL", _features(), "5min")
    total = sum(v.probs["direction"].values())
    assert total == pytest.approx(1.0, abs=1e-3)
    for p in v.probs["direction"].values():
        assert 0.0 <= p <= 1.0


def test_regime_probabilities_sum_to_one_and_risk_off_is_binary():
    adv = OfflineJevAdvisor()
    v = adv.regime("AAPL", _features(vol_bps=80.0))
    assert sum(v.probs["regime"].values()) == pytest.approx(1.0, abs=1e-3)
    assert v.probs["risk_off"]["yes"] + v.probs["risk_off"]["no"] == pytest.approx(1.0, abs=1e-6)
    assert v.top["risk_off"] in ("yes", "no")


def test_positive_momentum_favors_up():
    adv = OfflineJevAdvisor()
    up = adv.direction("AAPL", _features(ret_bps={"5": 40, "15": 40, "30": 40, "60": 40}), "5min")
    down = adv.direction("AAPL", _features(ret_bps={"5": -40, "15": -40, "30": -40, "60": -40}), "5min")
    assert up.probs["direction"]["up"] > down.probs["direction"]["up"]
    assert down.probs["direction"]["down"] > up.probs["direction"]["down"]


def test_high_cost_relative_to_vol_increases_flat_probability():
    adv = OfflineJevAdvisor()
    cheap = adv.direction("AAPL", _features(cost_bps=1.0, vol_bps=50.0), "5min")
    expensive = adv.direction("AAPL", _features(cost_bps=100.0, vol_bps=5.0), "5min")
    assert expensive.probs["direction"]["flat"] > cheap.probs["direction"]["flat"]


def test_weights_are_configurable_and_change_output():
    features = _features(ret_bps={"5": 40, "15": 40, "30": 40, "60": 40})
    default_adv = OfflineJevAdvisor()
    no_momentum_adv = OfflineJevAdvisor(OfflineWeights(momentum=0.0, mean_reversion=-0.6, imbalance=0.5, trend=0.8))
    v_default = default_adv.direction("AAPL", features, "5min")
    v_no_momentum = no_momentum_adv.direction("AAPL", features, "5min")
    assert v_default.probs["direction"]["up"] != v_no_momentum.probs["direction"]["up"]


def test_model_name_is_clearly_not_jev():
    adv = OfflineJevAdvisor()
    v = adv.direction("AAPL", _features(), "5min")
    assert "offline" in adv.model
    assert v.model == adv.model


def test_generic_ask_handles_choice_score_noul_and_sums_to_one():
    adv = OfflineJevAdvisor()
    questions = {
        "direction": Choice(criteria={"up": None, "down": None, "flat": None}),
        "entry_quality": Score(criteria=["poor", "weak", "fair", "good", "excellent"]),
        "risk_off": Noul(criteria={"true": None, "false": None}),
    }
    v = adv.ask(_features(), questions)
    assert set(v.probs.keys()) == {"direction", "entry_quality", "risk_off"}
    assert sum(v.probs["direction"].values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(v.probs["entry_quality"].values()) == pytest.approx(1.0, abs=1e-3)
    assert v.probs["risk_off"]["yes"] + v.probs["risk_off"]["no"] == pytest.approx(1.0, abs=1e-6)
    assert v.top["entry_quality"] in {"0", "1", "2", "3", "4"}


def test_generic_ask_is_deterministic():
    adv = OfflineJevAdvisor()
    questions = {"risk_off": Noul(criteria={"true": None, "false": None})}
    features = _features(z_vwap=1.5)
    v1 = adv.ask(features, questions)
    v2 = adv.ask(dict(features), questions)
    assert v1.probs == v2.probs
