from __future__ import annotations

import pytest
from typesafe_sdk import Choice

from jevtrader.forecast.advisor import ForecastAdvisor


def test_direction_returns_none_without_forecast_features():
    advisor = ForecastAdvisor()
    assert advisor.direction("AAPL", {}, "5min") is None


def test_direction_probabilities_sum_to_one():
    advisor = ForecastAdvisor()
    for p_up, p_down in [(0.6, 0.1), (0.1, 0.6), (0.4, 0.4), (0.9, 0.9), (0.0, 0.0), (1.0, 0.0)]:
        view = advisor.direction("AAPL", {"tfm_p_up_gt_cost": p_up, "tfm_p_down_gt_cost": p_down}, "5min")
        assert view is not None
        total = sum(view.probs["direction"].values())
        assert total == pytest.approx(1.0, abs=1e-9)
        for p in view.probs["direction"].values():
            assert p >= 0.0


def test_direction_picks_top_label_consistently():
    advisor = ForecastAdvisor()
    view = advisor.direction("AAPL", {"tfm_p_up_gt_cost": 0.8, "tfm_p_down_gt_cost": 0.05}, "5min")
    assert view.top["direction"] == "up"
    assert view.confidence["direction"] == max(view.probs["direction"].values())


def test_regime_returns_none_without_vol_ratio():
    advisor = ForecastAdvisor()
    assert advisor.regime("AAPL", {}) is None


def test_regime_labels_volatile_chop_when_vol_forecast_much_higher_than_realized():
    advisor = ForecastAdvisor()
    view = advisor.regime("AAPL", {"tfm_vol_ratio": 2.5})
    assert view is not None
    assert view.top["regime"] == "volatile_chop"
    assert sum(view.probs["regime"].values()) == pytest.approx(1.0, abs=1e-3)  # 4dp-rounded components
    assert set(view.probs["regime"].keys()) == {"trending_up", "trending_down", "mean_reverting", "volatile_chop", "quiet"}


def test_regime_labels_quiet_when_vol_forecast_much_lower_than_realized():
    advisor = ForecastAdvisor()
    view = advisor.regime("AAPL", {"tfm_vol_ratio": 0.3})
    assert view.top["regime"] == "quiet"


def test_regime_risk_off_is_a_valid_noul():
    advisor = ForecastAdvisor()
    view = advisor.regime("AAPL", {"tfm_vol_ratio": 1.0})
    assert set(view.probs["risk_off"].keys()) == {"yes", "no"}
    assert view.probs["risk_off"]["yes"] + view.probs["risk_off"]["no"] == pytest.approx(1.0)


def test_ask_direction_shaped_question():
    advisor = ForecastAdvisor()
    questions = {
        "direction": Choice(
            instructions="up/down/flat?",
            criteria={"up": "goes up", "down": "goes down", "flat": "stays flat"},
        )
    }
    state = {"tfm_p_up_gt_cost": 0.7, "tfm_p_down_gt_cost": 0.1}
    view = advisor.ask(state, questions)
    assert view is not None
    assert set(view.probs["direction"].keys()) == {"up", "down", "flat"}
    assert sum(view.probs["direction"].values()) == pytest.approx(1.0, abs=1e-9)


def test_ask_unknown_question_returns_none():
    advisor = ForecastAdvisor()
    questions = {
        "regime": Choice(instructions="what regime?", criteria={"a": "a", "b": "b"}),
    }
    assert advisor.ask({"tfm_p_up_gt_cost": 0.5, "tfm_p_down_gt_cost": 0.1}, questions) is None
