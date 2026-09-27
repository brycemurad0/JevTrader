from __future__ import annotations

from pathlib import Path

from jevtrader.risk.limits import RiskLimits, TradingHours

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "risk.yaml"


def test_defaults_are_conservative():
    limits = RiskLimits()
    assert 0 < limits.max_daily_loss_pct < 0.1
    assert 0 < limits.max_drawdown_pct < 0.5
    assert limits.max_gross_exposure_pct <= 1.5
    assert limits.enforce_long_only_for_non_shortable is True
    assert isinstance(limits.trading_hours, TradingHours)


def test_loads_shipped_yaml():
    assert CONFIG_PATH.exists(), "config/risk.yaml must ship with the repo"
    limits = RiskLimits.from_yaml(CONFIG_PATH)
    assert limits.max_position_notional_usd == 5000.0
    assert limits.crypto_max_pct_equity == 0.25
    assert limits.trading_hours.tz == "America/New_York"
    assert limits.trading_hours.regular_start == "09:30"


def test_from_dict_overrides_and_nested_trading_hours():
    limits = RiskLimits.from_dict(
        {
            "max_drawdown_pct": 0.25,
            "trading_hours": {"allow_extended_hours": True, "extended_start": "05:00"},
        }
    )
    assert limits.max_drawdown_pct == 0.25
    assert limits.trading_hours.allow_extended_hours is True
    assert limits.trading_hours.extended_start == "05:00"
    # untouched fields keep their dataclass defaults
    assert limits.trading_hours.regular_start == "09:30"
    assert limits.max_position_notional_usd == RiskLimits().max_position_notional_usd


def test_to_dict_round_trips_through_from_dict():
    limits = RiskLimits(max_drawdown_pct=0.2)
    d = limits.to_dict()
    assert d["max_drawdown_pct"] == 0.2
    assert isinstance(d["trading_hours"], dict)
    rebuilt = RiskLimits.from_dict(d)
    assert rebuilt == limits
