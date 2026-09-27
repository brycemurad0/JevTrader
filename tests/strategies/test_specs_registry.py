"""Every strategy module registers, has a complete spec with honest notes, and its params live in
the spec (not hidden in code)."""

from jevtrader.core.registry import all_strategies
from jevtrader.core.strategy import Strategy, StrategySpec

EXPECTED = {
    "as_market_maker",
    "imbalance_alpha",
    "vpin_monitor",
    "zscore_reversion",
    "range_breakout",
    "trend_follow",
    "vol_squeeze",
    "time_of_day",
    "pairs_kalman",
    "lead_lag",
    "zscore_reversion_jev",
    "trend_follow_jev",
    "range_breakout_jev",
    "meta_allocator",
}


def test_all_library_strategies_are_registered():
    reg = all_strategies()
    missing = EXPECTED - set(reg)
    assert not missing, missing


def test_specs_are_complete_and_honest():
    for name, cls in all_strategies().items():
        if name not in EXPECTED:
            continue
        spec = cls.spec
        assert isinstance(spec, StrategySpec)
        assert spec.name == name
        assert spec.description and len(spec.description) > 20
        assert spec.asset_classes and set(spec.asset_classes) <= {"equity", "crypto"}
        assert spec.frequency
        assert spec.style
        assert isinstance(spec.default_params, dict)
        assert len(spec.notes) > 60, f"{name}: notes must state when it fails / fee sensitivity"
        assert issubclass(cls, Strategy)


def test_params_merge_defaults_with_overrides():
    from jevtrader.strategies.vwap_reversion import ZScoreReversion

    s = ZScoreReversion(["SPY"], {"entry_z": 3.0})
    assert s.params["entry_z"] == 3.0
    assert s.params["window"] == ZScoreReversion.spec.default_params["window"]
    assert s.id == "zscore_reversion:SPY"


def test_jev_variants_default_to_gated():
    from jevtrader.strategies.jev_gated import TrendFollowJev, ZScoreReversionJev

    assert ZScoreReversionJev.spec.default_params["jev_gate"] is True
    assert TrendFollowJev.spec.uses_jev is True
    assert "ReplayJevAdvisor" in ZScoreReversionJev.spec.notes
