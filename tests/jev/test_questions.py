from __future__ import annotations

from typesafe_sdk import Choice, Noul, Score

from jevtrader.jev.questions import (
    direction_questions,
    entry_quality_questions,
    rebalance_tilt_questions,
    regime_questions,
)


def test_direction_labels_match_protocol():
    qs = direction_questions("5min", 10.0)
    assert "direction" in qs
    direction = qs["direction"]
    assert isinstance(direction, Choice)
    assert set(direction.criteria.keys()) == {"up", "down", "flat"}
    # cost is surfaced explicitly to the model
    assert "10.0" in "".join(str(v) for v in direction.criteria.values())


def test_direction_conviction_is_optional():
    with_conv = direction_questions("1h", 5.0, include_conviction=True)
    without_conv = direction_questions("1h", 5.0, include_conviction=False)
    assert "conviction" in with_conv and isinstance(with_conv["conviction"], Score)
    assert "conviction" not in without_conv
    assert "direction" in without_conv


def test_regime_labels_match_protocol():
    qs = regime_questions()
    assert set(qs.keys()) == {"regime", "risk_off"}
    assert isinstance(qs["regime"], Choice)
    assert set(qs["regime"].criteria.keys()) == {
        "trending_up",
        "trending_down",
        "mean_reverting",
        "volatile_chop",
        "quiet",
    }
    assert isinstance(qs["risk_off"], Noul)
    assert set(qs["risk_off"].criteria.keys()) == {"true", "false"}


def test_entry_quality_is_0_to_4_rubric():
    qs = entry_quality_questions()
    assert list(qs.keys()) == ["entry_quality"]
    score = qs["entry_quality"]
    assert isinstance(score, Score)
    assert len(score.criteria) == 5  # scores 0..4


def test_rebalance_tilt_one_question_per_asset():
    assets = ["BTC/USD", "ETH/USD", "SOL/USD"]
    qs = rebalance_tilt_questions(assets)
    assert set(qs.keys()) == {f"tilt:{a}" for a in assets}
    for name, q in qs.items():
        assert isinstance(q, Score)
        assert len(q.criteria) == 5


def test_rebalance_tilt_requires_at_least_one_asset():
    import pytest

    with pytest.raises(ValueError):
        rebalance_tilt_questions([])


def test_all_builders_produce_client_ready_questions():
    """Every value returned must be usable directly as a `typesafe_sdk` question object."""
    for qs in (
        direction_questions("5min", 8.0),
        regime_questions(),
        entry_quality_questions(),
        rebalance_tilt_questions(["AAPL"]),
    ):
        for name, q in qs.items():
            assert isinstance(q, (Choice, Score, Noul)), f"{name} is not a typesafe_sdk question object"
