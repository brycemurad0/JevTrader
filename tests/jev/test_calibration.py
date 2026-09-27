from __future__ import annotations

import math

import pytest

from jevtrader.jev.calibration import calibrate_binary, calibrate_direction


def _entry(p_up, fwd_ret_bps, cost_bps=2.0, p_down=None):
    p_down = p_down if p_down is not None else max(0.0, 1.0 - p_up - 0.1)
    return {
        "answers": {"probs": {"direction": {"up": p_up, "down": p_down, "flat": max(0.0, 1.0 - p_up - p_down)}}},
        "fwd_ret_bps": fwd_ret_bps,
        "state": {"cost_bps": cost_bps},
    }


def test_calibrate_direction_perfect_calibration_zero_brier():
    # p=1.0 always followed by a real up move; p=0.0 never is -> brier == 0, log_loss ~ 0
    entries = [_entry(1.0, 10.0, cost_bps=1.0) for _ in range(5)] + [_entry(0.0, -10.0, cost_bps=1.0) for _ in range(5)]
    result = calibrate_direction(entries, label="up")
    assert result["n"] == 10
    assert result["brier"] == pytest.approx(0.0, abs=1e-6)
    assert result["log_loss"] < 0.01


def test_calibrate_direction_hand_computed_brier():
    # two entries: p=0.8 outcome True (fwd 5bps > cost 2bps), p=0.3 outcome False (fwd -1bps, not < -cost)
    entries = [_entry(0.8, 5.0, cost_bps=2.0), _entry(0.3, -1.0, cost_bps=2.0)]
    result = calibrate_direction(entries, label="up")
    expected_brier = ((0.8 - 1.0) ** 2 + (0.3 - 0.0) ** 2) / 2
    assert result["brier"] == pytest.approx(expected_brier, abs=1e-6)


def test_calibrate_direction_skips_entries_without_forward_return():
    entries = [_entry(0.8, 5.0), _entry(0.6, None)]
    result = calibrate_direction(entries, label="up")
    assert result["n"] == 1


def test_calibrate_direction_down_label_uses_negative_cost_threshold():
    # a -5bps move with a 2bps cost clears the down threshold (-5 < -2)
    entries = [_entry(0.2, -5.0, cost_bps=2.0, p_down=0.7)]
    result = calibrate_direction(entries, label="down")
    assert result["n"] == 1
    # p(down)=0.7, outcome True -> brier = (0.7-1)^2
    assert result["brier"] == pytest.approx(0.09, abs=1e-6)


def test_calibrate_direction_rejects_bad_label():
    with pytest.raises(ValueError):
        calibrate_direction([], label="sideways")


def test_reliability_curve_bins_and_ece():
    # 10 entries all with p=0.9 and all actually "up": one bin, mean_predicted=0.9, empirical=1.0
    entries = [_entry(0.9, 10.0, cost_bps=1.0) for _ in range(10)]
    result = calibrate_direction(entries, label="up", n_bins=10)
    nonzero_bins = [b for b in result["reliability"] if b["n"] > 0]
    assert len(nonzero_bins) == 1
    bucket = nonzero_bins[0]
    assert bucket["n"] == 10
    assert bucket["mean_predicted"] == pytest.approx(0.9)
    assert bucket["empirical_rate"] == pytest.approx(1.0)
    assert result["ece"] == pytest.approx(0.1, abs=1e-3)


def test_hit_rate_by_confidence_high_confidence_correct_predictions():
    # p=0.95 (very confident "up"), always actually up -> should land in the top confidence bucket with hit_rate 1.0
    entries = [_entry(0.95, 20.0, cost_bps=1.0) for _ in range(4)]
    result = calibrate_direction(entries, label="up")
    top_bucket = result["hit_rate_by_confidence"][-1]
    assert top_bucket["n"] == 4
    assert top_bucket["hit_rate"] == pytest.approx(1.0)


def test_edge_after_costs_matches_hand_calculation():
    entries = [
        _entry(0.9, 10.0, cost_bps=3.0),
        _entry(0.9, 6.0, cost_bps=3.0),
        _entry(0.4, 20.0, cost_bps=3.0),  # below threshold, excluded
    ]
    result = calibrate_direction(entries, label="up", threshold=0.5)
    edge = result["edge_after_costs"]
    assert edge["n"] == 2
    assert edge["avg_fwd_ret_bps"] == pytest.approx((10.0 + 6.0) / 2)
    assert edge["avg_cost_bps"] == pytest.approx(3.0)
    assert edge["edge_after_costs_bps"] == pytest.approx((10.0 + 6.0) / 2 - 3.0)


def test_calibrate_binary_generic_with_custom_outcome_fn():
    entries = [
        {"answers": {"probs": {"risk_off": {"yes": 0.9, "no": 0.1}}}, "extra": {"was_volatile": True}},
        {"answers": {"probs": {"risk_off": {"yes": 0.1, "no": 0.9}}}, "extra": {"was_volatile": False}},
    ]
    result = calibrate_binary(
        entries,
        question="risk_off",
        label="yes",
        outcome_fn=lambda e: e["extra"]["was_volatile"],
    )
    assert result["n"] == 2
    assert result["brier"] == pytest.approx(((0.9 - 1) ** 2 + (0.1 - 0) ** 2) / 2)


def test_markdown_summary_is_nonempty_and_mentions_question():
    entries = [_entry(0.8, 10.0, cost_bps=1.0)]
    result = calibrate_direction(entries, label="up")
    assert "direction" in result["markdown"]
    assert "Brier" in result["markdown"]


def test_empty_entries_do_not_crash():
    result = calibrate_direction([], label="up")
    assert result["n"] == 0
    assert result["brier"] is None
    assert result["edge_after_costs"]["n"] == 0
    assert isinstance(result["markdown"], str)
