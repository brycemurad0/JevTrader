from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import pytest

from jevtrader.core.fees import BpsFees
from jevtrader.core.interfaces import JevView
from jevtrader.core.types import AssetClass, Instrument
from jevtrader.rebalance.policy import (
    RebalancePolicy,
    apply_jev_tilt,
    band_breach,
    calendar_due,
    decide,
    estimated_cost,
    expected_te_reduction,
)


def _instruments(symbols):
    return {s: Instrument.infer(s) for s in symbols}


class _StubJev:
    """Minimal JevAdvisorProtocol stand-in: returns a fixed p(risk_off) per symbol."""

    def __init__(self, p_risk_off: Mapping[str, float]):
        self._p = p_risk_off

    def direction(self, symbol, features, horizon):
        return None

    def regime(self, symbol, features):
        p = self._p.get(symbol)
        if p is None:
            return None
        return JevView(
            probs={"regime": {"volatile_chop": 1.0}, "risk_off": {"yes": p, "no": 1 - p}},
            top={"regime": "volatile_chop", "risk_off": "yes" if p > 0.5 else "no"},
            confidence={"risk_off": 0.9},
            latency_ms=10.0,
            late=False,
        )

    def ask(self, state, questions):
        return None


# ----------------------------------------------------------------------------- band_breach


def test_band_breach_absolute():
    policy = RebalancePolicy(abs_band=0.05, rel_band=0.90)
    current = {"SPY": 0.50, "TLT": 0.20}
    target = {"SPY": 0.40, "TLT": 0.22}  # SPY drifted 0.10 (breach), TLT drifted 0.02 (no breach)
    breach = band_breach(current, target, policy)
    assert breach["SPY"] is True
    assert breach["TLT"] is False


def test_band_breach_relative_for_small_targets():
    policy = RebalancePolicy(abs_band=0.90, rel_band=0.25)
    current = {"GLD": 0.02}
    target = {"GLD": 0.05}  # drift 0.03, relative to target 0.05 = 60% > 25% band
    breach = band_breach(current, target, policy)
    assert breach["GLD"] is True


def test_band_breach_within_tolerance():
    policy = RebalancePolicy(abs_band=0.05, rel_band=0.25)
    current = {"SPY": 0.40}
    target = {"SPY": 0.41}
    breach = band_breach(current, target, policy)
    assert breach["SPY"] is False


# ----------------------------------------------------------------------------- calendar


def test_calendar_due_when_never_rebalanced():
    assert calendar_due(None, pd.Timestamp.now(tz="UTC"), "1w") is True


def test_calendar_due_respects_frequency():
    now = pd.Timestamp("2024-01-10", tz="UTC")
    last = pd.Timestamp("2024-01-05", tz="UTC")
    assert calendar_due(last, now, "1w") is False  # only 5 days
    assert calendar_due(last, now, "1d") is True


# ----------------------------------------------------------------------------- cost / benefit


def test_expected_te_reduction_zero_when_at_target():
    cov = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0002]], index=["SPY", "TLT"], columns=["SPY", "TLT"])
    te = expected_te_reduction({"SPY": 0.5, "TLT": 0.5}, {"SPY": 0.5, "TLT": 0.5}, cov)
    assert te == pytest.approx(0.0)


def test_expected_te_reduction_positive_when_drifted():
    cov = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0002]], index=["SPY", "TLT"], columns=["SPY", "TLT"])
    te = expected_te_reduction({"SPY": 0.7, "TLT": 0.3}, {"SPY": 0.5, "TLT": 0.5}, cov)
    assert te > 0.0


def test_estimated_cost_scales_with_notional():
    fee_model = BpsFees(maker_bps=0.0, taker_bps=10.0)
    instruments = _instruments(["AAPL"])
    cost_small = estimated_cost({"AAPL": 1_000.0}, instruments, fee_model, spread_bps=10.0, prices={"AAPL": 100.0})
    cost_large = estimated_cost({"AAPL": 10_000.0}, instruments, fee_model, spread_bps=10.0, prices={"AAPL": 100.0})
    assert cost_large == pytest.approx(cost_small * 10)


# ----------------------------------------------------------------------------- jev tilt


def test_apply_jev_tilt_none_jev_is_noop():
    targets = {"BTC/USD": 0.30, "SPY": 0.70}
    tilted, log = apply_jev_tilt(targets, {"BTC/USD": {}}, jev=None, policy=RebalancePolicy())
    assert tilted == targets
    assert log == {}


def test_apply_jev_tilt_reduces_risky_weight_when_risk_off():
    policy = RebalancePolicy(tilt_k=0.5, tilt_max=0.20)
    targets = {"BTC/USD": 0.30, "SPY": 0.70}
    jev = _StubJev({"BTC/USD": 0.8})  # high risk-off probability
    tilted, log = apply_jev_tilt(targets, {"BTC/USD": {}}, jev=jev, policy=policy)
    assert tilted["BTC/USD"] < targets["BTC/USD"]
    assert tilted["SPY"] == targets["SPY"]  # untouched, not in risky_symbols
    assert "BTC/USD" in log


def test_apply_jev_tilt_is_capped():
    policy = RebalancePolicy(tilt_k=5.0, tilt_max=0.10)  # huge k, tight cap
    targets = {"BTC/USD": 0.30}
    jev = _StubJev({"BTC/USD": 1.0})  # certain risk-off
    tilted, log = apply_jev_tilt(targets, {"BTC/USD": {}}, jev=jev, policy=policy)
    min_allowed = targets["BTC/USD"] * (1 - policy.tilt_max)
    assert tilted["BTC/USD"] == pytest.approx(min_allowed, rel=1e-6)
    assert log["BTC/USD"] == pytest.approx(1 - policy.tilt_max)


def test_apply_jev_tilt_never_increases_weight():
    policy = RebalancePolicy(tilt_k=0.5, tilt_max=0.20)
    targets = {"BTC/USD": 0.30}
    jev = _StubJev({"BTC/USD": 0.0})  # no risk-off signal at all
    tilted, log = apply_jev_tilt(targets, {"BTC/USD": {}}, jev=jev, policy=policy)
    assert tilted["BTC/USD"] <= targets["BTC/USD"]


# ----------------------------------------------------------------------------- decide()


@pytest.fixture
def small_cov():
    return pd.DataFrame(
        [[0.0004, 0.0001], [0.0001, 0.0009]],
        index=["SPY", "BTC/USD"],
        columns=["SPY", "BTC/USD"],
    )


def test_decide_no_trade_within_band(small_cov):
    policy = RebalancePolicy(abs_band=0.10, rel_band=0.90)
    current = {"SPY": 0.60, "BTC/USD": 0.40}
    target = {"SPY": 0.62, "BTC/USD": 0.38}  # tiny drift, within band
    decision = decide(
        policy, current, target, equity=100_000, cov=small_cov,
        prices={"SPY": 400.0, "BTC/USD": 60_000.0}, instruments=_instruments(["SPY", "BTC/USD"]),
        fee_model=BpsFees(0, 5), spread_bps=5.0,
    )
    assert decision.should_trade is False
    assert decision.trade_weights == {}


def test_decide_no_trade_when_cost_exceeds_benefit(small_cov):
    policy = RebalancePolicy(abs_band=0.0001, rel_band=0.0001, min_trade_notional=1.0)
    current = {"SPY": 0.60, "BTC/USD": 0.40}
    target = {"SPY": 0.601, "BTC/USD": 0.399}  # tiny drift -> tiny benefit
    # Expensive fees dwarf the tiny expected benefit.
    decision = decide(
        policy, current, target, equity=100_000, cov=small_cov,
        prices={"SPY": 400.0, "BTC/USD": 60_000.0}, instruments=_instruments(["SPY", "BTC/USD"]),
        fee_model=BpsFees(0, 500.0), spread_bps=500.0,
    )
    assert decision.should_trade is False
    assert decision.est_cost_usd >= decision.est_benefit_usd


def test_decide_trades_when_benefit_exceeds_cost(small_cov):
    policy = RebalancePolicy(abs_band=0.03, rel_band=0.05, min_trade_notional=1.0)
    current = {"SPY": 0.80, "BTC/USD": 0.20}
    target = {"SPY": 0.60, "BTC/USD": 0.40}  # large drift -> large benefit
    decision = decide(
        policy, current, target, equity=100_000, cov=small_cov,
        prices={"SPY": 400.0, "BTC/USD": 60_000.0}, instruments=_instruments(["SPY", "BTC/USD"]),
        fee_model=BpsFees(0, 5.0), spread_bps=5.0,
    )
    assert decision.should_trade is True
    assert "SPY" in decision.trade_weights
    assert "BTC/USD" in decision.trade_weights
    assert decision.est_benefit_usd > decision.est_cost_usd


def test_decide_skips_trades_below_min_notional(small_cov):
    policy = RebalancePolicy(abs_band=0.001, rel_band=0.001, min_trade_notional=1_000_000.0)
    current = {"SPY": 0.60, "BTC/USD": 0.40}
    target = {"SPY": 0.59, "BTC/USD": 0.41}
    decision = decide(
        policy, current, target, equity=100_000, cov=small_cov,
        prices={"SPY": 400.0, "BTC/USD": 60_000.0}, instruments=_instruments(["SPY", "BTC/USD"]),
        fee_model=BpsFees(0, 5.0), spread_bps=5.0,
    )
    assert decision.should_trade is False


def test_decide_applies_and_logs_jev_tilt(small_cov):
    policy = RebalancePolicy(abs_band=0.03, rel_band=0.05, min_trade_notional=1.0, tilt_k=0.8, tilt_max=0.3)
    current = {"SPY": 0.80, "BTC/USD": 0.20}
    target = {"SPY": 0.60, "BTC/USD": 0.40}
    jev = _StubJev({"BTC/USD": 0.9})
    decision = decide(
        policy, current, target, equity=100_000, cov=small_cov,
        prices={"SPY": 400.0, "BTC/USD": 60_000.0}, instruments=_instruments(["SPY", "BTC/USD"]),
        fee_model=BpsFees(0, 5.0), spread_bps=5.0,
        jev=jev, risky_symbols={"BTC/USD": {}},
    )
    assert decision.tilt_log.get("BTC/USD") is not None
    assert decision.target_weights["BTC/USD"] < target["BTC/USD"]
