from __future__ import annotations

import pytest

from jevtrader.core.types import AssetClass, Instrument
from jevtrader.risk.sizing import (
    fixed_fractional_qty,
    kelly_fraction,
    kelly_from_jev,
    lot_round,
    vol_target_qty,
)


def test_fixed_fractional_qty_basic():
    # risk 1% of $10,000 = $100; stop is $2 away -> 50 units
    qty = fixed_fractional_qty(equity=10_000, risk_pct=0.01, entry_price=100, stop_price=98)
    assert qty == pytest.approx(50)


def test_fixed_fractional_qty_degenerate_inputs():
    assert fixed_fractional_qty(0, 0.01, 100, 98) == 0.0
    assert fixed_fractional_qty(10_000, 0.01, 100, 100) == 0.0  # no stop distance


def test_vol_target_qty_scales_with_target_over_asset_vol():
    # target 10% annual vol, asset realizes 20% annual vol -> half the notional
    qty = vol_target_qty(equity=10_000, target_annual_vol=0.10, price=100, asset_annual_vol=0.20)
    assert qty == pytest.approx(50)  # notional 5000 / price 100


def test_vol_target_qty_respects_max_leverage():
    qty = vol_target_qty(equity=10_000, target_annual_vol=1.0, price=100, asset_annual_vol=0.10, max_leverage=1.0)
    assert qty * 100 <= 10_000 + 1e-6


def test_kelly_fraction_zero_when_no_edge():
    assert kelly_fraction(p_win=0.4, payoff_ratio=1.0) == 0.0  # 40% win, 1:1 payoff -> negative edge


def test_kelly_fraction_positive_edge_capped():
    f = kelly_fraction(p_win=0.6, payoff_ratio=1.0, fraction=0.25, cap_pct_equity=0.05)
    assert 0 < f <= 0.05


def test_kelly_from_jev_zero_when_edge_leq_cost():
    # edge = 0.5*40 - 0.3*40 - 10 = 20-12-10 = -2 <= 0
    f = kelly_from_jev(p_up=0.5, p_down=0.3, up_move_bps=40, down_move_bps=40, cost_bps=10)
    assert f == 0.0


def test_kelly_from_jev_zero_when_edge_exactly_costs():
    f = kelly_from_jev(p_up=0.5, p_down=0.5, up_move_bps=20, down_move_bps=20, cost_bps=0)
    assert f == 0.0


def test_kelly_from_jev_positive_and_capped():
    f = kelly_from_jev(
        p_up=0.65, p_down=0.15, up_move_bps=100, down_move_bps=80, cost_bps=15, fraction=0.25, cap_pct_equity=0.05
    )
    assert 0 < f <= 0.05


def test_kelly_from_jev_hard_cap_binds_on_strong_edge():
    f = kelly_from_jev(
        p_up=0.95, p_down=0.02, up_move_bps=500, down_move_bps=50, cost_bps=5, fraction=1.0, cap_pct_equity=0.05
    )
    assert f == pytest.approx(0.05)


def test_lot_round_equity_floors_to_whole_shares():
    inst = Instrument("AAPL", AssetClass.EQUITY, lot_size=1.0)
    assert lot_round(10.7, inst) == 10.0
    assert lot_round(-10.7, inst) == -10.0


def test_lot_round_crypto_fractional_lot():
    inst = Instrument("BTC/USD", AssetClass.CRYPTO, lot_size=1e-6)
    rounded = lot_round(0.123456789, inst)
    assert rounded == pytest.approx(0.123456, abs=1e-9)


def test_lot_round_below_min_notional_returns_zero():
    inst = Instrument("AAPL", AssetClass.EQUITY, lot_size=1.0, min_notional=100.0)
    # rounds to 1 share; at $50/share that's $50 notional, below the $100 minimum -> 0
    assert lot_round(1.5, inst, min_notional=100.0, price=50.0) == 0.0
