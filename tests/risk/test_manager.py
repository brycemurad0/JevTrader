from __future__ import annotations

import pandas as pd
import pytest

from conftest import AFTER_HOURS, NOW, loose_limits, make_account, make_order, make_position, make_quote
from jevtrader.core.types import Fill, Liquidity, OrderType, Side
from jevtrader.risk.limits import TradingHours
from jevtrader.risk.manager import RiskManager


# ------------------------------------------------------------------------------- basic approve


def test_approves_order_within_all_limits():
    rm = RiskManager(loose_limits())
    account = make_account(equity=10_000)
    order = make_order("AAPL", Side.BUY, 10)
    quote = make_quote("AAPL", 100)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(10)
    assert len(rm.audit) == 1
    assert rm.audit[0]["approved"] is True


# ------------------------------------------------------------------------------- position cap


def test_resizes_down_to_per_symbol_position_notional_cap():
    rm = RiskManager(loose_limits(max_position_notional_usd=2_000.0))
    account = make_account(equity=10_000)
    order = make_order("AAPL", Side.BUY, 100)  # 100 * 50 = $5,000 requested
    quote = make_quote("AAPL", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(40)  # 2000 / 50


def test_position_cap_uses_tighter_of_abs_and_pct_equity():
    rm = RiskManager(loose_limits(max_position_notional_usd=1_000_000.0, max_position_pct_equity=0.10))
    account = make_account(equity=10_000)  # 10% of equity = $1,000
    order = make_order("AAPL", Side.BUY, 100)
    quote = make_quote("AAPL", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(20)  # 1000 / 50


# ------------------------------------------------------------------------------- order notional cap


def test_resizes_down_to_max_order_notional():
    rm = RiskManager(loose_limits(max_order_notional_usd=500.0))
    account = make_account(equity=100_000)
    order = make_order("AAPL", Side.BUY, 100)
    quote = make_quote("AAPL", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(10)  # 500 / 50


# ------------------------------------------------------------------------------- gross / net exposure


def test_resizes_down_to_gross_exposure_cap():
    rm = RiskManager(loose_limits(max_gross_exposure_pct=0.5))
    positions = {"AAPL": make_position("AAPL", 80, 50.0)}  # $4,000 already gross
    account = make_account(equity=10_000, positions=positions)  # gross cap = $5,000
    order = make_order("MSFT", Side.BUY, 100)
    quote = make_quote("MSFT", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(20)  # budget 1000 / 50


def test_resizes_down_to_net_exposure_cap():
    rm = RiskManager(loose_limits(max_net_exposure_pct=0.3))
    account = make_account(equity=10_000)  # net cap = $3,000
    order = make_order("MSFT", Side.BUY, 100)
    quote = make_quote("MSFT", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(60)  # 3000 / 50


def test_selling_to_reduce_exposure_is_never_blocked_by_exposure_caps():
    rm = RiskManager(loose_limits(max_gross_exposure_pct=0.01, max_net_exposure_pct=0.01))
    positions = {"AAPL": make_position("AAPL", 100, 50.0)}
    account = make_account(equity=10_000, positions=positions)
    order = make_order("AAPL", Side.SELL, 50)
    quote = make_quote("AAPL", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(50)


# ------------------------------------------------------------------------------- crypto cap


def test_resizes_down_to_crypto_asset_class_cap():
    rm = RiskManager(loose_limits(crypto_max_pct_equity=0.10))
    account = make_account(equity=10_000)  # crypto cap = $1,000
    order = make_order("ETH/USD", Side.BUY, 10)
    quote = make_quote("ETH/USD", 2_000)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(0.5)  # 1000 / 2000


# ------------------------------------------------------------------------------- long-only


def test_long_only_clamps_sell_to_available_position():
    rm = RiskManager(loose_limits())
    positions = {"BTC/USD": make_position("BTC/USD", 1.0, 30_000.0)}
    account = make_account(equity=50_000, positions=positions)
    order = make_order("BTC/USD", Side.SELL, 2.0)  # non-shortable: Instrument.infer -> shortable=False
    quote = make_quote("BTC/USD", 30_000)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(1.0)


def test_long_only_rejects_sell_with_no_position():
    rm = RiskManager(loose_limits())
    account = make_account(equity=50_000)
    order = make_order("BTC/USD", Side.SELL, 1.0)
    quote = make_quote("BTC/USD", 30_000)
    decision = rm.check(order, account, quote, NOW)
    assert not decision.approved
    assert decision.order is None
    assert "long-only" in decision.reason


# ------------------------------------------------------------------------------- per-strategy cap


def test_resizes_down_to_per_strategy_capital_cap():
    rm = RiskManager(loose_limits(default_strategy_capital_pct=0.10))
    account = make_account(equity=10_000)  # strategy cap = $1,000
    order = make_order("AAPL", Side.BUY, 100, strategy_id="momentum_v1")
    quote = make_quote("AAPL", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(20)


def test_per_strategy_cap_accounts_for_prior_fills():
    rm = RiskManager(loose_limits(default_strategy_capital_pct=0.10))
    account = make_account(equity=10_000)  # strategy cap = $1,000
    fill = Fill("prior", "AAPL", Side.BUY, 15, 50.0, fee=0.0, liquidity=Liquidity.TAKER, ts=NOW, strategy_id="momentum_v1")
    rm.on_fill(fill, make_account(equity=10_000, positions={"AAPL": make_position("AAPL", 15, 50.0)}))
    order = make_order("AAPL", Side.BUY, 100, strategy_id="momentum_v1")
    quote = make_quote("AAPL", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    # budget left = 1000 - 15*50 = 250 -> 5 shares
    assert decision.order.qty == pytest.approx(5)


# ------------------------------------------------------------------------------- max open orders


def test_max_open_orders_per_symbol():
    rm = RiskManager(loose_limits(max_open_orders_per_symbol=1))
    account = make_account(equity=100_000)
    quote = make_quote("AAPL", 50)
    d1 = rm.check(make_order("AAPL", Side.BUY, 1), account, quote, NOW)
    assert d1.approved
    d2 = rm.check(make_order("AAPL", Side.BUY, 1), account, quote, NOW)
    assert not d2.approved
    assert "max_open_orders_per_symbol" in d2.reason


# ------------------------------------------------------------------------------- throttle


def test_throttle_max_orders_per_minute():
    rm = RiskManager(loose_limits(max_orders_per_minute=2, max_open_orders_per_symbol=1000))
    account = make_account(equity=100_000)
    quote = make_quote("AAPL", 50)
    d1 = rm.check(make_order("AAPL", Side.BUY, 1), account, quote, NOW)
    d2 = rm.check(make_order("AAPL", Side.BUY, 1), account, quote, NOW)
    d3 = rm.check(make_order("AAPL", Side.BUY, 1), account, quote, NOW)
    assert d1.approved and d2.approved
    assert not d3.approved
    assert "throttled" in d3.reason


# ------------------------------------------------------------------------------- fat finger


def test_fat_finger_rejects_limit_far_from_mid():
    rm = RiskManager(loose_limits(fat_finger_band_bps=50.0))
    account = make_account(equity=100_000)
    quote = make_quote("AAPL", 100)
    order = make_order("AAPL", Side.BUY, 1, type=OrderType.LIMIT, limit_price=110.0)  # 10% away
    decision = rm.check(order, account, quote, NOW)
    assert not decision.approved
    assert "fat-finger" in decision.reason


def test_fat_finger_allows_limit_within_band():
    rm = RiskManager(loose_limits(fat_finger_band_bps=200.0))
    account = make_account(equity=100_000)
    quote = make_quote("AAPL", 100)
    order = make_order("AAPL", Side.BUY, 1, type=OrderType.LIMIT, limit_price=101.0)  # 100bps away
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved


# ------------------------------------------------------------------------------- min notional


def test_min_notional_rejects_dust_order():
    rm = RiskManager(loose_limits(min_notional_usd=100.0))
    account = make_account(equity=100_000)
    order = make_order("AAPL", Side.BUY, 0.5)
    quote = make_quote("AAPL", 100)  # notional = $50 < $100
    decision = rm.check(order, account, quote, NOW)
    assert not decision.approved
    assert "min_notional" in decision.reason


# ------------------------------------------------------------------------------- trading hours


def test_equity_rejected_outside_trading_hours():
    rm = RiskManager(loose_limits(trading_hours=TradingHours(enabled=True)))
    account = make_account(equity=100_000)
    order = make_order("AAPL", Side.BUY, 1)
    quote = make_quote("AAPL", 100)
    decision = rm.check(order, account, quote, AFTER_HOURS)
    assert not decision.approved
    assert "trading hours" in decision.reason


def test_equity_approved_inside_trading_hours():
    rm = RiskManager(loose_limits(trading_hours=TradingHours(enabled=True)))
    account = make_account(equity=100_000)
    order = make_order("AAPL", Side.BUY, 1)
    quote = make_quote("AAPL", 100)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved


def test_crypto_ignores_trading_hours():
    rm = RiskManager(loose_limits(trading_hours=TradingHours(enabled=True)))
    account = make_account(equity=100_000)
    order = make_order("BTC/USD", Side.BUY, 0.01)
    quote = make_quote("BTC/USD", 30_000)
    decision = rm.check(order, account, quote, AFTER_HOURS)
    assert decision.approved


# ------------------------------------------------------------------------------- reduce-only


def test_reduce_only_allowed_even_when_halted():
    rm = RiskManager(loose_limits())
    positions = {"AAPL": make_position("AAPL", 10, 50.0)}
    account = make_account(equity=10_000, positions=positions)
    rm.kill("test kill")
    assert rm.halted

    reduce_order = make_order("AAPL", Side.SELL, 10, reduce_only=True)
    quote = make_quote("AAPL", 50)
    decision = rm.check(reduce_order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(10)

    new_risk_order = make_order("AAPL", Side.BUY, 1)
    decision2 = rm.check(new_risk_order, account, quote, NOW)
    assert not decision2.approved
    assert "halted" in decision2.reason


def test_reduce_only_clamped_to_available_position():
    rm = RiskManager(loose_limits())
    positions = {"AAPL": make_position("AAPL", 5, 50.0)}
    account = make_account(equity=10_000, positions=positions)
    order = make_order("AAPL", Side.SELL, 100, reduce_only=True)
    quote = make_quote("AAPL", 50)
    decision = rm.check(order, account, quote, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(5)


def test_reduce_only_rejected_with_no_position():
    rm = RiskManager(loose_limits())
    account = make_account(equity=10_000)
    order = make_order("AAPL", Side.SELL, 10, reduce_only=True)
    decision = rm.check(order, account, None, NOW)
    assert not decision.approved


def test_reduce_only_works_without_a_quote():
    rm = RiskManager(loose_limits())
    positions = {"AAPL": make_position("AAPL", 5, 50.0)}
    account = make_account(equity=10_000, positions=positions)
    order = make_order("AAPL", Side.SELL, 5, reduce_only=True)
    decision = rm.check(order, account, None, NOW)
    assert decision.approved
    assert decision.order.qty == pytest.approx(5)


# ------------------------------------------------------------------------------- kill switch / drawdown


def test_drawdown_kill_switch_halts_and_flags_flatten():
    rm = RiskManager(loose_limits(max_drawdown_pct=0.10))
    rm.on_mark(make_account(equity=10_000), NOW)
    assert not rm.halted
    rm.on_mark(make_account(equity=8_900), NOW)  # -11% from HWM
    assert rm.halted
    assert rm.flatten
    assert "kill switch" in rm.halt_reason

    order = make_order("AAPL", Side.BUY, 1)
    quote = make_quote("AAPL", 100)
    decision = rm.check(order, make_account(equity=8_900), quote, NOW)
    assert not decision.approved


def test_reset_clears_kill_switch():
    rm = RiskManager(loose_limits(max_drawdown_pct=0.10))
    rm.on_mark(make_account(equity=10_000), NOW)
    rm.on_mark(make_account(equity=8_900), NOW)
    assert rm.halted
    rm.reset()
    assert not rm.halted
    assert not rm.flatten


# ------------------------------------------------------------------------------- daily loss halt


def test_daily_loss_halts_new_risk_but_is_not_a_kill_switch():
    rm = RiskManager(loose_limits(max_daily_loss_pct=0.02, max_drawdown_pct=1.0))
    rm.on_mark(make_account(equity=10_000), NOW)  # sets today's baseline
    rm.on_mark(make_account(equity=9_700), NOW)  # -3% daily
    assert rm.halted
    assert not rm.flatten  # daily-loss halt, not the drawdown kill switch

    order = make_order("AAPL", Side.BUY, 1)
    quote = make_quote("AAPL", 100)
    decision = rm.check(order, make_account(equity=9_700), quote, NOW)
    assert not decision.approved
    assert "daily loss" in decision.reason


def test_daily_loss_halt_clears_on_next_trading_day():
    rm = RiskManager(loose_limits(max_daily_loss_pct=0.02, max_drawdown_pct=1.0, daily_reset_hour_utc=0))
    rm.on_mark(make_account(equity=10_000), NOW)
    rm.on_mark(make_account(equity=9_700), NOW)
    assert rm.halted
    next_day = NOW + pd.Timedelta(days=1)
    rm.on_mark(make_account(equity=9_700), next_day)
    assert not rm.halted


# ------------------------------------------------------------------------------- consecutive losses


def test_consecutive_loss_cooldown_and_expiry():
    rm = RiskManager(loose_limits(max_consecutive_losses=2, consecutive_loss_cooldown_minutes=30.0))

    def losing_round_trip(ts):
        rm.on_fill(
            Fill("c1", "AAPL", Side.BUY, 10, 100.0, fee=0.0, liquidity=Liquidity.TAKER, ts=ts, strategy_id="s1"),
            make_account(equity=10_000),
        )
        rm.on_fill(
            Fill("c2", "AAPL", Side.SELL, 10, 90.0, fee=0.0, liquidity=Liquidity.TAKER, ts=ts, strategy_id="s1"),
            make_account(equity=9_900),
        )

    losing_round_trip(NOW)
    assert not rm.halted
    losing_round_trip(NOW)
    assert rm.halted  # 2 consecutive losses trips the cooldown

    order = make_order("MSFT", Side.BUY, 1)
    quote = make_quote("MSFT", 50)
    decision = rm.check(order, make_account(equity=9_800), quote, NOW)
    assert not decision.approved
    assert "cooldown" in decision.reason

    later = NOW + pd.Timedelta(minutes=31)
    decision2 = rm.check(order, make_account(equity=9_800), quote, later)
    assert decision2.approved
