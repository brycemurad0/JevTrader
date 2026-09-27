from __future__ import annotations

import pandas as pd
import pytest

from jevtrader.core.fees import BpsFees
from jevtrader.core.types import AccountState, AssetClass, Instrument, Position, Side
from jevtrader.rebalance.planner import RebalancePlanner

NOW = pd.Timestamp("2024-01-02 15:00", tz="UTC")


def _instruments():
    return {
        "SPY": Instrument("SPY", AssetClass.EQUITY, tick_size=0.01, lot_size=1.0, min_notional=1.0, shortable=True),
        "TLT": Instrument("TLT", AssetClass.EQUITY, tick_size=0.01, lot_size=1.0, min_notional=1.0, shortable=True),
        "BTC/USD": Instrument("BTC/USD", AssetClass.CRYPTO, tick_size=0.01, lot_size=1e-6, min_notional=1.0, shortable=False),
    }


def _planner(offset_bps=5.0):
    return RebalancePlanner(_instruments(), BpsFees(maker_bps=0.0, taker_bps=10.0), limit_offset_bps=offset_bps)


def test_plan_sells_before_buys():
    account = AccountState(
        cash=0.0,
        equity=20_000.0,
        buying_power=20_000.0,
        positions={"SPY": Position("SPY", 100, 100.0)},  # $10,000 in SPY
    )
    prices = {"SPY": 100.0, "TLT": 90.0}
    targets = {"SPY": 0.0, "TLT": 1.0}  # sell all SPY, buy all TLT
    plan = _planner().plan(account, prices, targets, now=NOW)

    sides = [item.side for item in plan.items]
    assert Side.SELL in sides and Side.BUY in sides
    sell_positions = [i for i, s in enumerate(sides) if s is Side.SELL]
    buy_positions = [i for i, s in enumerate(sides) if s is Side.BUY]
    assert max(sell_positions) < min(buy_positions)  # every sell precedes every buy


def test_plan_never_exceeds_available_cash():
    account = AccountState(cash=1_000.0, equity=1_000.0, buying_power=1_000.0, positions={})
    prices = {"SPY": 400.0, "TLT": 90.0}
    targets = {"SPY": 0.5, "TLT": 0.5}
    plan = _planner().plan(account, prices, targets, now=NOW)
    total_buy_notional = sum(i.notional for i in plan.items if i.side is Side.BUY)
    assert total_buy_notional <= account.cash + 1e-6


def test_plan_scales_down_buys_when_cash_insufficient_even_after_sells():
    account = AccountState(
        cash=100.0,
        equity=10_100.0,
        buying_power=100.0,
        positions={"SPY": Position("SPY", 25, 100.0)},  # $2,500 currently
    )
    prices = {"SPY": 100.0, "TLT": 100.0}
    # target wants to shrink SPY a bit and buy a lot of TLT - more than cash allows
    targets = {"SPY": 0.05, "TLT": 0.95}
    plan = _planner().plan(account, prices, targets, now=NOW)
    total_buy = sum(i.notional for i in plan.items if i.side is Side.BUY)
    cash_available = account.cash + sum(i.notional for i in plan.items if i.side is Side.SELL)
    assert total_buy <= cash_available + 1e-6
    assert any("scaled" in n or "trimmed" in n for n in plan.notes) or total_buy <= cash_available


def test_plan_respects_long_only_non_shortable():
    account = AccountState(cash=0.0, equity=10_000.0, buying_power=0.0, positions={})
    prices = {"BTC/USD": 30_000.0}
    targets = {"BTC/USD": -0.5}  # nonsensical negative target: nothing to sell, no position held
    plan = _planner().plan(account, prices, targets, now=NOW)
    assert all(i.side is not Side.SELL or i.qty <= 0 for i in plan.items)
    assert not any(i.symbol == "BTC/USD" for i in plan.items)  # no position to sell -> skipped


def test_plan_respects_lot_size_and_min_notional():
    account = AccountState(cash=10_000.0, equity=10_000.0, buying_power=10_000.0, positions={})
    prices = {"BTC/USD": 30_000.0}
    targets = {"BTC/USD": 0.00001}  # tiny target -> notional below min_notional
    plan = _planner().plan(account, prices, targets, now=NOW)
    assert plan.items == []


def test_plan_limit_prices_offset_from_mid():
    account = AccountState(cash=10_000.0, equity=10_000.0, buying_power=10_000.0, positions={})
    prices = {"SPY": 100.0}
    targets = {"SPY": 0.5}
    plan = _planner(offset_bps=10.0).plan(account, prices, targets, now=NOW)
    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.side is Side.BUY
    assert item.limit_price == pytest.approx(100.0 * 1.001)


def test_plan_to_markdown_and_to_orders():
    account = AccountState(cash=10_000.0, equity=10_000.0, buying_power=10_000.0, positions={})
    prices = {"SPY": 100.0}
    targets = {"SPY": 0.5}
    plan = _planner().plan(account, prices, targets, now=NOW)
    md = plan.to_markdown()
    assert "Rebalance plan" in md
    assert "dry-run" in md.lower()
    orders = plan.to_orders(strategy_id="smart_rebalance")
    assert len(orders) == len(plan.items)
    assert orders[0].tag == "rebalance"


def test_empty_plan_markdown():
    account = AccountState(cash=10_000.0, equity=10_000.0, buying_power=10_000.0, positions={})
    plan = _planner().plan(account, {}, {}, now=NOW)
    assert plan.items == []
    assert "No trades" in plan.to_markdown()
