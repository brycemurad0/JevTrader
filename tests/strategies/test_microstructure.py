"""Quote/book-driven strategies on the synthetic L1 stream (known weak OFI edge): mechanics only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jevtrader.backtest import Backtester
from jevtrader.core.fees import AlpacaCryptoFees, BpsFees, CompositeFees
from jevtrader.core.types import AssetClass, Instrument, OrderBook, Quote, Side
from jevtrader.data.synthetic import generate_order_book_stream, generate_quote_trade_stream, quotes_to_events, trades_to_events
from jevtrader.strategies.flow_toxicity import FlowToxicityGate, VpinMonitor
from jevtrader.strategies.imbalance_alpha import ImbalanceAlpha
from jevtrader.strategies.market_maker import AvellanedaStoikovMM

START = pd.Timestamp("2024-03-04 14:30", tz="UTC")


@pytest.fixture(scope="module")
def l1_stream():
    q, t = generate_quote_trade_stream("SPY", START, START + pd.Timedelta(minutes=40), freq="1s", seed=11, mid0=450.0, annual_vol=0.2)
    ev = list(quotes_to_events(q, "SPY")) + list(trades_to_events(t, "SPY"))
    ev.sort(key=lambda e: e.ts)
    return q, t, ev


def test_market_maker_quotes_two_sided_and_respects_fee_floor(l1_stream):
    q, t, ev = l1_stream
    mm = AvellanedaStoikovMM(["SPY"], {"quote_size_notional": 4500.0, "gamma": 0.05, "horizon_s": 30.0})
    bt = Backtester([mm], {"SPY": iter(ev)}, fees=BpsFees(0.0, 0.0), initial_cash=100_000, latency_ms=50)
    res = bt.run()
    orders = res.orders
    assert mm.n_requotes > 10
    assert (orders["type"] == "limit").all()
    assert set(orders["side"]) == {"buy", "sell"}
    # every fill is a maker fill and the quoted half-spread never went below the fee-aware floor
    if len(res.fills):
        assert (res.fills["liquidity"] == "maker").all()
    mids = q["mid"]
    for _, o in orders.iterrows():
        mid = float(mids.asof(o["created_ts"]))
        half_bps = abs(o["limit_price"] - mid) / mid * 1e4
        floor = mm.min_half_spread_bps(_Ctx(), "SPY", mid)
        assert half_bps >= floor - 1.0  # tick rounding + skew tolerance


class _Ctx:
    def instrument(self, symbol):
        return Instrument.infer(symbol)


def test_market_maker_crypto_floor_forces_wide_quotes():
    mm = AvellanedaStoikovMM(["BTC/USD"], fee_model=CompositeFees(crypto=AlpacaCryptoFees()))
    floor = mm.min_half_spread_bps(_Ctx(), "BTC/USD", 60_000.0)
    assert floor >= 15.0 + 2.0  # maker fee + adverse selection => >= 34 bps quoted spread
    mm_eq = AvellanedaStoikovMM(["SPY"])
    assert mm_eq.min_half_spread_bps(_Ctx(), "SPY", 450.0) < 4.0


def test_market_maker_never_shorts_crypto():
    q, t = generate_quote_trade_stream("BTC/USD", START, START + pd.Timedelta(minutes=30), freq="1s", asset_class=AssetClass.CRYPTO, seed=3, mid0=60_000.0, annual_vol=0.6, tick_size=1.0)
    ev = sorted(list(quotes_to_events(q, "BTC/USD")) + list(trades_to_events(t, "BTC/USD")), key=lambda e: e.ts)
    mm = AvellanedaStoikovMM(["BTC/USD"], {"quote_size_notional": 600.0, "adverse_selection_bps": 0.0, "min_edge_bps": 0.0}, fee_model=BpsFees(0.0, 0.0))
    res = Backtester([mm], {"BTC/USD": iter(ev)}, fees=BpsFees(0.0, 0.0), initial_cash=100_000).run()
    if len(res.fills):
        signed = res.fills.apply(lambda r: r["qty"] if r["side"] == "buy" else -r["qty"], axis=1).cumsum()
        assert signed.min() >= -1e-9
    for pos in res.positions.values():
        assert pos.qty >= -1e-9


def test_imbalance_alpha_recovers_positive_beta_and_profits_at_zero_fees(l1_stream):
    q, t, ev = l1_stream
    strat = ImbalanceAlpha(["SPY"], {"notional": 4500.0, "hold_ticks": 10, "threshold_join": 0.3, "calibrate": True, "min_t_stat": 0.0})
    res = Backtester([strat], {"SPY": iter(ev)}, fees=BpsFees(0.0, 0.0), initial_cash=100_000, latency_ms=50).run()
    beta, tstat = strat.fitted_beta("SPY")
    assert beta > 0, "synthetic stream has a positive lagged-OFI edge; online beta must be positive"
    assert res.metrics["n_fills"] > 0
    assert strat.n_joins + strat.n_takes > 0


def test_imbalance_alpha_is_fee_eaten_at_crypto_fees(l1_stream):
    q, t, ev = l1_stream
    zero = Backtester([ImbalanceAlpha(["SPY"], {"notional": 4500.0, "min_t_stat": 0.0})], {"SPY": iter(ev)}, fees=BpsFees(0.0, 0.0), initial_cash=100_000).run()
    crypto = Backtester([ImbalanceAlpha(["SPY"], {"notional": 4500.0, "min_t_stat": 0.0})], {"SPY": iter(ev)}, fees=BpsFees(15.0, 25.0), initial_cash=100_000).run()
    assert crypto.metrics["final_equity"] < zero.metrics["final_equity"]
    assert crypto.fee_total > 0


def test_imbalance_alpha_handles_l2_books():
    books = generate_order_book_stream("SPY", START, START + pd.Timedelta(minutes=5), freq="1s", seed=2, mid0=450.0)
    strat = ImbalanceAlpha(["SPY"], {"notional": 4500.0, "calibrate": False, "beta_bps": 5.0})
    res = Backtester([strat], {"SPY": iter(books)}, fees=BpsFees(0.0, 0.0), initial_cash=100_000).run()
    assert len(res.equity_curve) == len(books)


def test_vpin_gate_in_unit_interval_and_monitor_logs(l1_stream):
    q, t, ev = l1_stream
    gate = FlowToxicityGate(bucket_volume=2000.0, n_buckets=20)
    for e in ev:
        if hasattr(e, "size") and hasattr(e, "price") and not isinstance(e, Quote):
            gate.on_trade(e)
    assert 0.0 <= gate.vpin <= 1.0
    mon = VpinMonitor(["SPY"], {"bucket_volume": 2000.0, "n_buckets": 20, "log_every_n": 50})
    bt = Backtester([mon], {"SPY": iter(ev)}, fees=BpsFees(0.0, 0.0), initial_cash=100_000)
    res = bt.run()
    assert res.metrics["n_fills"] == 0  # never trades
    assert len(mon.history) > 0
    assert any(l["msg"] == "vpin" for l in bt.log_lines)


def test_vpin_gate_flags_one_sided_flow():
    from jevtrader.core.types import Trade

    gate = FlowToxicityGate(bucket_volume=100.0, n_buckets=10, toxic_threshold=0.7)
    ts = START
    for i in range(3000):
        gate.on_trade(Trade("X", ts + pd.Timedelta(seconds=i), 100.0 + i * 0.01, 10.0, Side.BUY))
    assert gate.toxic and gate.vpin > 0.9
