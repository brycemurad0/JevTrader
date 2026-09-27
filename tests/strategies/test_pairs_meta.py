"""Kalman pairs on a synthetic cointegrated pair (known truth) and the meta-allocator plumbing."""

from __future__ import annotations

import pandas as pd

from jevtrader.backtest import Backtester, SlippageModel
from jevtrader.core.fees import BpsFees, CompositeFees
from jevtrader.data.synthetic import bars_to_events, generate_cointegrated_pair
from jevtrader.strategies.meta_allocator import MetaAllocator
from jevtrader.strategies.pairs_kalman import KalmanHedge, PairsKalman

START = pd.Timestamp("2024-01-02 14:30", tz="UTC")


def _pair(days=30, seed=4, beta=1.5):
    return generate_cointegrated_pair("AAA", "BBB", START, START + pd.Timedelta(days=days), freq="15min", seed=seed, beta=beta, half_life_bars=20, spread_vol=0.2)


def test_kalman_converges_to_true_beta():
    pair = _pair()
    kf = KalmanHedge(delta=1e-4)
    for a, b in zip(pair["AAA"]["close"], pair["BBB"]["close"]):
        _, _, beta = kf.update(float(a), float(b))
    assert abs(beta - 1.5) < 0.15


def test_pairs_strategy_trades_dollar_neutral_and_profits_on_ground_truth():
    pair = _pair(days=40)
    strat = PairsKalman(["AAA", "BBB"], {"bar_minutes": 15, "entry_z": 2.0, "exit_z": 0.5, "warmup": 60, "alloc_frac": 0.3})
    bt = Backtester([strat], {s: bars_to_events(df, s) for s, df in pair.items()}, fees=BpsFees(0.0, 0.0), fill_model=SlippageModel(0.0, 0.0, 0.0, float("inf")), initial_cash=100_000, allow_leverage=True, leverage=2.0)
    res = bt.run()
    assert res.metrics["n_fills"] >= 8
    assert set(res.fills["symbol"]) == {"AAA", "BBB"}
    # legs are opposite-signed on each entry
    first_ts = res.fills["ts"].iloc[0]
    legs = res.fills[res.fills["ts"] == first_ts]
    assert len(legs) == 2 and set(legs["side"]) == {"buy", "sell"}
    assert res.metrics["total_return"] > 0, "on a true OU spread with zero costs the pairs trade must be profitable"


def test_pairs_costs_matter_with_alpaca_fees():
    pair = _pair(days=40)
    mk = lambda: PairsKalman(["AAA", "BBB"], {"bar_minutes": 15, "entry_z": 2.0, "exit_z": 0.5, "warmup": 60, "alloc_frac": 0.3})
    free = Backtester([mk()], {s: bars_to_events(df, s) for s, df in pair.items()}, fees=BpsFees(0.0, 0.0), fill_model=SlippageModel(0.0, 0.0, 0.0, float("inf")), initial_cash=100_000, allow_leverage=True, leverage=2.0).run()
    paid = Backtester([mk()], {s: bars_to_events(df, s) for s, df in pair.items()}, fees=CompositeFees(), fill_model=SlippageModel(1.0, 0.5, 0.0, float("inf")), initial_cash=100_000, allow_leverage=True, leverage=2.0).run()
    assert paid.metrics["final_equity"] < free.metrics["final_equity"]


def test_meta_allocator_runs_children_with_virtual_positions_and_weights(btc_bars, spy_bars):
    children = [
        {"name": "zscore_reversion", "symbols": ["BTC/USD"], "params": {"bar_minutes": 5, "window": 12, "entry_z": 1.5, "exit_z": 0.0, "vol_window": 48, "max_vol_mult": 100.0, "session_only": False, "alloc_frac": 1.0, "min_delta_frac": 0.0}},
        {"name": "trend_follow", "symbols": ["BTC/USD"], "params": {"bar_minutes": 5, "mode": "ema", "fast": 5, "slow": 20, "alloc_frac": 1.0, "min_delta_frac": 0.0}},
    ]
    meta = MetaAllocator([], {"children": children, "scheme": "inverse_vol", "lookback": 100, "rebalance_bars": 60, "min_weight": 0.1, "max_weight": 0.9, "capital_frac": 0.8})
    assert meta.symbols == ["BTC/USD"]
    bt = Backtester([meta], {"BTC/USD": bars_to_events(btc_bars, "BTC/USD")}, fees=CompositeFees(), initial_cash=100_000, fill_model=SlippageModel(max_participation=float("inf")))
    res = bt.run()
    assert res.metrics["n_fills"] > 0
    assert all(o.startswith("meta_allocator") for o in res.fills["strategy_id"])
    assert abs(sum(meta._weights.values()) - 1.0) < 1e-6
    assert len(meta.weight_history) >= 1, "weights should have been re-estimated at least once"
    # each child's virtual positions are tracked separately and the account nets them
    virt = sum(cc.positions.get("BTC/USD").qty if cc.positions.get("BTC/USD") else 0.0 for cc in meta._child_ctx.values())
    real = res.positions.get("BTC/USD").qty if res.positions.get("BTC/USD") else 0.0
    assert abs(virt - real) < 1e-6
    # long-only crypto is still respected through the proxy
    signed = res.fills.apply(lambda r: r["qty"] if r["side"] == "buy" else -r["qty"], axis=1).cumsum()
    assert signed.min() >= -1e-9


def test_meta_allocator_equal_weight_and_hrp_schemes(btc_bars):
    children = [
        {"name": "zscore_reversion", "symbols": ["BTC/USD"], "params": {"bar_minutes": 5, "window": 12, "entry_z": 1.5, "exit_z": 0.0, "vol_window": 48, "max_vol_mult": 100.0, "session_only": False, "alloc_frac": 1.0, "min_delta_frac": 0.0}},
        {"name": "trend_follow", "symbols": ["BTC/USD"], "params": {"bar_minutes": 5, "mode": "ema", "fast": 5, "slow": 20, "alloc_frac": 1.0, "min_delta_frac": 0.0}},
    ]
    for scheme in ("equal_weight", "hrp"):
        meta = MetaAllocator([], {"children": children, "scheme": scheme, "lookback": 100, "rebalance_bars": 60})
        res = Backtester([meta], {"BTC/USD": bars_to_events(btc_bars, "BTC/USD")}, fees=CompositeFees(), initial_cash=100_000, fill_model=SlippageModel(max_participation=float("inf"))).run()
        assert len(res.equity_curve) == len(btc_bars)
