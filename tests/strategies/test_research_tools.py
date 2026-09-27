"""Research harness: loaders/splits/resampling, fast-screen invariants, record/replay round trip,
and a tiny end-to-end `run_bar_study` on synthetic data (no cached real data required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jevtrader.core.fees import BpsFees
from jevtrader.core.types import AssetClass, Side
from jevtrader.data.synthetic import generate_quote_trade_stream, quotes_to_events, trades_to_events
from jevtrader.research import signals as S
from jevtrader.research.experiments import Dataset, run_bar_study
from jevtrader.research.fastscreen import ALPACA_CRYPTO_TAKER, CostSpec, fee_ladder, robustness, simulate_positions, sweep_grid
from jevtrader.research.loaders import chrono_split, resample, to_bar_events, us_session_only
from jevtrader.research.record_replay import (
    MarketDataRecorder,
    compare_sim_vs_paper,
    load_market_events,
    realized_cost_bps,
    replay_strategy,
    write_events,
)
from jevtrader.strategies.imbalance_alpha import ImbalanceAlpha


def test_chrono_split_is_60_20_20_and_ordered(btc_bars):
    sp = chrono_split(btc_bars)
    n = len(btc_bars)
    assert abs(len(sp.train) - 0.6 * n) <= 1 and abs(len(sp.val) - 0.2 * n) <= 1
    assert sp.train.index[-1] < sp.val.index[0] < sp.val.index[-1] < sp.holdout.index[0]
    assert "train" in sp.describe()


def test_resample_is_right_closed_and_labelled(btc_bars):
    r = resample(btc_bars, "15min")
    first = r.index[0]
    src = btc_bars.loc[first - pd.Timedelta(minutes=14) : first]
    assert r["open"].iloc[0] == src["open"].iloc[0]
    assert r["close"].iloc[0] == src["close"].iloc[-1]
    assert r["high"].iloc[0] == src["high"].max()
    assert abs(r["volume"].iloc[0] - src["volume"].sum()) < 1e-9


def test_us_session_filter(spy_bars):
    s = us_session_only(spy_bars)
    m = s.index.hour * 60 + s.index.minute
    assert m.min() >= 13 * 60 + 31 and m.max() <= 20 * 60


def test_to_bar_events_preserves_order_and_values(btc_bars):
    ev = list(to_bar_events(btc_bars.head(50), "BTC/USD"))
    assert len(ev) == 50 and ev[0].ts == btc_bars.index[0] and ev[-1].close == btc_bars["close"].iloc[49]


def test_simulate_positions_costs_and_edge_arithmetic(btc_bars):
    bars = resample(btc_bars, "5min")
    # always long => one entry, no exits; gross edge equals buy&hold from open[1] to open[-1]
    tgt = pd.Series(1.0, index=bars.index)
    r = simulate_positions(bars, tgt, CostSpec(0.0, 0.0, 0.0), asset_class=AssetClass.CRYPTO)
    bh = bars["open"].iloc[-1] / bars["open"].iloc[1] - 1
    assert abs(r.equity.iloc[-1] - 1 - bh) < 1e-9
    assert r.n_round_trips == 0.5
    # costs: flipping every bar at 10 bps/side loses ~turnover * 10 bps
    flip = pd.Series(np.where(np.arange(len(bars)) % 2 == 0, 1.0, 0.0), index=bars.index)
    zero = simulate_positions(bars, flip, CostSpec(0.0, 0.0, 0.0), asset_class=AssetClass.CRYPTO)
    paid = simulate_positions(bars, flip, CostSpec(10.0, 0.0, 0.0), asset_class=AssetClass.CRYPTO)
    expected_cost = paid.turnover.sum() * 10 / 1e4
    assert abs((zero.net_returns.sum() - paid.net_returns.sum()) - expected_cost) < 1e-9
    assert abs(paid.gross_edge_bps_per_rt - zero.gross_edge_bps_per_rt) < 1e-9
    assert abs(paid.breakeven_fee_bps - paid.gross_edge_bps_per_rt / 2) < 1e-9


def test_long_only_clips_negative_targets(btc_bars):
    bars = resample(btc_bars, "5min")
    tgt = pd.Series(-1.0, index=bars.index)
    r = simulate_positions(bars, tgt, CostSpec(0.0), asset_class=AssetClass.CRYPTO, long_only=True)
    assert (r.position == 0).all()


def test_fee_ladder_and_robustness_are_monotone_in_fees(btc_bars):
    bars = resample(btc_bars, "5min")
    tgt = S.ema_crossover(bars, 5, 20)
    lad = fee_ladder(bars, tgt, ALPACA_CRYPTO_TAKER, fees_bps=(25.0, 10.0, 0.0), asset_class=AssetClass.CRYPTO)
    assert list(lad["fee_bps_per_side"]) == [25.0, 10.0, 0.0]
    assert lad["total_return"].is_monotonic_increasing
    rob = robustness(bars, tgt, ALPACA_CRYPTO_TAKER, AssetClass.CRYPTO)
    assert set(rob) >= {"base", "fees_x1.5", "fees_x2", "slippage_x2", "passed"}
    assert rob["fees_x2"]["total_return"] <= rob["base"]["total_return"]


def test_sweep_grid_records_n_trials(btc_bars):
    bars = resample(btc_bars, "5min")
    df = sweep_grid({"fast": [3, 5], "slow": [10, 20]}, lambda p: simulate_positions(bars, S.ema_crossover(bars, p["fast"], p["slow"]), CostSpec(1.0), asset_class=AssetClass.CRYPTO, compute_full_metrics=False))
    assert len(df) == 4 and df.attrs["n_trials"] == 4 and "gross_edge_bps_per_rt" in df


def test_run_bar_study_end_to_end_on_synthetic(btc_bars):
    ds = Dataset("synthetic BTC", btc_bars, AssetClass.CRYPTO, ALPACA_CRYPTO_TAKER, "BTC/USD", long_only=True)
    grid = {"fast": [3, 5], "slow": [10, 20]}
    out = run_bar_study(ds, "5min", lambda b, p: S.ema_crossover(b, int(p["fast"]), int(p["slow"]), long_only=True), grid, min_train_rt=1, stability_params=("fast", "slow"), crypto_ladder=True)
    assert out.n_trials == 4
    assert set(out.rows) == {"train", "val", "holdout"}
    assert out.verdict in {"VIABLE", "MARGINAL", "NOT VIABLE"}
    md = out.to_markdown()
    assert "Fee ladder" in md and "Verdict" in md and "breakeven_fee_bps_per_side" in md


def test_record_replay_round_trip(tmp_path):
    start = pd.Timestamp("2024-03-04 14:30", tz="UTC")
    q, t = generate_quote_trade_stream("SPY", start, start + pd.Timedelta(minutes=3), freq="1s", seed=1, mid0=450.0)
    ev = sorted(list(quotes_to_events(q, "SPY")) + list(trades_to_events(t, "SPY")), key=lambda e: e.ts)
    n = write_events(tmp_path / "market.jsonl.gz", ev)
    assert n == len(ev)
    loaded = load_market_events([tmp_path / "market.jsonl.gz"])
    assert len(loaded["SPY"]) == len(ev)
    assert loaded["SPY"][0].ts == ev[0].ts and type(loaded["SPY"][5]) is type(ev[5])
    only_quotes = load_market_events([tmp_path / "market.jsonl.gz"], kinds={"quote"})
    assert len(only_quotes["SPY"]) == len(q)
    res = replay_strategy(ImbalanceAlpha(["SPY"], {"notional": 4500.0, "min_t_stat": 0.0}), loaded, fees=BpsFees(0.0, 0.0))
    assert len(res.equity_curve) == len({e.ts for e in ev})  # engine marks equity once per timestamp
    paper = res.fills.copy()
    cmp = compare_sim_vs_paper(res, paper)
    assert cmp["sim_n_fills"] == cmp["paper_n_fills"]
    if len(paper):
        assert np.isfinite(realized_cost_bps(paper))


def test_market_recorder_strategy_writes_files(tmp_path, btc_bars):
    from jevtrader.backtest import Backtester

    rec = MarketDataRecorder(["BTC/USD"], {"dir": str(tmp_path), "flush_every": 10})
    res = Backtester([rec], {"BTC/USD": to_bar_events(btc_bars.head(300), "BTC/USD")}, initial_cash=1000).run()
    assert res.metrics["n_fills"] == 0 and rec.n_recorded == 300
    files = sorted(tmp_path.glob("market_*.jsonl"))
    assert files
    loaded = load_market_events(files)
    assert len(loaded["BTC/USD"]) == 300


def test_realized_cost_with_reference_mid():
    fills = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC"), "side": ["buy", "sell"], "qty": [10.0, 10.0], "price": [100.02, 99.98], "fee": [0.0, 0.0]})
    mids = pd.Series([100.0, 100.0], index=fills["ts"])
    # paid 2 cents each way on a $100 stock => 2 bps per side
    assert abs(realized_cost_bps(fills, mids) - 2.0) < 1e-9
