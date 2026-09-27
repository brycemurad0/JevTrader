"""The BarAggregator must reproduce `research.loaders.resample`, and the vectorized fast-screen
must agree with the event-driven Backtester on the same strategy/costs."""

import numpy as np
import pandas as pd

from jevtrader.backtest import Backtester, SlippageModel
from jevtrader.core.fees import AlpacaCryptoFees, CompositeFees
from jevtrader.core.types import AssetClass, Bar
from jevtrader.research import signals as S
from jevtrader.research.fastscreen import CostSpec, simulate_positions
from jevtrader.research.loaders import resample, to_bar_events
from jevtrader.strategies._base import BarAggregator
from jevtrader.strategies.vwap_reversion import ZScoreReversion


def test_aggregator_matches_resample(btc_bars):
    agg = BarAggregator(5)
    out = []
    for b in to_bar_events(btc_bars, "BTC/USD"):
        out.extend(agg.push(b))
    ref = resample(btc_bars, "5min")
    # resample emits the trailing partial bucket; the aggregator (correctly) waits for it to close
    assert len(ref) - 1 <= len(out) <= len(ref)
    ref = ref.iloc[: len(out)]
    got = pd.DataFrame({"open": [b.open for b in out], "high": [b.high for b in out], "low": [b.low for b in out], "close": [b.close for b in out], "volume": [b.volume for b in out]}, index=pd.DatetimeIndex([b.ts for b in out]))
    pd.testing.assert_index_equal(got.index, ref.index, check_names=False)
    for c in ["open", "high", "low", "close", "volume"]:
        np.testing.assert_allclose(got[c].to_numpy(), ref[c].to_numpy(), rtol=1e-12)


def test_aggregator_emits_open_bucket_after_gap():
    agg = BarAggregator(5)
    t0 = pd.Timestamp("2024-01-01 10:01", tz="UTC")
    bars = [Bar("X", t0 + pd.Timedelta(minutes=i), 1, 2, 0.5, 1.5, 10) for i in range(3)]  # 10:01..10:03, bucket 10:05 open
    for b in bars:
        assert agg.push(b) == []
    late = Bar("X", pd.Timestamp("2024-01-01 10:12", tz="UTC"), 1, 1, 1, 1, 1)  # gap; belongs to 10:15 bucket
    out = agg.push(late)
    assert len(out) == 1 and out[0].ts == pd.Timestamp("2024-01-01 10:05", tz="UTC") and out[0].volume == 30


def test_fast_screen_agrees_with_event_engine(btc_bars):
    """Same signal, same costs (25 bps taker + 2.5 half-spread + 1 slip, no impact), 5-min bars,
    long-only crypto. Total return and trade counts must agree closely; residual differences come
    from lot rounding and equity compounding."""
    bars5 = resample(btc_bars, "5min")
    params = dict(window=12, entry_z=1.5, exit_z=0.0, vol_window=48, max_vol_mult=100.0)
    tgt = S.zscore_vwap_reversion(bars5, params["window"], params["entry_z"], params["exit_z"], vol_window=params["vol_window"], max_vol_mult=params["max_vol_mult"], long_only=True)
    cost = CostSpec(25.0, 2.5, 1.0)
    fast = simulate_positions(bars5, tgt, cost, asset_class=AssetClass.CRYPTO, long_only=True)
    assert fast.n_round_trips >= 5, "test needs some trades to be meaningful"

    strat = ZScoreReversion(["BTC/USD"], {**params, "bar_minutes": 5, "alloc_frac": 1.0, "min_delta_frac": 0.0, "session_only": False})
    bt = Backtester(
        [strat],
        {"BTC/USD": to_bar_events(btc_bars, "BTC/USD")},
        fees=CompositeFees(crypto=AlpacaCryptoFees()),
        fill_model=SlippageModel(half_spread_bps=2.5, slippage_bps=1.0, impact_coeff_bps=0.0, max_participation=float("inf")),
        initial_cash=100_000.0,
    )
    res = bt.run()
    n_rt_event = len(res.fills) / 2.0
    assert abs(n_rt_event - fast.n_round_trips) <= max(1.0, 0.15 * fast.n_round_trips)
    # equity compounding / lot rounding differ slightly; require agreement within 6% of the move (min 50 bps)
    assert abs(res.metrics["total_return"] - fast.metrics["total_return"]) < max(0.005, 0.06 * abs(fast.metrics["total_return"]))
    assert np.sign(res.metrics["total_return"]) == np.sign(fast.metrics["total_return"])
    assert (res.fills["liquidity"] == "taker").all()
    # fee per fill: 25 bps tier-1 taker, dropping to 22/20 as the SimBroker's rolling 30-day
    # volume crosses Alpaca's $100k/$500k tiers (the fast screen holds tier 1 fixed = conservative)
    assert 12.0 <= res.metrics["cost_per_trade_bps"] <= 25.0
