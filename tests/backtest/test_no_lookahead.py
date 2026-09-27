"""Verifies the core no-lookahead guarantee in both fill models: an order can never be filled
using the same bar (bar-mode) or before `latency_ms` has elapsed (quote-mode) that it was
created in reaction to."""

import pandas as pd

from jevtrader.backtest import Backtester, SlippageModel
from jevtrader.core.strategy import Strategy, StrategySpec
from jevtrader.core.types import Order, OrderType, Side
from jevtrader.data.synthetic import bars_to_events, generate_bars, generate_quote_trade_stream, quotes_to_events

EQ_START = pd.Timestamp("2024-01-02 14:30", tz="UTC")


class _BuyOnceOnBar(Strategy):
    spec = StrategySpec(name="_test_buy_once_bar", description="", asset_classes=("equity",), frequency="1min", style="momentum")

    def on_start(self, ctx):
        self.done = False

    def on_bar(self, bar, ctx):
        if not self.done:
            ctx.submit(Order(symbol=bar.symbol, side=Side.BUY, qty=10, type=OrderType.MARKET))
            self.done = True


def test_bar_mode_market_order_fills_at_next_bar_never_the_creation_bar():
    end = EQ_START + pd.Timedelta(hours=1, minutes=30)
    df = generate_bars("AAPL", EQ_START, end, freq="1min", seed=7)
    data = {"AAPL": list(bars_to_events(df, "AAPL"))}
    strat = _BuyOnceOnBar(["AAPL"])
    zero_cost = SlippageModel(half_spread_bps=0, slippage_bps=0, impact_coeff_bps=0)
    bt = Backtester([strat], data, initial_cash=100_000, fill_model=zero_cost)
    result = bt.run()

    assert len(result.fills) == 1
    fill = result.fills.iloc[0]
    order_row = result.orders.iloc[0]
    assert order_row["created_ts"] == df.index[0]  # submitted while reacting to the first bar
    assert fill["ts"] == df.index[1]  # but only fills on the bar AFTER that
    assert abs(fill["price"] - df.iloc[1]["open"]) < 1e-6  # zero cost -> exact next-bar open


class _BuyOnceOnQuote(Strategy):
    spec = StrategySpec(name="_test_buy_once_quote", description="", asset_classes=("equity",), frequency="tick", style="momentum")

    def on_start(self, ctx):
        self.done = False

    def on_quote(self, quote, ctx):
        if not self.done:
            ctx.submit(Order(symbol=quote.symbol, side=Side.BUY, qty=1, type=OrderType.MARKET))
            self.done = True


def test_quote_mode_market_order_never_fills_before_latency_elapses():
    end = EQ_START + pd.Timedelta(minutes=1)
    q, _ = generate_quote_trade_stream("AAPL", EQ_START, end, freq="1s", seed=3)
    data = {"AAPL_quotes": list(quotes_to_events(q, "AAPL"))}
    strat = _BuyOnceOnQuote(["AAPL"])
    bt = Backtester([strat], data, initial_cash=100_000, latency_ms=2500)
    result = bt.run()

    assert len(result.fills) == 1
    fill_ts = result.fills.iloc[0]["ts"]
    submit_ts = result.orders.iloc[0]["created_ts"]
    assert (fill_ts - submit_ts).total_seconds() >= 2.5
    # and it must not have filled on the very first tick after submission if that tick is < latency away
    assert fill_ts != submit_ts


def test_orders_submitted_mid_run_only_see_future_bars():
    """A second order submitted partway through the run must also skip its own creation bar."""

    class _BuyEveryTenBars(Strategy):
        spec = StrategySpec(name="_test_buy_every_ten", description="", asset_classes=("equity",), frequency="1min", style="momentum")

        def on_start(self, ctx):
            self.count = 0

        def on_bar(self, bar, ctx):
            self.count += 1
            if self.count % 10 == 0:
                ctx.submit(Order(symbol=bar.symbol, side=Side.BUY, qty=1, type=OrderType.MARKET))

    end = EQ_START + pd.Timedelta(hours=1)
    df = generate_bars("AAPL", EQ_START, end, freq="1min", seed=9)
    data = {"AAPL": list(bars_to_events(df, "AAPL"))}
    strat = _BuyEveryTenBars(["AAPL"])
    bt = Backtester([strat], data, initial_cash=1_000_000)
    result = bt.run()

    ts_to_pos = {ts: i for i, ts in enumerate(df.index)}
    for _, order_row in result.orders.iterrows():
        matching_fills = result.fills[result.fills["client_order_id"] == order_row["client_order_id"]]
        for _, fill_row in matching_fills.iterrows():
            assert ts_to_pos[fill_row["ts"]] > ts_to_pos[order_row["created_ts"]]
