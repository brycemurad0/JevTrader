"""LIVE-mode construction must be refused unless BOTH the operator confirmation env var is set
AND every strategy_id passed in has a passing LIVE promotion record."""

from __future__ import annotations

import pytest

from jevtrader.core.broker import TradingMode
from jevtrader.execution.alpaca_broker import AlpacaBroker, LiveTradingNotAllowed
from jevtrader.live import promotion
from tests.execution.fakes import FakeTradingClient, make_live_settings, make_settings

BACKTEST_REPORT = {
    "oos_sharpe_after_fees": 1.5, "dsr_prob": 0.97, "max_drawdown": 0.1, "n_trades": 100, "fee_stress_pass": True,
}
PAPER_SUMMARY = {"trading_days": 25, "n_trades": 50, "sharpe": 0.8, "max_drawdown": 0.1, "cost_bps": 12.0}


def test_live_construction_raises_without_confirm(tmp_path):
    settings = make_settings(tmp_path, live_confirm="")
    with pytest.raises(LiveTradingNotAllowed, match="JEV_LIVE_CONFIRM"):
        AlpacaBroker(TradingMode.LIVE, client=FakeTradingClient(), settings=settings, strategy_ids=["cand1"])


def test_live_construction_raises_without_strategy_ids(tmp_path):
    settings = make_live_settings(tmp_path)
    with pytest.raises(LiveTradingNotAllowed, match="strategy_id"):
        AlpacaBroker(TradingMode.LIVE, client=FakeTradingClient(), settings=settings, strategy_ids=())


def test_live_construction_raises_without_promotion_record(tmp_path):
    settings = make_live_settings(tmp_path)
    with pytest.raises(promotion.PromotionError):
        AlpacaBroker(TradingMode.LIVE, client=FakeTradingClient(), settings=settings, strategy_ids=["never_promoted"])


def test_live_construction_succeeds_with_passing_promotion(tmp_path):
    settings = make_live_settings(tmp_path)
    candidate_id = "meanrev:AAPL"
    assert promotion.promote_to_paper(candidate_id, BACKTEST_REPORT, settings=settings).passed
    assert promotion.promote_to_live(candidate_id, PAPER_SUMMARY, settings=settings).passed

    broker = AlpacaBroker(TradingMode.LIVE, client=FakeTradingClient(), settings=settings, strategy_ids=[candidate_id])
    assert broker.mode is TradingMode.LIVE


def test_live_construction_raises_if_only_paper_promoted(tmp_path):
    settings = make_live_settings(tmp_path)
    candidate_id = "meanrev:MSFT"
    assert promotion.promote_to_paper(candidate_id, BACKTEST_REPORT, settings=settings).passed
    with pytest.raises(promotion.PromotionError):
        AlpacaBroker(TradingMode.LIVE, client=FakeTradingClient(), settings=settings, strategy_ids=[candidate_id])
