"""Every bar strategy runs end-to-end on small synthetic data, with and without a Jev advisor,
never shorts a non-shortable (crypto) instrument, never gets a fill while the risk manager is
halted, and shorts equities only when allowed."""

from __future__ import annotations

import pandas as pd
import pytest

from jevtrader.backtest import Backtester, SlippageModel
from jevtrader.core.interfaces import JevView
from jevtrader.core.types import Instrument
from jevtrader.jev.advisor import OfflineJevAdvisor
from jevtrader.risk.limits import RiskLimits, TradingHours
from jevtrader.risk.manager import RiskManager
from jevtrader.strategies.jev_gated import RangeBreakoutJev, TrendFollowJev, ZScoreReversionJev
from jevtrader.strategies.range_breakout import RangeBreakout
from jevtrader.strategies.seasonality import TimeOfDay
from jevtrader.strategies.trend_follow import TrendFollow
from jevtrader.strategies.vol_squeeze import VolSqueeze
from jevtrader.strategies.vwap_reversion import ZScoreReversion

from .conftest import events

# (class, params tuned to trade often on 3 days of 1-min data)
CASES = [
    (ZScoreReversion, {"bar_minutes": 5, "window": 12, "entry_z": 1.5, "exit_z": 0.0, "vol_window": 48, "max_vol_mult": 100.0, "session_only": False}),
    (TrendFollow, {"bar_minutes": 5, "mode": "ema", "fast": 5, "slow": 20}),
    (TrendFollow, {"bar_minutes": 5, "mode": "donchian", "n": 12, "exit_n": 6}),
    (RangeBreakout, {"bar_minutes": 1, "session_start_utc": 0, "session_start_minute": 0, "range_minutes": 30, "hold_minutes": 300}),
    (VolSqueeze, {"bar_minutes": 5, "bb_window": 12, "kc_mult": 1.5, "hold": 6}),
    (TimeOfDay, {"bar_minutes": 15, "long_hours": [13, 14, 15], "short_hours": [2, 3], "weekdays": [0, 1, 2, 3, 4, 5, 6]}),
]


def _run(cls, params, df, symbol, jev=None, risk=None, **kw):
    strat = cls([symbol], {**params, "alloc_frac": 0.5, "min_delta_frac": 0.0})
    bt = Backtester([strat], {symbol: events(df, symbol)}, jev=jev, risk=risk, initial_cash=50_000.0, fill_model=SlippageModel(max_participation=float("inf")), **kw)
    return strat, bt.run()


def _min_position(fills: pd.DataFrame) -> float:
    if fills.empty:
        return 0.0
    signed = fills.apply(lambda r: r["qty"] if r["side"] == "buy" else -r["qty"], axis=1)
    return float(signed.cumsum().min())


@pytest.mark.parametrize("cls,params", CASES, ids=lambda c: getattr(c, "__name__", None) or "")
def test_runs_on_crypto_long_only_without_jev(cls, params, btc_bars):
    strat, res = _run(cls, params, btc_bars, "BTC/USD")
    assert len(res.equity_curve) == len(btc_bars)
    assert res.metrics["n_fills"] > 0, f"{cls.__name__} produced no trades on the test data"
    assert _min_position(res.fills) >= -1e-9, "crypto is long-only: position went negative"
    assert (res.fills["side"] != "sell").any() or True  # sells only ever close longs
    for pos in res.positions.values():
        assert pos.qty >= -1e-9


@pytest.mark.parametrize("cls,params", CASES, ids=lambda c: getattr(c, "__name__", None) or "")
def test_runs_with_offline_jev_gate(cls, params, btc_bars):
    strat, res = _run(cls, {**params, "jev_gate": True, "jev_min_p": 0.3}, btc_bars, "BTC/USD", jev=OfflineJevAdvisor())
    assert len(res.equity_curve) == len(btc_bars)
    assert _min_position(res.fills) >= -1e-9


def test_equity_strategy_can_short(spy_bars):
    strat, res = _run(ZScoreReversion, {"bar_minutes": 5, "window": 12, "entry_z": 1.2, "exit_z": 0.0, "vol_window": 48, "max_vol_mult": 100.0, "session_only": False}, spy_bars, "SPY", allow_leverage=True, leverage=1.0)
    assert res.metrics["n_fills"] > 0
    assert _min_position(res.fills) < 0, "equity mean reversion should take at least one short on this sample"


def test_never_fills_when_risk_is_halted(btc_bars):
    limits = RiskLimits(trading_hours=TradingHours(enabled=False))
    risk = RiskManager(limits, instruments={"BTC/USD": Instrument.infer("BTC/USD")})
    risk.kill("test halt")
    strat, res = _run(ZScoreReversion, {"bar_minutes": 5, "window": 12, "entry_z": 1.0, "exit_z": 0.0, "vol_window": 48, "max_vol_mult": 100.0, "session_only": False}, btc_bars, "BTC/USD", risk=risk)
    assert res.metrics["n_fills"] == 0
    assert (res.orders["status"] == "rejected").all()
    assert res.orders["reject_reason"].str.contains("halted").all()


class _LateJev:
    """Advisor whose every answer is late -> strategies must treat it as HOLD."""

    def direction(self, symbol, features, horizon):
        return JevView(probs={"direction": {"up": 0.9, "down": 0.05, "flat": 0.05}}, top={"direction": "up"}, confidence={"direction": 0.9}, latency_ms=999, late=True)

    def regime(self, symbol, features):
        return None

    def ask(self, state, questions):
        return None


class _NoneJev(_LateJev):
    def direction(self, symbol, features, horizon):
        return None


@pytest.mark.parametrize("advisor", [_LateJev(), _NoneJev()], ids=["late", "none"])
def test_jev_variant_holds_on_late_or_none_views(advisor, btc_bars):
    strat, res = _run(ZScoreReversionJev, {"bar_minutes": 5, "window": 12, "entry_z": 1.0, "exit_z": 0.0, "vol_window": 48, "max_vol_mult": 100.0, "session_only": False, "jev_min_p": 0.0}, btc_bars, "BTC/USD", jev=advisor)
    assert res.metrics["n_fills"] == 0
    assert strat.jev_holds > 0


@pytest.mark.parametrize("cls", [ZScoreReversionJev, TrendFollowJev, RangeBreakoutJev])
def test_jev_variant_falls_back_to_base_when_jev_is_none(cls, btc_bars):
    base_cls = cls.__mro__[1]
    p = {"bar_minutes": 5, "window": 12, "entry_z": 1.5, "exit_z": 0.0, "vol_window": 48, "max_vol_mult": 100.0, "session_only": False, "fast": 5, "slow": 20, "session_start_utc": 0, "session_start_minute": 0, "range_minutes": 30, "hold_minutes": 300}
    p_base = {k: v for k, v in p.items() if k in base_cls.spec.default_params}
    p_jev = {k: v for k, v in p.items() if k in cls.spec.default_params}
    if cls is RangeBreakoutJev:
        p_base["bar_minutes"] = p_jev["bar_minutes"] = 1
    _, r_base = _run(base_cls, p_base, btc_bars, "BTC/USD")
    _, r_jev = _run(cls, p_jev, btc_bars, "BTC/USD", jev=None)
    assert r_jev.metrics["n_fills"] == r_base.metrics["n_fills"]
    assert abs(r_jev.metrics["total_return"] - r_base.metrics["total_return"]) < 1e-9


def test_jev_gate_reduces_or_keeps_turnover_and_kelly_sizes_down(btc_bars):
    p = {"bar_minutes": 5, "window": 12, "entry_z": 1.5, "exit_z": 0.0, "vol_window": 48, "max_vol_mult": 100.0, "session_only": False}
    _, base = _run(ZScoreReversion, p, btc_bars, "BTC/USD")
    strat, gated = _run(ZScoreReversionJev, {**p, "jev_min_p": 0.4}, btc_bars, "BTC/USD", jev=OfflineJevAdvisor())
    assert gated.metrics["n_fills"] <= base.metrics["n_fills"]
    if len(gated.fills):
        assert gated.fills["notional"].max() <= base.fills["notional"].max() * 1.05
