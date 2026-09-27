"""The backtest -> paper -> live promotion gate: pure pass/fail logic plus record persistence."""

from __future__ import annotations

import pytest

from jevtrader.config import LIVE_CONFIRM_PHRASE, Settings
from jevtrader.live import promotion


def _settings(tmp_path) -> Settings:
    return Settings(
        alpaca_key="k", alpaca_secret="s", alpaca_paper=True, alpaca_live_key="", alpaca_live_secret="",
        alpaca_data_feed="iex", typesafe_api_key="", jev_model="jev-latest", jev_latency_budget_ms=400,
        live_confirm=LIVE_CONFIRM_PHRASE, data_dir=tmp_path / "data", runs_dir=tmp_path / "runs",
        state_dir=tmp_path / "state",
    )


GOOD_BACKTEST = {"oos_sharpe_after_fees": 1.5, "dsr_prob": 0.97, "max_drawdown": 0.10, "n_trades": 100, "fee_stress_pass": True, "cost_bps": 10.0}
GOOD_PAPER = {"trading_days": 25, "n_trades": 40, "sharpe": 0.8, "max_drawdown": 0.12, "cost_bps": 14.0}


# --------------------------------------------------------------------------------- pure gate logic

def test_evaluate_backtest_passes():
    result = promotion.evaluate_backtest(GOOD_BACKTEST)
    assert result.passed
    assert result.reasons == []
    assert bool(result) is True


@pytest.mark.parametrize("field,bad_value", [
    ("oos_sharpe_after_fees", 0.1),
    ("dsr_prob", 0.5),
    ("max_drawdown", 0.9),
    ("n_trades", 2),
])
def test_evaluate_backtest_fails_individual_thresholds(field, bad_value):
    report = {**GOOD_BACKTEST, field: bad_value}
    result = promotion.evaluate_backtest(report)
    assert not result.passed
    assert any(field in r for r in result.reasons)


def test_evaluate_backtest_fails_without_fee_stress_pass():
    report = {**GOOD_BACKTEST, "fee_stress_pass": False}
    result = promotion.evaluate_backtest(report)
    assert not result.passed
    assert any("fee_stress_pass" in r for r in result.reasons)


def test_evaluate_paper_passes():
    result = promotion.evaluate_paper(GOOD_PAPER, GOOD_BACKTEST)
    assert result.passed


def test_evaluate_paper_fails_on_insufficient_trading_days():
    result = promotion.evaluate_paper({**GOOD_PAPER, "trading_days": 3}, GOOD_BACKTEST)
    assert not result.passed
    assert any("trading_days" in r for r in result.reasons)


def test_evaluate_paper_fails_on_cost_deviation_from_backtest():
    # backtest modeled 10bps of cost; paper realized 40bps -> the backtest fill simulator is not
    # representative of reality, so the candidate should not go live.
    result = promotion.evaluate_paper({**GOOD_PAPER, "cost_bps": 40.0}, GOOD_BACKTEST)
    assert not result.passed
    assert any("deviates" in r for r in result.reasons)


def test_evaluate_paper_ignores_cost_check_when_costs_missing():
    backtest = {k: v for k, v in GOOD_BACKTEST.items() if k != "cost_bps"}
    paper = {k: v for k, v in GOOD_PAPER.items() if k != "cost_bps"}
    result = promotion.evaluate_paper(paper, backtest)
    assert result.passed


# --------------------------------------------------------------------------------- persistence

def test_promote_to_paper_writes_record_on_pass(tmp_path):
    settings = _settings(tmp_path)
    result = promotion.promote_to_paper("cand1", GOOD_BACKTEST, settings=settings)
    assert result.passed
    record = promotion.load_promotion("cand1", settings)
    assert record["stage"] == "paper"
    assert record["backtest_metrics"]["oos_sharpe_after_fees"] == GOOD_BACKTEST["oos_sharpe_after_fees"]


def test_promote_to_paper_writes_nothing_on_fail(tmp_path):
    settings = _settings(tmp_path)
    bad = {**GOOD_BACKTEST, "n_trades": 1}
    result = promotion.promote_to_paper("cand1", bad, settings=settings)
    assert not result.passed
    assert promotion.load_promotion("cand1", settings) is None


def test_promote_to_live_requires_prior_paper_promotion(tmp_path):
    settings = _settings(tmp_path)
    result = promotion.promote_to_live("never_promoted", GOOD_PAPER, settings=settings)
    assert not result.passed
    assert "backtest->paper" in result.reasons[0] or "paper" in result.reasons[0]


def test_promote_to_live_full_pipeline(tmp_path):
    settings = _settings(tmp_path)
    promotion.promote_to_paper("cand1", GOOD_BACKTEST, settings=settings)
    result = promotion.promote_to_live("cand1", GOOD_PAPER, settings=settings)
    assert result.passed

    record = promotion.load_promotion("cand1", settings)
    assert record["stage"] == "live"
    assert record["live_started_at"] is not None
    assert record["capital_frac"] == pytest.approx(0.10)


def test_promote_to_live_fails_and_does_not_upgrade_stage(tmp_path):
    settings = _settings(tmp_path)
    promotion.promote_to_paper("cand1", GOOD_BACKTEST, settings=settings)
    bad_paper = {**GOOD_PAPER, "sharpe": -1.0}
    result = promotion.promote_to_live("cand1", bad_paper, settings=settings)
    assert not result.passed
    record = promotion.load_promotion("cand1", settings)
    assert record["stage"] == "paper"  # unchanged


def test_current_capital_frac_scale_schedule(tmp_path):
    settings = _settings(tmp_path)
    promotion.promote_to_paper("cand1", GOOD_BACKTEST, settings=settings)
    promotion.promote_to_live("cand1", GOOD_PAPER, settings=settings)

    # freshly live -> initial fraction
    assert promotion.current_capital_frac("cand1", settings=settings) == pytest.approx(0.10)

    # backdate live_started_at to simulate having been live for 15 days
    record = promotion.load_promotion("cand1", settings)
    import pandas as pd

    record["live_started_at"] = (pd.Timestamp.now("UTC") - pd.Timedelta(days=15)).isoformat()
    promotion._save_promotion(record, settings)
    assert promotion.current_capital_frac("cand1", settings=settings) == pytest.approx(0.25)


def test_current_capital_frac_zero_when_not_live(tmp_path):
    settings = _settings(tmp_path)
    assert promotion.current_capital_frac("never_promoted", settings=settings) == 0.0


def test_assert_live_allowed_raises_without_record(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(promotion.PromotionError):
        promotion.assert_live_allowed("cand1", settings)


def test_assert_live_allowed_passes_with_live_record(tmp_path):
    settings = _settings(tmp_path)
    promotion.promote_to_paper("cand1", GOOD_BACKTEST, settings=settings)
    promotion.promote_to_live("cand1", GOOD_PAPER, settings=settings)
    promotion.assert_live_allowed("cand1", settings)  # must not raise


def test_assert_live_allowed_raises_when_only_paper(tmp_path):
    settings = _settings(tmp_path)
    promotion.promote_to_paper("cand1", GOOD_BACKTEST, settings=settings)
    with pytest.raises(promotion.PromotionError):
        promotion.assert_live_allowed("cand1", settings)


def test_list_promotions(tmp_path):
    settings = _settings(tmp_path)
    promotion.promote_to_paper("cand1", GOOD_BACKTEST, settings=settings)
    promotion.promote_to_paper("cand2", GOOD_BACKTEST, settings=settings)
    records = promotion.list_promotions(settings=settings)
    assert {r["candidate_id"] for r in records} == {"cand1", "cand2"}


def test_candidate_id_sanitized_for_filesystem(tmp_path):
    settings = _settings(tmp_path)
    promotion.promote_to_paper("meanrev:AAPL,MSFT", GOOD_BACKTEST, settings=settings)
    assert promotion.load_promotion("meanrev:AAPL,MSFT", settings) is not None


def test_load_thresholds_merges_yaml_file(tmp_path):
    path = tmp_path / "promotion.yaml"
    path.write_text("backtest:\n  min_trades: 5\n")
    th = promotion.load_thresholds(path)
    assert th["backtest"]["min_trades"] == 5
    # untouched keys fall back to defaults
    assert th["backtest"]["min_oos_sharpe_after_fees"] == promotion.DEFAULT_THRESHOLDS["backtest"]["min_oos_sharpe_after_fees"]


def test_load_thresholds_falls_back_to_defaults_when_file_missing(tmp_path):
    th = promotion.load_thresholds(tmp_path / "does_not_exist.yaml")
    assert th == promotion.DEFAULT_THRESHOLDS
