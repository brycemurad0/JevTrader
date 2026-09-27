# Promotion gate: backtest -> paper -> live

JevTrader never lets a strategy trade real money because it "looked good in a backtest". A
candidate (a strategy + symbol set + parameters, identified by a `candidate_id` string -- by
convention `Strategy.id`, e.g. `"meanrev:AAPL,MSFT"`) has to earn each stage:

```
backtest report  --evaluate_backtest-->  paper (forward test, N trading days)  --evaluate_paper-->  live (capital-capped, scaling up)
```

Implemented in `jevtrader/live/promotion.py`; thresholds live in `config/promotion.yaml` (any key
left out falls back to the defaults baked into `promotion.py`).

## Stage 1: backtest -> paper

`promote_to_paper(candidate_id, backtest_report, settings=...)` checks `backtest_report` (the
dict of metrics your walk-forward backtest report produces -- see `jevtrader/backtest/`) against
the `backtest` thresholds:

| metric | default threshold | why |
|---|---|---|
| `oos_sharpe_after_fees` | >= 1.0 | out-of-sample, after `jevtrader/core/fees.py` costs |
| `dsr_prob` | >= 0.95 | deflated Sharpe ratio probability -- guards against overfitting/selection bias across a parameter sweep |
| `max_drawdown` | <= 0.20 | backtest max drawdown |
| `n_trades` | >= 30 | statistical significance floor |
| `fee_stress_pass` | must be `true` | the report must say the strategy still clears its bar under a fee-stress scenario |

On a pass, a promotion record is written to `settings.state_dir/promotions/<candidate_id>.json`
with `"stage": "paper"`. On a fail, nothing is written and `GateResult.reasons` lists exactly
which checks failed (a `GateResult` is falsy when it fails, so `if not promote_to_paper(...): ...`
works).

At this point the strategy is expected to run in **paper mode** through `LiveRunner` +
`AlpacaBroker(mode=TradingMode.PAPER)` for real, writing its forward-test record to
`runs_dir/<run_id>/events.jsonl` (see `jevtrader.live.runner.Journal`).

## Stage 2: paper -> live

Once you have enough paper-trading history, summarize the run's journal into a `paper_summary`
dict (`trading_days`, `n_trades`, `sharpe`, `max_drawdown`, and `cost_bps` -- realized fees +
spread + slippage in bps of notional) and call:

```python
promote_to_live(candidate_id, paper_summary, settings=settings)
```

Checked against the `paper` thresholds:

| metric | default threshold | why |
|---|---|---|
| `trading_days` | >= 20 | enough calendar time to see different regimes |
| `n_trades` | >= 30 | statistical significance floor, again |
| `sharpe` | >= 0.5 | realized paper Sharpe (lower bar than backtest -- paper is noisier and shorter) |
| `max_drawdown` | <= 0.20 | realized paper drawdown |
| `\|paper cost_bps - backtest cost_bps\|` | <= 15 bps | **paper-vs-backtest cost/slippage must agree.** If paper trading's realized costs are far higher than the backtest's fill simulator assumed, the backtest is not trustworthy and the strategy does not go live no matter how good its Sharpe looks. |

This requires the candidate to already hold a `stage="paper"` (or `"live"`) record --
`promote_to_live` refuses a candidate that skipped stage 1.

On a pass, the record is upgraded to `"stage": "live"`, `live_started_at` is stamped (once --
re-running `promote_to_live` on an already-live candidate does not reset the clock), and
`capital_frac` is set to the `live.initial_capital_frac` threshold (default **10%** of intended
size).

## Stage 3: live, with a scale-up schedule

`current_capital_frac(candidate_id, settings=settings)` returns the fraction of intended size the
candidate should currently be sized at, as a step function of days since `live_started_at`, per
`live.scale_schedule` in `config/promotion.yaml` (default: 10% at day 0, 25% at day 10, 50% at day
20, 100% at day 30). Strategies/sizing code should multiply their intended notional by this
fraction rather than assuming full size the moment they go live.

## The safety gate `AlpacaBroker` actually enforces

`AlpacaBroker.__init__(mode=TradingMode.LIVE, ..., strategy_ids=[...])` raises
`LiveTradingNotAllowed` unless **both**:

1. `settings.live_allowed()` is `True` -- i.e. `JEV_LIVE_CONFIRM=I_UNDERSTAND_THIS_IS_REAL_MONEY`
   is set in the environment (see `docs/ARCHITECTURE.md`), and
2. every id in `strategy_ids` passes `assert_live_allowed(strategy_id, settings)` -- i.e. has a
   `stage="live"` promotion record from step 2 above.

Constructing a LIVE broker with an empty `strategy_ids` is refused outright: there is no such
thing as "live trading nothing in particular". This means the promotion pipeline above is not
just a paper-trail -- it is a hard runtime precondition for ever routing a real order.

## Record format

`settings.state_dir/promotions/<candidate_id>.json` (candidate id sanitized: `/`, `:`, `,`, and
spaces become `_`):

```json
{
  "candidate_id": "meanrev:AAPL,MSFT",
  "stage": "live",
  "promoted_at": "2026-08-01T00:00:00+00:00",
  "backtest_metrics": { "oos_sharpe_after_fees": 1.4, "dsr_prob": 0.97, "...": "..." },
  "paper_metrics": { "trading_days": 25, "n_trades": 61, "sharpe": 0.8, "...": "..." },
  "live_started_at": "2026-08-15T00:00:00+00:00",
  "capital_frac": 0.10,
  "notes": ""
}
```

`list_promotions(settings=settings)` returns every stored record (e.g. for a promotion dashboard
or CLI report).
