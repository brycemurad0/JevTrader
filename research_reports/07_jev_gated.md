# 07 - Jev-gated variants (OfflineJevAdvisor stand-in)

_Generated 2026-09-27._

**Read this first.** `OfflineJevAdvisor` is a transparent logistic heuristic with no forecasting skill. These runs show how the gate changes *turnover and sizing* (it can only remove entries and shrink size via fractional Kelly), not whether Jev adds alpha. The real test is `ReplayJevAdvisor.from_log(...)` over recorded paper-trading decisions plus the Kev/Laya/Jev calibration scoreboard (`jevtrader.jev.calibration`).

|  | sharpe | total_return | max_dd | n_fills | jev_holds | avg_fill_notional |
|---|---|---|---|---|---|---|
| SPX zscore_reversion 15m [base] | -0.497 | -0.004 | -0.015 | 88 | 0 | 45658.752 |
| SPX zscore_reversion 15m [jev-gated (offline)] | -0.054 | -0.000 | -0.003 | 8 | 133 | 45842.925 |
| SPX trend_follow 60m EMA [base] | -0.501 | -0.009 | -0.025 | 36 | 0 | 90070.536 |
| SPX trend_follow 60m EMA [jev-gated (offline)] | -0.164 | -0.002 | -0.016 | 24 | 801 | 45805.523 |
| C zscore_reversion 15m [base] | 0.725 | 0.015 | -0.038 | 63 | 0 | 49946.074 |
| C zscore_reversion 15m [jev-gated (offline)] | -0.629 | -0.003 | -0.016 | 10 | 93 | 49899.344 |
| C trend_follow 60m EMA [base] | -0.227 | -0.011 | -0.035 | 152 | 0 | 7228.519 |
| C trend_follow 60m EMA [jev-gated (offline)] | -1.522 | -0.023 | -0.039 | 27 | 240 | 11224.637 |

Columns: `jev_holds` = entries the gate suppressed; `avg_fill_notional` shows the Kelly down-sizing. Both variants run on validation+holdout (out-of-sample for the base parameters).

**Verdict: MECHANICS OK (falls back to base with jev=None; late/None = HOLD). Alpha unproven by construction -- run paper with DecisionLog on, then `experiments.jev_gate_replay(log_path, ...)`.**