# 08 - Meta-allocator (inverse-vol / HRP across strategy sleeves)

_Generated 2026-09-27._

Runs `zscore_reversion` (15m) and `trend_follow` (60m EMA) as sleeves on the SPX500 proxy and C over validation+holdout, with weights re-estimated from the sleeves' own virtual P&L. Compared with equal weights. This is plumbing: the allocator cannot create edge, only distribute risk.

| scheme | sharpe | total_return | max_dd | n_fills | final_weights |
|---|---|---|---|---|---|
| equal_weight | -0.084 | -0.003 | -0.031 | 208 | {'zscore_reversion:SPY': 0.33, 'zscore_reversion:C': 0.33, 'trend_follow:SPY': 0.33} |
| inverse_vol | 0.073 | 0.002 | -0.025 | 239 | {'zscore_reversion:SPY': 0.47, 'zscore_reversion:C': 0.24, 'trend_follow:SPY': 0.29} |
| hrp | 0.237 | 0.006 | -0.025 | 268 | {'zscore_reversion:SPY': 0.54, 'zscore_reversion:C': 0.1, 'trend_follow:SPY': 0.36} |

**Verdict: plumbing validated (virtual per-child positions, weight-scaled orders, scheme weights from `rebalance.targets`). Useful once >= 2 sleeves are individually VIABLE; today it mostly adds risk control.**