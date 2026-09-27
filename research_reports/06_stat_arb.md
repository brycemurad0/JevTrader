# 06 - Stat-arb: Kalman pairs (synthetic mechanics) and BTC->SPX lead-lag (real)

_Generated 2026-09-27._

## Pairs with Kalman hedge ratio -- MECHANICS on synthetic cointegrated pairs

No real equity pair data is cached, so this only shows the strategy recovers a known cointegration (beta=1.5, OU spread of a given half-life and vol) and what it earns net of Alpaca equity costs (3.3 bps per leg round trip) on that ground truth. Read the `spread_vol` rows together: a stationary spread std of ~60 bps of price (spread_vol 0.15) leaves room for costs and for the hedge-ratio estimation error (beta error x price level is itself ~0.5-1 dollar on a 100-dollar pair); a ~20 bps spread (spread_vol 0.05) does not and loses money. It says NOTHING about real pairs -- run the validation list below on Alpaca history first.

| half_life | spread_vol | sharpe | total_return | max_dd | n_fills | hit_rate | final_beta |
|---|---|---|---|---|---|---|---|
| 30.0 | 0.150 | 0.875 | 0.007 | -0.013 | 48 | 0.500 | 1.511 |
| 60.0 | 0.150 | 0.514 | 0.004 | -0.014 | 36 | 0.500 | 1.528 |
| 30.0 | 0.050 | -3.858 | -0.008 | -0.009 | 96 | 0.479 | 1.497 |

Real-validation list for the user (Alpaca 15-min bars, >= 1 year, both legs shortable): XLE/XOP, GLD/GDX, QQQ/XLK, KO/PEP, XLF/KBE, IWM/IJR, USO/XLE, MSFT/AAPL. Expect 2 legs x round trip ~ 3-7 bps cost per spread trade on Alpaca; require gross > 15 bps per spread trade to be worth it.

**Verdict: RESEARCH (mechanics validated; no real evidence yet).**

## BTC -> SPX500 lead-lag (real data, US session overlap)

Cross-correlation corr(BTC ret[t-k], SPX ret[t]) at 5-min lags (whole overlap sample):

| lag | corr |
|---|---|
| lag 0 | 0.336 |
| lag 1 | 0.002 |
| lag 2 | 0.010 |
| lag 3 | 0.004 |

### SPX500 CFD 1m as SPY proxy (0.15 fee + 0.25 half-spread + 0.25 slip per side) @ 5min BTC->SPX

Chosen params (best train Sharpe with positive neighbourhood, n_trials=27): `{'lookback': 1, 'threshold_bps': 40.0, 'hold': 3}`

| split | sharpe | total_return | max_drawdown | n_round_trips | trades_per_day | gross_edge_bps_per_rt | cost_bps_per_rt | net_edge_bps_per_rt | breakeven_fee_bps_per_side | exposure |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 1.27 | 0.03 | -0.01 | 158.00 | 1.43 | 2.96 | 1.30 | 1.66 | 0.98 | 0.08 |
| val | -1.39 | -0.00 | -0.01 | 21.00 | 0.60 | 0.10 | 1.30 | -1.20 | -0.45 | 0.03 |
| holdout | -2.90 | -0.01 | -0.01 | 63.00 | 1.74 | -0.02 | 1.30 | -1.32 | -0.51 | 0.09 |

* Neighbourhood mean train Sharpe (adjacent grid points): **0.23**
* Deflated Sharpe (validation, n_trials=27): **0.006**; holdout DSR: 0.001
* Robustness (holdout): base SR -3.04 | fees x1.5 -3.38 | fees x2 -3.72 | slippage x2 -5.29 -> FAIL

Parameter-stability table (train Sharpe):

| lookback | 10.0 | 20.0 | 40.0 |
|---|---|---|---|
| 1 | -5.69 | -3.44 | 1.35 |
| 3 | -3.87 | -2.20 | 0.74 |
| 6 | -2.49 | -2.77 | 0.63 |

**Verdict: NOT VIABLE**
