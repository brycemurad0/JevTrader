# 02b - BTC mean reversion with MAKER entries and Alpaca fee tiers (event-driven)

_Generated 2026-09-27._

Sample: validation + holdout, 2026-01-18 -> 2026-09-27 (361,774 1-min bars), 60-min z-score reversion (window 30, entry 2.0, exit 0.0, vol filter 1.5), long-only, 100% of equity per trade so fee tiers are hit as they would be on a $100k account.

| setup | sharpe | total_return | max_dd | n_fills | maker_share | fee_bps_per_fill | fees_usd | gross_pnl_usd | net_pnl_usd |
|---|---|---|---|---|---|---|---|---|---|
| taker, tier-1 fixed (25 bps) | -3.227 | -0.455 | -0.463 | 162 | 0.000 | 25.000 | 28700.392 | -16838.718 | -45539.110 |
| taker, Alpaca tiers by rolling 30d volume | -2.641 | -0.393 | -0.402 | 162 | 0.000 | 18.320 | 22049.652 | -17201.205 | -39250.857 |
| maker limit entry, tier-1 fixed (15 bps) | -2.640 | -0.392 | -0.401 | 162 | 0.500 | 19.994 | 24119.652 | -15107.460 | -39227.112 |
| maker limit entry, Alpaca tiers | -2.044 | -0.322 | -0.332 | 162 | 0.500 | 13.289 | 16823.412 | -15384.457 | -32207.869 |
| maker limit entry, zero fees (edge check) | -0.695 | -0.136 | -0.238 | 162 | 0.500 | 0.000 | 0.000 | -13627.422 | -13627.422 |

Reading: `gross_pnl_usd` is what the signal makes before fees; if it is near zero or negative even at zero fees, no fee tier rescues it. Maker entries pay 15 instead of 25 bps and avoid the spread, but they fill less often and adversely (the limit sits at the close; it only fills when the next bar trades through it).

**Verdict: NOT VIABLE at any fee tier -- the signal loses money even with zero fees, so no maker/tier improvement can rescue it.**