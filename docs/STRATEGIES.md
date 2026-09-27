# Strategy catalogue, evidence and verdicts

This is the honest state of the JevTrader strategy library as of 2026-09-27. Every strategy below is
implemented in `jevtrader/strategies/`, registered under the name in the heading, unit-tested in
`tests/strategies/`, and evaluated by `jevtrader/research/experiments.py` on the real data in
`data/cache/` (Bitstamp BTC/USD 1-min 2025-01 -> 2026-09; SPX500 CFD 1-min and Citigroup 1-min
2026-03 -> 2026-09). Numbers are in `research_reports/`; regenerate with
`python -m jevtrader.research.experiments`.

**TL;DR**

| # | strategy | markets | frequency | verdict at Alpaca fees | one-line reason |
|---|---|---|---|---|---|
| 1 | `as_market_maker` (Avellaneda-Stoikov) | crypto / equity | quotes | crypto **NOT VIABLE**; equity **RESEARCH** | 15 bps maker fee forces a >= 35 bps quoted spread on BTC; equity mechanics fine, edge unproven without real L2 |
| 2 | `imbalance_alpha` (OBI / microprice) | crypto / equity | quotes/books | TAKE branch **NOT VIABLE** anywhere; JOIN branch **RESEARCH** (equity) | predicted moves are sub-bp; only maker fills on zero-commission equities can carry it |
| 3 | `vpin_monitor` / `FlowToxicityGate` | any | trades | filter, not alpha | validate by conditioning maker fills on VPIN during paper |
| 4 | `zscore_reversion` (VWAP z-score MR) | equity (crypto) | 5-30 min | **MARGINAL** on single stocks (C); **NOT VIABLE** on SPX proxy and on BTC | C: +45..88 bps gross per round trip OOS vs 3.3 bps cost, but only ~12 trades per split; BTC gross 5 bps vs 57 bps cost |
| 5 | `range_breakout` (ORB / session) | equity / crypto | 1 min | **NOT VIABLE** | negative or ~zero gross edge everywhere OOS |
| 6 | `trend_follow` (EMA / Donchian, vol-targeted) | equity / crypto | 15 min - 4 h | **NOT VIABLE** (intraday) | flips too often; BTC 4h EMA gross 99 bps/rt on holdout but negative on train/val, 0.07 trades/day |
| 7 | `vol_squeeze` | equity / crypto | 5 min | **NOT VIABLE** | direction after the squeeze is a coin flip after costs |
| 8 | `time_of_day` (seasonality) | equity / crypto | 15 min | **NOT VIABLE** | no BTC hour survives 24-way multiple testing; C first/last-hour effect died on holdout |
| 9 | `pairs_kalman` | equity | 15 min | **RESEARCH** | mechanics validated on synthetic cointegration; no real pair data cached |
| 10 | `lead_lag` (BTC -> SPX) | equity | 5 min | **NOT VIABLE** | corr(BTC[t], SPX[t]) = 0.34 but corr at lag 1 = 0.002: no lead |
| 11 | `*_jev` variants | as base | as base | **MECHANICS OK**, alpha untested | OfflineJevAdvisor has no skill by design; real test = ReplayJevAdvisor on paper decision logs |
| 12 | `meta_allocator` | portfolio | 15 min | plumbing validated | inverse-vol / HRP across sleeves; needs >= 2 viable sleeves to matter |
| -- | `smart_rebalance` (jevtrader/rebalance) | equity + crypto | daily | the only thing that *should* run on crypto at tier-1 fees | low turnover is the only way to beat a 30-50 bps round trip |

Nothing in the library is **VIABLE** (>= 1.0 out-of-sample Sharpe, DSR >= 0.95, positive net edge,
fee-stress pass) by the promotion gate's own standard. That is the truthful result of testing on
~1.7 years of BTC and ~6 months of equity 1-min data, and it is what one should expect: the
promotion gate is deliberately strict, the equity samples are short, and Alpaca's crypto fees kill
every intraday strategy before it starts.

---

## 1. The HFT question, answered plainly

The user asked for "the most quant-based backtested strategies for HFT". Here is what the numbers
say about HFT from a retail Alpaca account.

**Costs (from `jevtrader/core/fees.py`, verified in tests):**

| venue | fee per side | round trip (2 x fee) | + spread/slippage model | total round trip assumed in research |
|---|---|---|---|---|
| Alpaca crypto tier 1 taker | 25 bps | 50 bps | 2 x (2.5 + 1.0) | **57 bps** |
| Alpaca crypto tier 1 maker | 15 bps | 30 bps | 0 (you earn the spread) | **30 bps** |
| Alpaca crypto tier 5/6 (>= $10M / $25M per 30 days) | 5 / 2 bps taker | 10 / 4 bps | 7 bps | 17 / 11 bps |
| Alpaca US equity, liquid ETF (SPY-like) | ~0.15 bps (SEC/TAF on sells only) | 0.3 bps | 2 x (0.25 + 0.25) | **1.3 bps** |
| Alpaca US equity, single stock (C-like) | ~0.15 bps | 0.3 bps | 2 x (1.0 + 0.5) | **3.3 bps** |

**What the market offers per trade at intraday horizons (real data, gross of costs):**

* BTC 1-min realized vol is 6 bps/bar; 5-min 14 bps; 60-min 47 bps. Linear autocorrelation of BTC
  returns is ~0 at 1, 5, 15 and 60 minutes. The best gross edge any bar strategy extracted from
  BTC in-sample was 5-15 bps per round trip; out-of-sample 5 bps at 5-min MR. Against 30-57 bps
  of cost that is a guaranteed loss. **Break-even fee for BTC intraday strategies: 1-7 bps per
  side** (fee ladder tables in `research_reports/02..04`), i.e. Alpaca's top tiers or a venue with
  maker rebates. Even then the edge is thin.
* Equities: 1-min strategies are dead on arrival too (0.1-0.5 bps gross per trade vs 1.3-3.3 bps
  cost). 5-30 min mean reversion on a single stock produced 20-90 bps gross per round trip -- the
  one place where retail costs leave room -- but on ~12 trades per out-of-sample split, which is
  why it is MARGINAL, not VIABLE.
* Order-book imbalance (the best-documented short-horizon microstructure signal) has an IC of a
  few percent at 1-10 s: roughly 0.1-0.5 bps of expected move per event. No taker fee anywhere is
  that small. As a maker it lives on queue position, which a tens-of-ms REST/WebSocket client
  loses to co-located firms.

**Therefore:**

1. True HFT (sub-second, latency-sensitive, thousands of trades a day) is **not achievable** from
   a retail Alpaca account. It requires co-location, direct market access, exchange member fee
   tiers / maker rebates and a sub-100 microsecond stack. None of that is a software problem you
   can solve in this repository.
2. The highest-frequency thing that is *realistically* viable at retail cost is **quote-driven
   market making / imbalance-joining on liquid US equities**, at seconds-to-minutes horizons,
   maker-only, small size. Its edge cannot be proven on the data we have (no real L1/L2); the
   tooling to prove it is here: run `market_recorder` + `vpin_monitor` next to `as_market_maker`
   in paper for a few days, then `research.record_replay` replays the real books through the
   SimBroker and reconciles simulated vs paper fills and costs.
3. On Alpaca **crypto** at tier 1, the fee arithmetic rules out every intraday strategy. Crypto
   should be traded there only as a low-turnover holding (`smart_rebalance`, or a daily/weekly
   trend or MR overlay with a handful of trades per month). If the user wants intraday crypto,
   they need a venue/tier with <= 2-3 bps per side -- and then the strategies in this library
   are the right starting point (they are already parameterised for it and the fee ladder tells
   them where the line is).

---

## 2. Methodology (what every number in the reports means)

* **Splits**: chronological 60 / 20 / 20 (train / validation / holdout) by row count of each 1-min
  data set. Sweeps run on train only. Validation picks/vetoes. Holdout is evaluated once, with the
  parameters already chosen. `research_reports/*` print all three so you can see the decay.
* **Selection**: best train Sharpe subject to >= 30 train round trips and a *positive
  neighbourhood* (mean Sharpe of the adjacent grid points). `n_trials` (grid size) is recorded and
  the **Deflated Sharpe Ratio** (`jevtrader.backtest.metrics.deflated_sharpe_ratio`) of the
  validation return stream is reported against it. With 80 trials and ~12 trades, DSR is
  necessarily tiny -- this is the correct statistical humility, not a bug.
* **Costs**: fast screen = `fee + half_spread + slippage` per side, charged on turnover, fills at
  the *next bar's open* (identical to the SimBroker's bar-mode rule). Event-driven confirmation
  = the registered `Strategy` class through `Backtester` + `SimBroker` with `CompositeFees`
  (Alpaca defaults, including the rolling 30-day crypto volume tiers) and the same spread/slippage
  model. `tests/strategies/test_aggregator_fastscreen.py` checks the two paths agree.
* **Stress**: fees x1.5, fees x2, slippage x2 (`robustness_report` mirror) and, for crypto, a
  fee ladder at 25 / 15 / 10 / 5 / 2 / 0 bps per side -> **break-even fee** = the per-side fee at
  which net expectancy is zero given the observed gross edge.
* **The single most important number**: `gross_edge_bps_per_rt` vs `cost_bps_per_rt`. If gross
  is not comfortably above cost *out of sample*, nothing else in the table matters.
* **Verdict labels**: VIABLE = OOS Sharpe >= 1.0 on both val and holdout, DSR >= 0.95, positive
  net edge, stress pass (the promotion gate's bar). MARGINAL = positive OOS Sharpe > 0.3 and
  positive net edge but fails the rest. NOT VIABLE = the rest. RESEARCH = mechanics validated on
  synthetic data only.

Data caveats: SPX500 is a CFD proxy for SPY (no spread; a conservative SPY-like cost was assumed);
Citigroup volume is a sampled feed (participation caps disabled; only relative volume used);
the equity samples are 6 months (train 3.7 months). Bitstamp is the BTC price reference; Alpaca's
BTC spread is usually wider than Bitstamp's, so the crypto cost model is if anything optimistic.

---

## 3. Catalogue

### 3.1 `as_market_maker` -- Avellaneda-Stoikov market maker (A.1)

* **What**: quotes bid/ask around a reservation price `r = mid - q*gamma*sigma^2*T` with spread
  `gamma*sigma^2*T + (2/gamma) ln(1+gamma/k)`, plus a microprice/imbalance skew, inventory caps
  (stops quoting the side that adds inventory beyond `max_inventory_units`), optional VPIN gate,
  cancel/replace throttling, post-only limit orders. **Hard rule**: quoted half-spread >=
  maker fee + `adverse_selection_bps` + `min_edge_bps`.
* **Why it might work**: earns the spread on uninformed two-way flow; the skew leans inventory
  back to zero.
* **When it fails**: informed / one-directional flow (fills are adverse), fee floor forcing quotes
  away from the touch, latency (we are last in queue).
* **Fee sensitivity**: on BTC at 15 bps maker the floor is >= 17.5 bps half-spread = 35 bps quoted
  spread on a market whose inside spread is 1-3 bps -> **NOT VIABLE**; break-even needs <= ~3 bps
  maker fee. On equities (0.15 bps) the floor is ~2.7 bps, i.e. it can quote at or inside a
  1-cent spread on a $50-500 stock -> **RESEARCH** until validated on recorded books.
* **Evidence**: `research_reports/01_microstructure_hft.md`: fee arithmetic table + synthetic L1
  runs (gross positive, fee-eaten at crypto tiers). Tests check the floor, two-sided quoting,
  maker-only fills, long-only inventory on crypto.
* **Starting params (equities)**: `quote_size_notional=200-500`, `max_inventory_units=3`,
  `gamma=0.05`, `horizon_s=30-60`, `adverse_selection_bps=1-2`, `vpin_gate=False` until validated.

### 3.2 `imbalance_alpha` -- order-book imbalance / microprice alpha (A.2)

* **What**: EW-smoothed top-`depth` imbalance; expected move `beta_bps * I` over `hold_ticks`
  with `beta` fitted online (recursive least squares with forgetting) and a t-stat guard. TAKE
  (IOC) only when the predicted move > taker round-trip cost; otherwise JOIN the touch as maker
  when |I| > `threshold_join` (no tick-by-tick chasing: re-post only when > `requote_ticks` away).
  Exits after `hold_ticks` or on signal flip. Optional VPIN and Jev gates.
* **Why it might work**: L2 imbalance is the best-documented short-horizon predictor
  (Cont-Kukanov-Stoikov 2014).
* **When it fails**: the edge per event is 0.1-0.5 bps; any taker fee kills the TAKE branch, and
  the JOIN branch depends on queue position and adverse selection.
* **Evidence**: `01_microstructure_hft.md`; tests show the online beta recovers the synthetic
  stream's positive lagged-OFI edge, that L2 books are handled, and that Alpaca crypto fees turn
  a profit into a loss. **Real edge unproven** -- record/replay on paper.
* **Starting params (equities)**: `hold_ticks=10`, `threshold_join=0.3`, `calibrate=True`,
  `min_t_stat=2.0`, `notional=200-500`.

### 3.3 `vpin_monitor` / `FlowToxicityGate` -- VPIN toxicity filter (A.3)

* **What**: volume-bucketed, bulk-volume-classified VPIN; `gate.toxic` when VPIN > threshold.
  `vpin_monitor` never trades; it logs the series for paper journals.
* **Use**: `vpin_gate=True` in `as_market_maker` / `imbalance_alpha` pulls quotes during toxic
  flow. **Validation plan**: regress maker fill markouts on VPIN-at-fill from the paper journal
  (`research.record_replay.load_journal`); enable only if the slope is significantly negative.

### 3.4 `zscore_reversion` -- intraday mean reversion to VWAP (B.4)  **MARGINAL (single stocks)**

* **What**: `z = (close - rolling VWAP_w) / std_w`; long when z < -entry_z, short (equities) when
  z > entry_z, exit at |z| < exit_z or `hold_max`; entries off when short-window vol >
  `max_vol_mult` x long-window vol; equities flatten outside 13:30-20:00 UTC; `entry_style`
  market (taker) or limit (maker).
* **Why it might work**: measured negative return autocorrelation at 5-60 min on equities
  (C: -0.09 at 5-15 min, -0.12 at 60 min; SPX: -0.03 to -0.06); dealer inventory effects.
* **When it fails**: trend/news regimes (vol filter), and costs: gross edge is a few bps to a few
  tens of bps per round trip.
* **Fee sensitivity / evidence** (`02_intraday_mean_reversion.md`):
  * BTC 5/15/60 min: OOS gross **+5 / -30 / -52 bps** per round trip vs **57 bps** cost ->
    **NOT VIABLE**; break-even fee <= 0. Fee ladder: still negative at 0 bps at 15/60 min.
    `02b_btc_maker_entry.md` re-runs the 60-min version event-driven over val+holdout (Jan-Sep
    2026, 162 fills) with maker limit entries and Alpaca's real volume tiers: taker tier-1 -45.5%,
    taker with tiers -39.3%, maker tier-1 -39.2%, maker with tiers -32.2%, and **-13.6% at zero
    fees** -- the signal itself loses on this sample, so no fee tier or order type rescues it.
  * SPX500 proxy 5/15/30 min: train Sharpe 1.8-2.3 -> validation negative -> **NOT VIABLE**. The
    in-sample edge did not survive (small gross edge, 1.3 bps cost).
  * Citigroup 15 min (window 90, entry 1.5, exit 0.5, vol filter 1.5): train SR 2.6 / val 2.1 /
    holdout 5.8; gross **50 / 45 / 88 bps** per round trip vs **3.3 bps** cost; neighbourhood
    train Sharpe 2.2 (every cell of the 5x4 stability table is positive); stress pass; event-driven
    confirmation +13.4% over val+holdout (fast path +15.8%), 35 fills. **But**: 11-13 round trips
    per split, DSR(val) = 0.04 at n_trials = 80. -> **MARGINAL**: promising, statistically
    unproven. It is the best candidate in the library and the one to paper-trade first.
* **Starting params (equities)**: `bar_minutes=15, window=30-90, entry_z=1.5-2.0, exit_z=0.5,
  max_vol_mult=1.5, alloc_frac=0.25, session_only=True`. Run on 3-6 liquid, volatile single
  names (not SPY) to get the trade count up.

### 3.5 `range_breakout` -- opening-range / session breakout (B.5)  **NOT VIABLE**

* Range = first `range_minutes` after the session open; long above / short below; one trade per
  session; flat after `hold_minutes`. BTC anchors at Asia/EU/US opens.
* Evidence (`04_breakouts.md`): every BTC anchor negative gross; SPX ORB OOS gross -2 bps vs 1.3
  cost; C ORB -8 bps vs 3.3. Kept as a reproducible negative and a Jev-gate testbed.

### 3.6 `trend_follow` -- EMA crossover / Donchian with vol targeting (B.6)  **NOT VIABLE (intraday)**

* Evidence (`03_trend_following.md`): 14 dataset x horizon cells; none positive on both val and
  holdout. BTC 4h EMA: holdout gross 99 bps/rt (break-even 46 bps) but negative train/val and
  0.07 trades/day -- not evidence. SPX 15-60 min EMA looked great in-sample (SR 2.5-3.0) because
  the training window was a bull run; validation/holdout flipped negative. Classic drift artefact.
* If used at all: daily bars, long-only crypto, `fast=10, slow=40-50`, expect < 1 trade/week.

### 3.7 `vol_squeeze` -- Bollinger-inside-Keltner release (B.7)  **NOT VIABLE**

* Negative gross edge on BTC at every horizon and on SPX/C 5-min. Compression is real; direction
  is not predictable after costs.

### 3.8 `time_of_day` -- intraday seasonality (B.8)  **NOT VIABLE**

* Evidence (`05_seasonality.md`): BTC max |t| over 24 hours = 2.3 (hour 22 UTC) -> noise after
  multiple testing; holding it lost money in every split. SPX proxy: no hour with |t| > 2;
  overnight drift +10 bps/day (t = 2.1, n = 130) vs intraday +2.7 -- a well-known effect but a
  daily round trip costs 1.3 bps and the sample is 6 months. C: first/last hour t = 2.1 / 3.3 on
  train, +11 bps/rt on val, -4 bps/rt on holdout -> dead.

### 3.9 `pairs_kalman` -- cointegration pairs with Kalman hedge ratio (C.9)  **RESEARCH**

* State [beta, alpha] random walk; z = innovation / EW std of innovations (not the filter's
  inflated S); dollar-neutral legs; equities only.
* Evidence (`06_stat_arb.md`, tests): recovers beta = 1.5 (+/-0.03) on synthetic pairs; at
  Alpaca equity costs it earns +0.7% / 60 days (SR 0.9, 24 spread trades) when the stationary
  spread std is ~60 bps of price, and **loses** (-0.8%) when it is ~20 bps, because the
  hedge-ratio estimation error (beta error x price level) is itself ~50-100 bps of noise on a
  100-dollar pair. Lesson baked into the defaults: `delta=1e-7` (a larger state-noise ratio
  made the filter absorb the spread and destroyed the signal -- corr(innovation, true spread)
  fell from 0.63 to 0.17 at `delta=1e-5`). **No real pair data cached**. Validation list for the
  user's Alpaca history pull: XLE/XOP, GLD/GDX, QQQ/XLK, KO/PEP, XLF/KBE, IWM/IJR, USO/XLE,
  MSFT/AAPL (>= 1 year of 15-min bars; require > 15 bps gross per spread trade to cover ~5-7 bps
  of two-leg costs, and a spread std well above the hedge error).

### 3.10 `lead_lag` -- BTC -> SPX (C.10)  **NOT VIABLE**

* corr(BTC[t], SPX[t]) = 0.34 at 5-min in the US session but corr(BTC[t-1], SPX[t]) = 0.002:
  the assets move together, BTC does not lead. Train SR 1.3 -> val/holdout negative.

### 3.11 `zscore_reversion_jev`, `trend_follow_jev`, `range_breakout_jev` (D)

* Same classes with `jev_gate=True`: before each entry, `build_state(..., cost_bps=round-trip
  cost)` -> `ctx.jev.direction(symbol, state, horizon)`; enter only if P(favourable move > cost)
  >= `jev_min_p` (0.55); `regime()` risk-off veto; size = `kelly_from_jev` (half-Kelly, capped at
  `alloc_frac`). `ctx.jev is None` -> identical to base (tested bit-for-bit); late/None view ->
  HOLD (tested).
* Evidence (`07_jev_gated.md`): with `OfflineJevAdvisor` the gate removed 85-95% of entries on
  the MR sleeves and ~30-80% on trend; P&L changes are not evidence of anything because the
  offline advisor has no skill by design. **Real test**: paper with `DecisionLog` on, then
  `experiments.jev_gate_replay(log_path, strategy, data)` (ReplayJevAdvisor) and the calibration
  scoreboard in `jevtrader.jev.calibration`.

### 3.12 `meta_allocator` (E)

* Runs child strategies as sleeves with virtual per-child positions; weights from
  `jevtrader.rebalance.targets` (`inverse_vol`, `hrp`, `risk_parity`, `equal_weight`,
  `min_variance`) estimated on the sleeves' own P&L sampled every `sample_minutes`; idle sleeves
  kept at `min_weight`; `capital_frac` leaves room for `smart_rebalance`'s core book.
* Evidence (`08_meta_allocator.md`): plumbing works; with one marginal sleeve it cannot add
  return, only risk control.

---

## 4. Recommended starting portfolio for paper trading

Goal of the first 30 paper days: (a) forward-test the one marginal edge with enough trades to
mean something, (b) collect real microstructure and Jev decision data so the RESEARCH strategies
can be judged, (c) verify the cost model (the promotion gate requires |paper cost - backtest cost|
<= 15 bps). Nothing here has passed the backtest gate, so paper is *research*, not a step to live.

Capital: assume a $100k paper account (the split scales; keep the same fractions).

| sleeve | strategy | symbols | size / capital | risk limits (RiskLimits) | purpose |
|---|---|---|---|---|---|
| Core (60%) | `smart_rebalance` (inverse_vol, monthly, 5/25 bands) | SPY, QQQ, TLT, GLD, BTC/USD (crypto capped 15%) | 60% | `crypto_max_pct_equity=0.15`, `max_position_pct_equity=0.25` | the only sane way to hold crypto at tier-1 fees; provides the base book |
| Alpha 1 (25%) | `zscore_reversion` 15-min, `alloc_frac=0.25` per name, `entry_style="market"` | 4-6 liquid, volatile single stocks (e.g. C, BAC, XOM, NVDA, AMD, JPM) -- **not** SPY | 25% | `max_position_notional_usd=6_000`, `max_daily_loss_pct=0.02`, `max_consecutive_losses=4` | the MARGINAL candidate; needs >= 30 trades and >= 20 trading days for `promote_to_paper` |
| Alpha 1b (0%, shadow) | `zscore_reversion_jev` same params, `DecisionLog` on | same names | 0% (`dry_run` via `alloc_frac=0.001`) or a separate paper account | none | records Jev's direction/regime views for `ReplayJevAdvisor` |
| Micro (10%, tiny) | `as_market_maker` `quote_size_notional=200`, `max_inventory_units=3`; `imbalance_alpha` `notional=200`, `min_t_stat=2.0` | 1-2 liquid equities (SPY, AAPL) | 10% | `max_order_notional_usd=500`, `max_orders_per_minute=20` | generate real maker fills and books; **do not size up** until record/replay shows sim ~ paper |
| Recording (0%) | `market_recorder`, `vpin_monitor` | every symbol above | 0 | none | the data that lets you validate 1-3 and the VPIN gate |
| Cash (5%) | -- | -- | 5% | -- | buffer |

Do **not** run on paper (they are documented negatives): `range_breakout`, `vol_squeeze`,
`time_of_day`, `lead_lag`, intraday `trend_follow`, and any intraday strategy on BTC/USD.

Crypto specifically: BTC/USD only via `smart_rebalance` until the account's 30-day volume tier (or
the venue) brings fees to <= 2-3 bps per side. If you later fund with BTC, the same applies -- the
fee is charged on notional regardless of funding currency.

After 20-30 paper days:

1. `research.record_replay.load_journal(run_dir/events.jsonl)` -> `realized_cost_bps(fills)`;
   compare with the backtest's `cost_per_trade_bps` + spread assumption (must be within 15 bps).
2. Replay recorded books through `as_market_maker`/`imbalance_alpha` (`replay_strategy`) and
   compare fills/PnL with the paper fills (`compare_sim_vs_paper`). Large disagreement = the L1
   queue model is optimistic for Alpaca; scale the market maker down or off.
3. `experiments.jev_gate_replay(decision_log, ZScoreReversionJev(...), data)` vs the base run
   over the same period; only if the gated version wins *and* the calibration scoreboard shows
   Jev's probabilities are calibrated should the `_jev` variant take capital.
4. Re-run `python -m jevtrader.research.experiments` with Alpaca history added to
   `data/cache/` (more names, longer window) -- the trade counts here are the binding constraint on
   every conclusion.

---

## 5. How to use the research harness

```bash
. .venv/bin/activate
python -m jevtrader.research.experiments            # all studies -> research_reports/*.md
python -m jevtrader.research.experiments 02 c       # one study, one data set
pytest tests/strategies                              # ~40 s, all synthetic
```

* `jevtrader/research/loaders.py` -- cached CSV -> UTC bar-close indexed DataFrames,
  `resample`, `us_session_only`, `chrono_split`, `to_bar_events`.
* `jevtrader/research/signals.py` -- vectorized target-position generators (shared indicator code
  with the strategies, e.g. `zscore_features`).
* `jevtrader/research/fastscreen.py` -- `CostSpec`, `simulate_positions`, `sweep_grid`,
  `fee_ladder`, `robustness`, `verdict`.
* `jevtrader/research/experiments.py` -- `run_bar_study` (train/val/holdout + DSR + stability +
  event confirmation) and the `study_*` entry points.
* `jevtrader/research/record_replay.py` -- `MarketDataRecorder`, `load_market_events`,
  `replay_strategy`, `load_journal`, `realized_cost_bps`, `compare_sim_vs_paper`.

Adding a strategy: subclass `jevtrader.strategies._base.BarStrategy` (bar aggregation, sizing,
cost estimate, Jev gate come for free), implement `on_agg_bar`, write a vectorized twin in
`signals.py`, add a `run_bar_study` call, and let the report decide the verdict -- not the author.
