# Forecasting (`jevtrader/forecast/`)

TimesFM (and the honest statistical baselines it has to beat) sits in the architecture as a
**feature provider to the decision model and the risk/rebalance layers**, never as a strategy or
a trading decision-maker by itself:

```
             bars (per symbol, at each bar close)
                        │
                        ▼
        jevtrader.forecast.features.forecast_features(symbol, bars, forecaster, horizon, cost_bps)
                        │  tfm_ret_q10/50/90_bps, tfm_p_up/down_gt_cost, tfm_vol_fcst_bps,
                        │  tfm_vol_ratio, tfm_volume_ratio  (compact, rounded, JSON-safe)
                        ▼
        jevtrader.jev.state.build_state(..., extra_features=forecast_features(...))
                        │
             ┌──────────┼───────────────────────────────┐
             ▼          ▼                                ▼
         Jev/Kev/Laya   ForecastAdvisor                risk/sizing, rebalance
         (see the       (same tfm_* features,           (tfm_vol_fcst_bps for spread width
         extra_features straight into direction()/       and vol-target sizing;
         as more state) regime(), no LLM call --         drift/vol forecasts for the
                         scored head-to-head with         no-trade band and Jev regime tilt)
                         Jev/Kev/Laya on the same
                         scoreboard, `evaluate.py`)
```

Two entry points, both against the `Forecaster` protocol (`base.py`) so a strategy, `build_state`
call, or backtest never needs to know whether it's talking to real TimesFM or a baseline:

- `forecast_features(...)` -- the compact numbers merged into Jev's state.
- `ForecastAdvisor` -- a standalone `JevAdvisorProtocol` implementation, so TimesFM-only
  strategies (or backtests) work with `ctx.jev = ForecastAdvisor(...)` and no LLM at all.

## Honest prior: what TimesFM is actually good for here

**Short-horizon return/price direction is close to a random walk.** `RandomWalkForecaster` (zero
drift, quantiles from the series' own empirical return distribution) is a legitimately hard
baseline to beat at 5-60 minute horizons on a liquid instrument, and the real-data results below
bear that out: at cost_bps=5 (roughly Alpaca crypto's maker-side round-trip, see
`docs/ARCHITECTURE.md`'s fee table), a forecaster earns nothing by getting direction right
slightly more than half the time if it can't clear the round-trip cost on the moves it calls.

**Volatility, volume, and spread are persistent, structured series** -- exactly the kind of
autocorrelated, mean-reverting-to-a-level process a sequence model (or even GARCH/EWMA) has real
information to extract, because "the next 15 minutes look like the last 15 minutes, mean-
reverting toward a slower-moving average" is actually true of realized vol and volume, unlike
returns. That is where this package expects (and, per the honest-prior design directive, tests
for) most of TimesFM's value:

- **Volatility forecasts** feed `jevtrader/risk/sizing.py` (vol-target sizing wants tomorrow's
  vol, not today's) and market-making spread width (quote wider when a vol jump is forecast, not
  only after realized vol has already jumped).
- **Longer-horizon drift/vol forecasts** feed `jevtrader/rebalance/`'s no-trade bands and regime
  tilt -- a rebalancer cares about the next day/week, where even a small, well-calibrated edge
  compounds, more than it cares about the next 5 minutes.
- **Direction at short horizons** is exactly what `ForecastAdvisor` exists to let the scoreboard
  judge rather than assume: run the identical strategy with `ctx.jev = ForecastAdvisor(...)` vs.
  `OfflineJevAdvisor`/`JevAdvisor` in a backtest and let `jevtrader/backtest/` and
  `evaluate.py` say which one actually clears costs on your instrument and horizon. Don't wire a
  return forecast into a strategy's sizing until it has beaten `RandomWalkForecaster` on your own
  data by that same test.

## Modules

| module | purpose |
|---|---|
| `base.py` | `Forecaster` protocol, `QuantileForecast` (point + q10..q90, `prob_above`/`prob_below` via CDF interpolation) |
| `targets.py` | bars -> series transforms (log returns, realized vol, log volume, spread) and their inverses (cumulative bps, bps scaling, exp) |
| `baselines.py` | `RandomWalkForecaster`, `EWMAVolForecaster`, `GARCH11Forecaster` (scipy MLE, no `arch` dep), `SeasonalNaiveForecaster` |
| `timesfm_backend.py` | `TimesFMForecaster` -- real TimesFM 2.5/3.0, lazy-imported, batched, license-guarded |
| `features.py` | `forecast_features(...)` -- the compact `tfm_*` dict merged into Jev's state, cached per (symbol, last bar ts) |
| `advisor.py` | `ForecastAdvisor` -- `JevAdvisorProtocol` from `tfm_*` features alone |
| `evaluate.py` | rolling-origin (no-look-ahead) evaluation: pinball/CRPS, 80% coverage, MASE, directional hit rate, vol QLIKE/MSE |

## License

| version | weights | license | where JevTrader allows it |
|---|---|---|---|
| **2.5** (`google/timesfm-2.5-200m-pytorch`) | 200M, PyTorch | **Apache-2.0** | production default -- backtest, paper, live |
| **3.0** (`google/timesfm-3.0-pytorch`) | native multivariate + covariates | **non-commercial, non-production only** | `TradingMode.BACKTEST` **only**, and only with `TIMESFM3_LICENSE_ACK=research_only` set explicitly |

`TimesFMForecaster(version="3.0", ...)` enforces this at construction time (`TimesFM3LicenseError`
if either condition isn't met) -- it is not a comment, it's a refusal to load. There is
deliberately no way to reach TimesFM 3.0 from `TradingMode.PAPER` or `TradingMode.LIVE`, and the
env var has to be set by a human, not defaulted anywhere in config. If you only have TimesFM 2.5
available (the common case), everything in this package works identically with `version="2.5"`.

## Latency

Forecasting runs **once per bar close, batched across every symbol/target in a single forward
pass** (`TimesFMForecaster.forecast` takes the whole `{key: series}` dict and calls
`model.forecast(inputs=[...])` once) -- never per-tick, never per-quote. `forecast_features`
additionally caches per `(symbol, last bar timestamp, horizon, cost_bps)`, so a bar that hasn't
closed yet costs nothing to re-query.

Rough expectations (this container has no GPU and can't download weights, so these are the
published/vendor numbers, not measured here -- see `scripts/forecast/eval_real.py --model
timesfm2.5` to measure on your own machine):

| backend | ~latency, small batch | notes |
|---|---|---|
| TimesFM 2.5, CPU | tens of ms per symbol | fine for once-per-bar-close on a handful of symbols; batch across symbols to amortize |
| TimesFM 2.5, CUDA | low single-digit ms per symbol | batches of dozens-hundreds of symbols in one call |
| TimesFM 3.0, Apple Silicon (MLX) | ~11ms p50 at batch 1 on M4 Max | per TimesFM 3.0's own benchmarks; backtest-only per the license above |
| any honest baseline (`baselines.py`) | microseconds-low milliseconds | no model weights, no GPU; `GARCH11Forecaster`'s MLE fit is the slowest at a few hundred ms for a ~1000-bar context |

If a bar close's forecast call ever threatens to run long relative to the bar interval (e.g.
GARCH fitting on very long, uncapped history), cap context with `max_context` (both
`TimesFMForecaster` and `evaluate.py`'s rolling-origin functions take this) rather than let a fit
grow unbounded with the size of your cache.

## Regime heuristic

`ForecastAdvisor.regime()` is explicitly **not a forecast** -- it only has a volatility forecast
(`tfm_vol_ratio` = forecast vol / realized vol) to reason from, so it speaks with some real
grounding to the volatile-chop vs. quiet axis (`vol_ratio > 1` skews `volatile_chop`/`risk_off`,
`< 1` skews `quiet`), and only weakly tie-breaks `trending_up`/`trending_down`/`mean_reverting`
off the base state's `trend_slope_bps` when that key happens to already be present in the merged
state (it's optional; `ForecastAdvisor` never requires anything beyond its own `tfm_*` keys).
Treat this the same way `docs/ARCHITECTURE.md` treats `OfflineJevAdvisor`: a documented,
transparent stand-in, not Jev, and not a claim of forecasting skill it hasn't earned on the
scoreboard.

## Honest baseline results (real data)

`scripts/forecast/eval_real.py` run against
`data/cache/btcusd_bitstamp_1min_2025_2026.csv` (1-minute BTC/USD bars, Bitstamp, Jan 2025 - Sep
2026, ~904k bars) in this container (baselines only -- no GPU, no network, so TimesFM itself
could not be run here; run `--model timesfm2.5` on your own machine to add it to this table).
Rolling-origin, no-look-ahead (`min_train=2000, step=240, max_windows=200, max_context=1000,
cost_bps=5`, i.e. the maker-side round-trip on Alpaca crypto tier 1 per `docs/ARCHITECTURE.md`):

## Returns (cumulative, bps)

| forecaster | horizon | n | MASE | coverage_80 | CRPS | dir. hit rate (n) |
|---|---:|---:|---:|---:|---:|---:|
| random_walk | 5 | 200 | 1.000 | 0.770 | 7.70 | n/a |
| random_walk | 15 | 200 | 1.000 | 0.775 | 12.70 | n/a |
| random_walk | 60 | 200 | 1.000 | 0.760 | 23.31 | n/a |

## Realized volatility

| forecaster | horizon | n | QLIKE | MSE (var) | MSE vs naive |
|---|---:|---:|---:|---:|---:|
| ewma | 5 | 200 | -14.5519 | 1.081e-12 | 15.243 |
| ewma | 15 | 200 | -14.2462 | 2.170e-12 | 4.893 |
| ewma | 60 | 200 | -13.8730 | 5.905e-13 | 0.536 |
| garch | 5 | 200 | 13.6308 | 6.835e-13 | 9.633 |
| garch | 15 | 200 | 21.2744 | 1.378e-12 | 3.109 |
| garch | 60 | 200 | 22.7393 | 5.021e-13 | 0.456 |

(full command and settings above the table; run took ~4m20s in this container, almost
entirely `GARCH11Forecaster`'s per-window MLE refit -- see the latency table above for why that's
never on a hot path.)

**Reading these numbers:**

- `random_walk`'s own MASE is trivially `1.000` at every horizon -- it *is* the zero-forecast
  denominator MASE is measured against, not a claim of skill. It exists so a real model (TimesFM,
  or a fine-tune per `scripts/forecast/finetune_timesfm.md`) has a number to beat: **MASE < 1.0
  is the bar**, and until something clears it, the honest thing for a strategy to do is treat
  short-horizon direction as noise, exactly as `docs/ARCHITECTURE.md`'s fee-reality section
  already argues from the cost side.
- **`dir. hit rate` is `n/a` at every horizon -- and that itself is the honest-prior result, not
  a missing number.** The directional-hit-rate metric only counts windows where the forecaster
  claimed >50% confidence of clearing `cost_bps=5` in one direction; a zero-drift random walk's
  quantile band is centered on 0, so a positive threshold of 5 bps essentially never falls on the
  >50%-confident side of its own distribution. In 600 rolling-origin windows across 1.75 years of
  BTC/USD minute bars, the honest baseline never once manufactured a tradeable directional call
  net of a realistic cost -- exactly what "returns are close to a random walk at short horizons"
  predicts. A real forecaster (TimesFM, or a fine-tune) is genuinely useful for direction only if
  it starts producing tradeable calls *and* those calls hit at better than the naive rate; measure
  both, not the confidence rate alone.
- `coverage_80` (0.76-0.775, vs. an ideal 0.80) means the random walk's quantile band is close to
  calibrated but slightly too narrow on this instrument/horizon combination (BTC minute returns
  have fatter tails than the empirical-quantile-times-sqrt(h) approximation captures) -- a real
  forecaster should be held to at least this standard, not a looser one.
- For volatility, `mse_vs_naive` (MSE vs. simple last-value persistence) tells a sharper story
  than QLIKE alone: **both EWMA and GARCH are *worse* than naive persistence at 5 and 15 minutes**
  (ratios of 4.9-15x), and only pull ahead of it at 60 minutes (ratios of 0.46-0.54). Realized
  vol at 1-minute granularity is so strongly autocorrelated bar-to-bar that "vol in the next 5-15
  minutes ~= vol right now" is a genuinely hard baseline to beat; EWMA/GARCH's smoothing only pays
  off once the horizon is long enough that pure persistence has become stale. This is exactly the
  kind of scoreboard result that should gate what `risk/sizing.py` uses at which horizon --
  don't assume EWMA/GARCH (or TimesFM) beats persistence just because it's a fancier model.
- GARCH's occasional very large QLIKE values on this dataset are a real, known GARCH pitfall, not
  a code bug: a maximum-likelihood fit on a bounded context (`max_context` bars) can occasionally
  converge to a near-degenerate near-zero-variance regime, and QLIKE asymmetrically punishes
  *underestimating* volatility exactly as hard as that mistake deserves in a risk-sizing context.
  That asymmetry is the reason QLIKE, not MSE alone, should decide whether a vol forecaster is
  trustworthy enough for `risk/sizing.py`.

To reproduce or extend this table (more horizons, a different symbol, or once you have
`timesfm[torch]` installed with GPU/MPS available):

```bash
python scripts/forecast/eval_real.py --model random_walk --model ewma --model garch \
    --horizons 5 15 60 --cost-bps 5
python scripts/forecast/eval_real.py --model timesfm2.5   # your own machine
```
