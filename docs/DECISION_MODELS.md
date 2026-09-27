# Decision models: Jev vs. Kev vs. Laya

JevTrader's decision layer ("Jev judges, code executes") doesn't have to be hosted Jev. Kev
(`github.com/jaredpalmer/kev`, Apache-2.0) and Laya (`github.com/NandhaKishorM/laya`,
Apache-2.0) serve the exact same `/v1/systemone` wire protocol, so `jevtrader/jev/backends.py`
can point the same `JevAdvisor` at any of them with no code change -- only `DECISION_BACKEND`
and a base URL differ. This doc covers what each one is, when to run which, the workflow for
fine-tuning a local model to actually beat hosted Jev on JevTrader's own questions, and the
honest empirical results from running that workflow once, here, on real BTC/USD data.

**Do not confuse this with the unrelated PyPI package `kev`** (a document/key-value store
library) -- the decision model is `jaredpalmer/kev` on GitHub specifically.

## Comparison

| | **Jev** (hosted) | **Kev** | **Laya** |
|---|---|---|---|
| Where it runs | TypeSafe's cloud | your machine (CUDA, ROCm, or MLX on Apple Silicon, auto-detected) | your machine (CUDA, ROCm, or Apple GPU) |
| License / cost | proprietary API, per-call pricing | Apache-2.0, free to self-host; ~$1/fine-tune run on a rented H100 (Modal) | Apache-2.0, free to self-host; fine-tune on free-tier Kaggle 2x T4 or your own GPU |
| Base checkpoints | undisclosed | Qwen3.5 0.8B / 4B / 9B + Qwen3.8-27B, LoRA r16 + a pointer head | `convaiinnovations/laya` (421M), `laya-multilingual` (322M) |
| Needs fine-tuning to be useful for JevTrader's questions? | No -- general-purpose, already trained | Recommended (always with `--init_from` an existing Kev checkpoint, never from the bare base model, to keep its general System One skill while specializing) | **Yes** -- near-chance zero-shot on domain-specific typed questions; must be fine-tuned (RLCD) first |
| Measured latency | ~236-276ms + network round trip | Apple Silicon: 0.8B ~149ms (new text) / ~28ms (cached prefix); 4B ~721ms. NVIDIA: 4B on an L40S ~42ms; 0.8B on an L4 ~23ms | NVIDIA T4: ~33-40ms/question. Apple GPU: ~193-464ms |
| Calibration | TypeSafe's own claim; not independently re-verified here | fine-tune-dependent -- must be established per checkpoint via `scoreboard.py`, not assumed | fine-tune-dependent (RLCD trains directly against strictly proper scoring rules, which is a point in its favor, but still must be verified per checkpoint) |
| API surface | reference implementation of `/v1/systemone` | serves the identical `/v1/systemone` contract (`kev.serve`), confirmed by reading its source in this environment | serves the identical `/v1/systemone` contract (`laya-serve`'s own module docstring says so explicitly), confirmed by reading its source in this environment |

None of the latency or accuracy numbers above are interchangeable across hardware or task —
they're the vendors' own reported figures for their own benchmarks, not JevTrader's. The only
numbers below that are ours are in [Empirical results](#empirical-results-btcusd-1-minute-bars),
from running this repo's own dataset builder + scoreboard against real BTC/USD bars.

## Recommendation by hardware

- **Apple Silicon Mac, no eGPU** (the common local dev setup): Kev's 0.8B checkpoint, served via
  MLX (automatic on Apple Silicon) — ~149ms cold, ~28ms with a cached state prefix, which is
  comfortably inside a `JEV_LATENCY_BUDGET_MS` of a few hundred ms. Laya is viable too but
  measured slower on an Apple GPU (~193-464ms); prefer it only if its fine-tuned accuracy
  clearly wins on `scoreboard.py`.
- **NVIDIA GPU** (a workstation card, or a cloud instance you already have running for other
  reasons — L4, L40S, etc.): Kev's 4B checkpoint on an L40S (~42ms) or 0.8B on an L4 (~23ms)
  give the best measured latency/quality tradeoff of the three; Laya's ~33-40ms/question on a
  T4 is competitive if its fine-tune wins on calibration and edge.
- **No GPU at all**: stay on hosted Jev for the live decision layer (paper/live). Kev/Laya only
  hit useful latency on a GPU or Apple Silicon; running either on CPU is not validated anywhere
  in this doc and is very unlikely to compete with `~250ms` hosted Jev, let alone a fast local
  one. You can still build the fine-tune dataset and fit the free, CPU-only logistic baseline
  (`finetune/baseline.py`) locally, and rent a GPU only for the occasional fine-tune run
  (`scripts/finetune/modal_kev.md`) — training is a batch job; serving is not, and that's the
  part that actually needs the GPU to be local (or at least low-latency-reachable) at trade time.

## Workflow: fine-tune -> scoreboard -> shadow-in-paper -> promote

1. **Build a dataset** from historical bars with `scripts/finetune/build_dataset.py`. Ground
   truth comes from REALIZED forward outcomes (never from a model's opinion): `direction`
   up/down/flat net of a stated round-trip cost, `regime`/`risk_off` from forward
   volatility/drawdown (a documented heuristic, not ground truth), `entry_quality` from the
   forward MFE/MAE ratio. States and question wording come from `jevtrader.jev.state.build_state`
   and `jevtrader.jev.questions` directly -- the exact functions `JevAdvisor` calls at serve
   time, so there is zero train/serve skew. Splits are chronological with an embargo (at least
   one horizon's worth of bars) between train/calibration/dev/test so no split's label depends
   on bars inside the next split's date range.
2. **Fit the honest baseline** (`jevtrader/jev/finetune/baseline.py`): a multinomial logistic
   regression on `build_state`'s numeric features (no sklearn — plain numpy + scipy), trained on
   `train`, temperature-calibrated on `calibration`. This is the bar a fine-tuned LLM has to
   clear; it answers in microseconds and has no learned understanding of anything. If Kev/Laya
   can't beat it, the extra latency and complexity of an LLM isn't earning its keep.
3. **Fine-tune Kev** (`scripts/finetune/kev_finetune.sh`, or `scripts/finetune/modal_kev.md` for
   a rented GPU) **or Laya** (`scripts/finetune/laya_finetune.md`, RLCD on 2x T4 via Kaggle) on
   `train.jsonl`. Kev fine-tunes always warm-start from an existing Kev checkpoint
   (`--init_from`) to keep its general System One skill rather than specializing from scratch.
   Optionally mix in `dataset.from_decision_log` distillation records (soft targets from hosted
   Jev's own logged paper-trading answers) as an auxiliary signal — clearly separate from the
   ground-truth records in any accounting, since they're a teacher's opinion, not what happened.
4. **Run the scoreboard** (`scripts/finetune/scoreboard.py`, or `finetune.scoreboard.run_scoreboard`
   directly) on the untouched `test` split, across every candidate you have running: hosted Jev,
   the fine-tuned Kev/Laya server, `OfflineJevAdvisor`, and the logistic baseline. It reports
   accuracy, Brier score, log loss, expected calibration error, a reliability curve, latency
   p50/p95/p99, coverage at a latency budget, and — the metric that actually matters for
   trading — mean forward return net of cost among the examples the model would have acted on
   ("edge after costs"), plus an illustrative cumulative PnL curve.
5. **Apply `beats(candidate, incumbent)`**: a candidate replaces an incumbent ONLY if it wins on
   ALL THREE, on the SAME untouched test split: lower Brier, higher after-cost edge, and lower
   p95 latency. It also reports bootstrap 95% confidence intervals on both sides' Brier and
   per-trade edge, so a "win" that's really noise from a small test set is visible as such
   rather than quietly accepted.
6. **Shadow it in paper trading before promoting**: wrap the current (incumbent) advisor and the
   candidate in `jevtrader.jev.advisor.ShadowAdvisor`. Trading decisions keep coming from the
   incumbent (`ShadowAdvisor.direction`/`.regime`/`.ask` always return the primary's answer,
   immediately, never blocked on a shadow); the candidate is asked the identical question on a
   background thread and its answer is logged (`DecisionLog`, tagged
   `extra={"backend": "kev", "shadow": true}`) purely for comparison. This collects a live,
   walk-forward, no-lookahead Jev-vs-Kev-vs-Laya comparison on identical states for free during
   normal paper trading, without ever risking a live decision on an unproven model.
7. **Promote**: once the shadow log confirms step 5's offline verdict over a real stretch of
   paper trading, set `DECISION_BACKEND=kev` (or `laya`) and the promotion is just an env
   variable — `make_advisor` builds the same `JevAdvisor` class against the new backend's
   `TypeSafeClient`, and every strategy that calls `ctx.jev.direction(...)` is unaffected.
8. **Repeat**: keep logging every decision (`DecisionLog`) in paper and live; periodically
   re-run steps 1-5 as more real data accrues, and re-fine-tune to correct for drift. Nothing
   here is a one-time gate — `beats()` is meant to be re-applied whenever there's a new
   candidate or enough new data to matter.

### Practical wiring

```bash
# .env (or shell) -- picks the backend make_advisor() builds in paper/live mode
DECISION_BACKEND=kev            # jev | kev | laya | offline (default: jev, i.e. unchanged
                                 # pre-existing behavior, when unset)
KEV_BASE_URL=http://127.0.0.1:8009    # default if unset
LAYA_BASE_URL=http://127.0.0.1:8000   # default if unset
LOCAL_MODEL_API_KEY=                  # usually unset -- a local server with no auth configured
                                       # still needs SOME non-empty key sent; jevtrader supplies
                                       # a harmless placeholder automatically when this is blank
```

```python
from jevtrader.jev.advisor import ShadowAdvisor, make_advisor
from jevtrader.jev.backends import DecisionBackend, make_client
from jevtrader.jev.log import DecisionLog

decision_log = DecisionLog(settings.runs_dir / "decisions.jsonl")
primary = make_advisor(settings, mode)  # hosted Jev, per $DECISION_BACKEND / has_jev
shadow_kev = JevAdvisor(make_client(DecisionBackend.KEV), latency_budget_ms=400)
advisor = ShadowAdvisor(primary, {"kev": shadow_kev}, decision_log=decision_log)
# strategies see `advisor` as ctx.jev -- same JevAdvisorProtocol, unaware it's shadowing anything
```

## Empirical results: BTC/USD, 1-minute bars

Run against `data/cache/btcusd_bitstamp_1min_2025_2026.csv` (Bitstamp BTC/USD, 1-minute bars,
2025-01-07 through 2026-09-27, ~904k bars). **No GPU or Hugging Face access in this environment,
so Kev/Laya themselves could not be fine-tuned or scored here** — the numbers below are
`OfflineJevAdvisor` (the NOT-Jev heuristic) vs. the logistic baseline, both scored by
`finetune/scoreboard.py` on an untouched, chronologically-embargoed test split, at every
combination of horizon (5min, 15min) and round-trip cost (0 bps "no cost", 30 bps "maker",
50 bps "taker" — from `docs/ARCHITECTURE.md`'s Alpaca crypto fee reality check). Dataset
parameters: `lookback_bars=120`, `stride=30` (subsampled for tractable build time in this
sandbox; still tens of thousands of examples per split), chronological 70/10/10/10 train/
calibration/dev/test split with a horizon-sized embargo between splits. Full JSONL + scoreboard
JSON/markdown for every combination are in `data/cache/finetune/` (gitignored, not committed;
regenerate with the command below).

**Base rates** (fraction of examples labelled up/down/flat, train vs. test -- direction, i.e.
"does the 5-minute forward move clear this cost hurdle"):

| horizon / cost | split | up | down | flat |
|---|---|---|---|---|
| 5min, no cost (0 bps) | train (n=21,100) | 49.1% | 50.1% | 0.7% |
| 5min, no cost (0 bps) | test (n=3,015) | 50.2% | 48.7% | 1.1% |
| 5min, maker (30 bps) | train (n=21,100) | 2.5% | 3.0% | 94.5% |
| 5min, maker (30 bps) | test (n=3,015) | 1.4% | 1.0% | 97.6% |
| 5min, taker (50 bps) | train (n=21,100) | 0.65% | 0.75% | 98.6% |
| 5min, taker (50 bps) | test (n=3,015) | 0.4% | 0.3% | 99.3% |

**Scoreboard** (`OfflineJevAdvisor` vs. the logistic baseline, on the untouched test split of
each; `edge_after_costs_bps` and `beats()` use the default 0.5 probability threshold):

| horizon / cost | backend | n | accuracy | Brier(up) | log loss(up) | ECE(up) | p50/p95/p99 latency (ms) | edge after costs (bps) | `beats(baseline, offline)` |
|---|---|---|---|---|---|---|---|---|---|
| 5min, 0 bps | offline | 3,015 | 24.4% | 0.300 | 0.811 | 0.194 | 0.008 / 0.015 / 0.027 | +0.67 (n=297) | -- |
| 5min, 0 bps | logit_baseline | 3,015 | 50.5% | 0.251 | 0.696 | 0.015 | 0.033 / 0.066 / 0.118 | +0.55 (n=1,014) | **False** (loses on edge and p95 latency) |
| 5min, 30 bps | offline | 3,015 | 97.5% | 0.016 | 0.093 | 0.041 | 0.007 / 0.012 / 0.037 | -33.0 (n≈70) | -- |
| 5min, 30 bps | logit_baseline | 3,015 | 97.4% | 0.013 | 0.060 | 0.002 | 0.032 / 0.058 / 0.084 | -154.2 (n≈41) | **False** (loses on edge and p95 latency) |
| 5min, 50 bps | offline | 3,015 | 99.3% | 0.0066 | 0.065 | 0.049 | 0.010 / 0.017 / 0.038 | -61.8 (n≈12) | -- |
| 5min, 50 bps | logit_baseline | 3,015 | 99.3% | 0.0037 | 0.020 | 0.0009 | 0.048 / 0.089 / 0.120 | none selected (never crosses p≥0.5) | **False** (no edge to compare; wins Brier/ECE, loses p95 latency) |

Full scoreboard JSON/markdown (including the reliability curve and PnL curve) for each of these
three is in `data/cache/finetune/btcusd_5min_{no_cost,maker30bps,taker50bps}/scoreboard.{json,md}`.
**The 15-minute horizon variants use the identical, already-built and tested pipeline
(`build_dataset.py`/`scoreboard.py`) but were not run here** — a single dataset build over the
full ~904k-bar series took ~135 seconds even at `--stride 30` (subsampling to keep the six
planned combinations tractable in this sandbox's time budget), and this environment's session
ended before the three 15-minute combinations finished; "Reproduce this" below runs any of them
in a few minutes on a normal machine.

### Reading these numbers honestly

**The headline finding is exactly what `docs/ARCHITECTURE.md`'s fee reality check predicts:
at a realistic cost hurdle, there is next to nothing to predict.** At Alpaca's crypto taker
round-trip cost (50 bps), 99.3% of all 5-minute windows in 21 months of BTC/USD 1-minute bars
don't move far enough to matter -- the label is "flat" almost every single time, for both model
and baseline alike, and accuracy numbers in the high 90s are a base-rate artifact, not skill.
The place real signal shows up is calibration, not classification:

- **The logistic baseline is consistently better calibrated than the offline heuristic** across
  every cost level (lower Brier, lower log loss, and a dramatically lower ECE — e.g. 0.0009 vs.
  0.049 at 50 bps) — unsurprising, since it's fit directly on this data's base rates and
  temperature-scaled on a held-out split, while the offline heuristic's weights are fixed,
  hand-picked defaults never tuned to BTC/USD specifically. This is the expected outcome, not
  evidence either one has real skill: a model that has learned "say flat" almost always will
  look "calibrated" once the base rate is that lopsided.
- **`beats()` says the baseline does not replace the offline heuristic at any cost level** —
  it wins on Brier every time, but loses on p95 latency every time (a Python-level logistic
  regression call is still ~2-6x slower than the hand-written heuristic's arithmetic in this
  unoptimized, one-call-at-a-time harness) and never wins on after-cost edge: at 0 bps its edge
  is a wafer-thin +0.55 bps on 1,014 trades (indistinguishable from noise before even counting
  the bootstrap CI in the saved `scoreboard.json`), and at 30/50 bps neither predictor is
  net-positive after costs, with the baseline notably WORSE than the heuristic at 30 bps
  (-154 bps vs. -33 bps edge) despite its better calibration — being well-calibrated about a
  near-impossible target does not make trading on it profitable.
- **Realistic costs erase any semblance of edge.** At 0 bps, both predictors' PnL curves and
  edge are near flat-to-random; at 30 and 50 bps, every combination that trades at all loses
  money after cost, on both offline and baseline. Nothing here supports trading raw 5-minute
  BTC/USD directional signals from `build_state`'s technical features at Alpaca's crypto fee
  schedule — full stop, independent of which decision model answers the question.
- **This is precisely the scoreboard's job, and precisely why `beats()` gates on after-cost
  edge, not just accuracy or calibration.** A fine-tuned Kev or Laya checkpoint faces the same
  bar: if it cannot show a positive, statistically meaningful (check the bootstrap CI) edge
  after costs on held-out data, it should not be promoted, no matter how good its accuracy or
  calibration numbers look in isolation. Longer horizons (15min+), coarser-cost variants
  (maker rebates, larger expected moves, or a different asset with more genuine trend), or
  richer features (e.g. a `jevtrader.forecast` quantile model's output merged in via
  `build_state`'s `extra_features`) are the more promising places to look for a hurdle a model
  can actually clear -- not squeezing more accuracy out of the same features at a horizon this
  short.

### Reproduce this

```bash
. .venv/bin/activate
python scripts/finetune/build_dataset.py \
    --bars data/cache/btcusd_bitstamp_1min_2025_2026.csv --symbol BTC/USD \
    --horizon 5min --cost-bps 50 --lookback-bars 120 --stride 30 \
    --out data/cache/finetune/btcusd_5min_taker50bps

python -c "
from jevtrader.jev.finetune.dataset import load_examples_jsonl
from jevtrader.jev.finetune.baseline import fit_logistic_baseline, calibrate_temperature, save_baseline
train = load_examples_jsonl('data/cache/finetune/btcusd_5min_taker50bps/train.jsonl')
calib = load_examples_jsonl('data/cache/finetune/btcusd_5min_taker50bps/calibration.jsonl')
b = calibrate_temperature(fit_logistic_baseline(train), calib)
save_baseline(b, 'runs/btcusd_5min_taker50bps_baseline.json')
"

python scripts/finetune/scoreboard.py \
    --test data/cache/finetune/btcusd_5min_taker50bps/test.jsonl \
    --offline --baseline runs/btcusd_5min_taker50bps_baseline.json \
    --out runs/btcusd_5min_taker50bps_scoreboard
```

Once you have a GPU (or a rented one — `scripts/finetune/modal_kev.md`), fine-tune Kev
(`scripts/finetune/kev_finetune.sh`) or Laya (`scripts/finetune/laya_finetune.md`) on the SAME
`train.jsonl`/`calibration.jsonl` and add `--kev http://127.0.0.1:8009` / `--laya
http://127.0.0.1:8000` (and `--jev`, if you want hosted Jev in the comparison too) to the
`scoreboard.py` call above — everything else in this doc's workflow applies unchanged.
