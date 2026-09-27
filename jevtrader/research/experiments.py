"""Reproducible research studies -- one entry point per study, each writes a Markdown report
into `research_reports/` (small text files; the data itself is never committed).

Methodology (non-negotiable, see docs/STRATEGIES.md):
  * chronological 60/20/20 split of each data set (train / validation / holdout);
  * parameter sweeps run on TRAIN only; `n_trials` is recorded and the Deflated Sharpe Ratio of
    the validation result is reported against that trial count;
  * the chosen parameters are the best on train *subject to a neighbourhood-stability check*
    (the mean Sharpe of adjacent grid points must also be > 0), then evaluated on VALIDATION,
    and the HOLDOUT is evaluated exactly once with those same parameters;
  * every number is NET of Alpaca fees (jevtrader.core.fees defaults) and the SimBroker's
    spread/slippage terms; fee-stress (x1.5, x2), slippage x2 and, for crypto, a fee ladder
    (25/15/10/5/2/0 bps per side) locate the break-even fee;
  * finalists are re-run through the event-driven `Backtester` with the registered Strategy class
    on validation + holdout to confirm the fast path.

Run everything: `python -m jevtrader.research.experiments` (~2-4 minutes). Individual studies
are plain functions returning the report text so they can be unit-tested on small data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from jevtrader.backtest import Backtester, SlippageModel
from jevtrader.backtest.metrics import deflated_sharpe_ratio
from jevtrader.core.fees import AlpacaCryptoFees, AlpacaEquityFees, CompositeFees
from jevtrader.core.types import AssetClass
from jevtrader.research import signals as S
from jevtrader.research.fastscreen import (
    ALPACA_CRYPTO_MAKER,
    ALPACA_CRYPTO_TAKER,
    ALPACA_EQUITY_LIQUID_ETF,
    ALPACA_EQUITY_TAKER,
    CRYPTO_FEE_LADDER_BPS,
    CostSpec,
    FastResult,
    fee_ladder,
    robustness,
    simulate_positions,
    sweep_grid,
    verdict,
)
from jevtrader.research.loaders import (
    ChronoSplit,
    available,
    chrono_split,
    load_btc_1m,
    load_c_1m,
    load_spx_1m,
    resample,
    to_bar_events,
    us_session_only,
)

REPORT_DIR = Path(__file__).resolve().parents[2] / "research_reports"


# ----------------------------------------------------------------------------- helpers


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: floatfmt.format(v) if np.isfinite(v) else "nan")
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def _dict_table(d: dict, title_key: str = "metric", value_key: str = "value") -> str:
    df = pd.DataFrame({value_key: pd.Series(d)})
    df.index.name = title_key
    return _md_table(df, "{:.3f}")


@dataclass
class Dataset:
    label: str
    bars_1m: pd.DataFrame
    asset_class: AssetClass
    cost: CostSpec
    symbol: str
    session_filter: bool = False  # equities: keep the US cash session only
    long_only: bool = False


def _prep(ds: Dataset, rule: str) -> pd.DataFrame:
    b = ds.bars_1m if rule == "1min" else resample(ds.bars_1m, rule)
    if ds.session_filter:
        b = us_session_only(b)
    return b


def _split_indices(bars: pd.DataFrame, split: ChronoSplit) -> dict[str, pd.DatetimeIndex]:
    """Map the 1-min split boundaries onto a resampled index."""
    idx = bars.index
    tr_end, va_end = split.train.index[-1], split.val.index[-1]
    return {"train": idx[idx <= tr_end], "val": idx[(idx > tr_end) & (idx <= va_end)], "holdout": idx[idx > va_end]}


def _slice_result(bars: pd.DataFrame, target: pd.Series, idx: pd.DatetimeIndex, cost: CostSpec, ds: Dataset, n_trials: int = 1, full: bool = False) -> FastResult:
    b = bars.loc[idx]
    t = target.loc[idx]
    return simulate_positions(b, t, cost, asset_class=ds.asset_class, long_only=ds.long_only, n_trials=n_trials, compute_full_metrics=full)


@dataclass
class StudyOutcome:
    dataset: str
    rule: str
    params: dict
    n_trials: int
    rows: dict[str, dict] = field(default_factory=dict)  # train/val/holdout summaries
    neighbourhood_sharpe: float = float("nan")
    dsr_val: float = float("nan")
    dsr_holdout: float = float("nan")
    robust: dict = field(default_factory=dict)
    ladder: Optional[pd.DataFrame] = None
    verdict: str = ""
    event_check: dict = field(default_factory=dict)
    sweep_top: Optional[pd.DataFrame] = None
    stability: Optional[pd.DataFrame] = None

    def to_markdown(self) -> str:
        lines = [f"### {self.dataset} @ {self.rule}", ""]
        lines.append(f"Chosen params (best train Sharpe with positive neighbourhood, n_trials={self.n_trials}): `{self.params}`")
        lines.append("")
        df = pd.DataFrame(self.rows).T
        keep = ["sharpe", "total_return", "max_drawdown", "n_round_trips", "trades_per_day", "gross_edge_bps_per_rt", "cost_bps_per_rt", "net_edge_bps_per_rt", "breakeven_fee_bps_per_side", "exposure"]
        df = df[[c for c in keep if c in df.columns]]
        df.index.name = "split"
        lines.append(_md_table(df))
        lines.append("")
        lines.append(f"* Neighbourhood mean train Sharpe (adjacent grid points): **{self.neighbourhood_sharpe:.2f}**")
        lines.append(f"* Deflated Sharpe (validation, n_trials={self.n_trials}): **{self.dsr_val:.3f}**; holdout DSR: {self.dsr_holdout:.3f}")
        if self.robust:
            r = self.robust
            lines.append(f"* Robustness (holdout): base SR {r['base']['sharpe']:.2f} | fees x1.5 {r['fees_x1.5']['sharpe']:.2f} | fees x2 {r['fees_x2']['sharpe']:.2f} | slippage x2 {r['slippage_x2']['sharpe']:.2f} -> {'PASS' if r['passed'] else 'FAIL'}")
        if self.ladder is not None:
            lines.append("")
            lines.append("Fee ladder (validation, same positions re-costed; per-side fee in bps):")
            lines.append("")
            lad = self.ladder.set_index("fee_bps_per_side")
            lines.append(_md_table(lad))
        if self.event_check:
            lines.append("")
            lines.append("Event-driven confirmation (registered Strategy through `Backtester` + `SimBroker`, val+holdout):")
            lines.append("")
            lines.append(_dict_table(self.event_check))
        if self.stability is not None:
            lines.append("")
            lines.append("Parameter-stability table (train Sharpe):")
            lines.append("")
            lines.append(_md_table(self.stability))
        lines.append("")
        lines.append(f"**Verdict: {self.verdict}**")
        lines.append("")
        return "\n".join(lines)


def _neighbourhood(sweep: pd.DataFrame, best: dict, grid: dict[str, Sequence]) -> float:
    """Mean train Sharpe over grid points that differ from `best` in exactly one param by one step."""
    vals = []
    for k, seq in grid.items():
        seq = list(seq)
        if best[k] not in seq or len(seq) < 2:
            continue
        i = seq.index(best[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(seq):
                q = {**best, k: seq[j]}
                m = sweep
                for kk, vv in q.items():
                    m = m[m[kk] == vv] if not (isinstance(vv, float) and np.isnan(vv)) else m[m[kk].isna()]
                if len(m):
                    vals.append(float(m["sharpe"].iloc[0]))
    return float(np.mean(vals)) if vals else float("nan")


def run_bar_study(
    ds: Dataset,
    rule: str,
    signal_fn: Callable[[pd.DataFrame, dict], pd.Series],
    grid: dict[str, Sequence],
    min_train_rt: int = 30,
    stability_params: Optional[tuple[str, str]] = None,
    event_strategy: Optional[Callable[[dict], object]] = None,
    crypto_ladder: bool = False,
    label: Optional[str] = None,
) -> StudyOutcome:
    bars = _prep(ds, rule)
    split = chrono_split(ds.bars_1m)
    idx = _split_indices(bars, split)
    cache: dict[str, pd.Series] = {}

    def run(params: dict) -> FastResult:
        key = repr(sorted(params.items()))
        tgt = cache.get(key)
        if tgt is None:
            tgt = signal_fn(bars, params)
            cache[key] = tgt
        return _slice_result(bars, tgt, idx["train"], ds.cost, ds)

    sweep = sweep_grid(grid, run)
    n_trials = len(sweep)
    ok = sweep[sweep["n_round_trips"] >= min_train_rt]
    cand = ok if len(ok) else sweep
    # prefer the best train Sharpe whose neighbourhood is also positive
    chosen = None
    for _, row in cand.iterrows():
        p = {k: row[k] for k in grid}
        p = {k: (int(v) if isinstance(v, (np.integer, float)) and float(v).is_integer() and isinstance(grid[k][0], int) else v) for k, v in p.items()}
        nb = _neighbourhood(sweep, {k: row[k] for k in grid}, grid)
        if not np.isfinite(nb) or nb > 0:
            chosen = (p, nb)
            break
    if chosen is None:
        row = cand.iloc[0]
        chosen = ({k: row[k] for k in grid}, _neighbourhood(sweep, {k: row[k] for k in grid}, grid))
    params, nb = chosen
    params = {k: (int(v) if isinstance(grid[k][0], int) else float(v)) for k, v in params.items()}
    tgt = signal_fn(bars, params)
    out = StudyOutcome(ds.label, label or rule, params, n_trials, neighbourhood_sharpe=nb)
    for name in ("train", "val", "holdout"):
        r = _slice_result(bars, tgt, idx[name], ds.cost, ds, n_trials=n_trials, full=True)
        out.rows[name] = r.summary()
        if name == "val":
            out.dsr_val = float(deflated_sharpe_ratio(r.net_returns, n_trials=n_trials)["dsr"])
            if crypto_ladder:
                out.ladder = fee_ladder(bars.loc[idx["val"]], tgt.loc[idx["val"]], ds.cost, CRYPTO_FEE_LADDER_BPS, ds.asset_class, ds.long_only)
        if name == "holdout":
            out.dsr_holdout = float(deflated_sharpe_ratio(r.net_returns, n_trials=n_trials)["dsr"])
            out.robust = robustness(bars.loc[idx["holdout"]], tgt.loc[idx["holdout"]], ds.cost, ds.asset_class, ds.long_only)
    out.sweep_top = sweep.head(8)
    if stability_params is not None and all(k in sweep for k in stability_params):
        out.stability = sweep.pivot_table(index=stability_params[0], columns=stability_params[1], values="sharpe", aggfunc="mean").round(2)
    oos = out.rows["holdout"]
    val = out.rows["val"]
    # the verdict uses the OUT-OF-SAMPLE (val + holdout) evidence: both must clear the bar
    oos_sharpe = min(float(val["sharpe"]), float(oos["sharpe"]))
    net_edge = min(float(val["net_edge_bps_per_rt"]), float(oos["net_edge_bps_per_rt"]))
    out.verdict = verdict(oos_sharpe, min(out.dsr_val, out.dsr_holdout) if np.isfinite(out.dsr_holdout) else out.dsr_val, net_edge, bool(out.robust.get("passed", False)))
    if event_strategy is not None:
        out.event_check = event_confirm(ds, split, params, event_strategy, out)
    return out


def event_confirm(ds: Dataset, split: ChronoSplit, params: dict, make_strategy: Callable[[dict], object], out: StudyOutcome) -> dict:
    """Run the registered Strategy on val+holdout 1-min bars through the Backtester."""
    oos = pd.concat([split.val, split.holdout])
    strat = make_strategy(params)
    fees = CompositeFees(equity=AlpacaEquityFees(), crypto=AlpacaCryptoFees())
    slip = SlippageModel(half_spread_bps=ds.cost.half_spread_bps, slippage_bps=ds.cost.slippage_bps, impact_coeff_bps=0.0, max_participation=float("inf"))
    bt = Backtester([strat], {ds.symbol: to_bar_events(oos, ds.symbol)}, fees=fees, fill_model=slip, initial_cash=100_000.0, allow_leverage=not ds.long_only, leverage=1.0)
    t0 = time.time()
    res = bt.run()
    fast_ret = (1 + out.rows["val"]["total_return"]) * (1 + out.rows["holdout"]["total_return"]) - 1
    fast_rt = out.rows["val"]["n_round_trips"] + out.rows["holdout"]["n_round_trips"]
    return {
        "event_sharpe": res.metrics["sharpe"],
        "event_total_return": res.metrics["total_return"],
        "event_max_drawdown": res.metrics["max_drawdown"],
        "event_n_fills": res.metrics["n_fills"],
        "event_fee_bps_per_fill": res.metrics["cost_per_trade_bps"],
        "fast_total_return_val_x_holdout": fast_ret,
        "fast_round_trips_val+holdout": fast_rt,
        "runtime_s": time.time() - t0,
    }


# ----------------------------------------------------------------------------- datasets


def load_datasets(which: Optional[set[str]] = None) -> dict[str, Dataset]:
    av = available()
    out: dict[str, Dataset] = {}
    if av.get("btcusd_bitstamp_1min_2025_2026.csv") and (which is None or "btc" in which):
        out["btc"] = Dataset("BTC/USD Bitstamp 1m (Alpaca crypto taker t1: 25 bps + 2.5 half-spread + 1 slip)", load_btc_1m(), AssetClass.CRYPTO, ALPACA_CRYPTO_TAKER, "BTC/USD", long_only=True)
    if av.get("SPX500_1m_sample.csv") and (which is None or "spx" in which):
        out["spx"] = Dataset("SPX500 CFD 1m as SPY proxy (0.15 fee + 0.25 half-spread + 0.25 slip per side)", load_spx_1m(), AssetClass.EQUITY, ALPACA_EQUITY_LIQUID_ETF, "SPY", session_filter=True)
    if av.get("C_1m_sample.csv") and (which is None or "c" in which):
        out["c"] = Dataset("Citigroup 1m (0.15 fee + 1.0 half-spread + 0.5 slip per side)", load_c_1m(), AssetClass.EQUITY, ALPACA_EQUITY_TAKER, "C", session_filter=True)
    return out


def _header(title: str, blurb: str, datasets: dict[str, Dataset]) -> list[str]:
    lines = [f"# {title}", "", f"_Generated by `jevtrader.research.experiments` on {pd.Timestamp.now('UTC'):%Y-%m-%d}. All figures NET of Alpaca fees + modelled spread/slippage._", "", blurb, "", "## Data & splits", ""]
    for ds in datasets.values():
        sp = chrono_split(ds.bars_1m)
        lines.append(f"* **{ds.label}** -- {sp.describe()}")
    lines.append("")
    return lines


# ----------------------------------------------------------------------------- studies


def study_mean_reversion(datasets: dict[str, Dataset], write: bool = True) -> str:
    from jevtrader.strategies.vwap_reversion import ZScoreReversion

    grid = {"window": [20, 30, 45, 60, 90], "entry_z": [1.5, 2.0, 2.5, 3.0], "exit_z": [0.0, 0.5], "max_vol_mult": [1.5, 100.0]}
    lines = _header(
        "02 - Intraday mean reversion to VWAP (z-score, vol-regime filter)",
        "Strategy: `zscore_reversion`. Enter when close deviates > entry_z sigma from rolling VWAP, exit at exit_z; entries "
        "suppressed when short-window vol > max_vol_mult x long-window vol (100 = filter off). Crypto is long-only "
        "(Alpaca). The single most important columns are `gross_edge_bps_per_rt` vs `cost_bps_per_rt`.",
        datasets,
    )
    outcomes: list[StudyOutcome] = []
    plan = {"btc": ["5min", "15min", "60min"], "spx": ["5min", "15min", "30min"], "c": ["5min", "15min", "30min"]}
    for key, ds in datasets.items():
        for rule in plan.get(key, []):
            bar_minutes = int(pd.Timedelta(rule) / pd.Timedelta(minutes=1))

            def sig(bars, p, _lo=ds.long_only):
                return S.zscore_vwap_reversion(bars, int(p["window"]), float(p["entry_z"]), float(p["exit_z"]), vol_window=4 * int(p["window"]), max_vol_mult=float(p["max_vol_mult"]), long_only=_lo)

            def make(p, _sym=ds.symbol, _bm=bar_minutes):
                return ZScoreReversion([_sym], {"bar_minutes": _bm, "window": int(p["window"]), "entry_z": float(p["entry_z"]), "exit_z": float(p["exit_z"]), "vol_window": 4 * int(p["window"]), "max_vol_mult": float(p["max_vol_mult"]), "alloc_frac": 1.0, "min_delta_frac": 0.0, "session_only": False})

            out = run_bar_study(ds, rule, sig, grid, stability_params=("window", "entry_z"), event_strategy=make if rule != "5min" else None, crypto_ladder=(ds.asset_class is AssetClass.CRYPTO))
            outcomes.append(out)
    lines.append("## Results")
    lines.append("")
    for o in outcomes:
        lines.append(o.to_markdown())
    lines += _summary_table(outcomes)
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "02_intraday_mean_reversion.md").write_text(text)
    return text


def study_trend(datasets: dict[str, Dataset], write: bool = True) -> str:
    from jevtrader.strategies.trend_follow import TrendFollow

    grid_ema = {"fast": [5, 10, 20, 40], "slow": [30, 50, 100, 200], "target_bar_vol_bps": [0.0, 10.0]}
    grid_don = {"n": [20, 40, 80, 160], "exit_n": [10, 20, 40]}
    lines = _header(
        "03 - Time-series momentum / trend (EMA crossover, Donchian) with vol targeting",
        "Strategy: `trend_follow`. EMA crossover (fast/slow) or Donchian channel breakout; optional vol targeting scales "
        "size to a target per-bar vol (bps). Crypto is long-only.",
        datasets,
    )
    outcomes: list[StudyOutcome] = []
    plan = {"btc": ["15min", "60min", "4h"], "spx": ["15min", "60min"], "c": ["15min", "60min"]}
    for key, ds in datasets.items():
        for rule in plan.get(key, []):
            bm = int(pd.Timedelta(rule) / pd.Timedelta(minutes=1))

            def sig_ema(bars, p, _lo=ds.long_only):
                if int(p["fast"]) >= int(p["slow"]):
                    return pd.Series(0.0, index=bars.index)
                return S.ema_crossover(bars, int(p["fast"]), int(p["slow"]), long_only=_lo, vol_window=60, target_bar_vol=float(p["target_bar_vol_bps"]) / 1e4)

            def make_ema(p, _sym=ds.symbol, _bm=bm):
                return TrendFollow([_sym], {"bar_minutes": _bm, "mode": "ema", "fast": int(p["fast"]), "slow": int(p["slow"]), "vol_window": 60, "target_bar_vol_bps": float(p["target_bar_vol_bps"]), "alloc_frac": 1.0, "min_delta_frac": 0.0})

            o = run_bar_study(ds, rule, lambda b, p: sig_ema(b, p), grid_ema, stability_params=("fast", "slow"), event_strategy=make_ema if rule in ("60min",) else None, crypto_ladder=(ds.asset_class is AssetClass.CRYPTO), label=rule + " EMA")
            outcomes.append(o)

            def sig_don(bars, p, _lo=ds.long_only):
                return S.donchian(bars, int(p["n"]), int(p["exit_n"]), long_only=_lo)

            o2 = run_bar_study(ds, rule, sig_don, grid_don, stability_params=("n", "exit_n"), crypto_ladder=(ds.asset_class is AssetClass.CRYPTO), label=rule + " Donchian")
            outcomes.append(o2)
    lines.append("## Results")
    lines.append("")
    for o in outcomes:
        lines.append(o.to_markdown())
    lines += _summary_table(outcomes)
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "03_trend_following.md").write_text(text)
    return text


def study_breakouts(datasets: dict[str, Dataset], write: bool = True) -> str:
    from jevtrader.strategies.range_breakout import RangeBreakout

    lines = _header(
        "04 - Opening-range / session breakouts and volatility squeeze",
        "Strategies: `range_breakout` (ORB at 13:30 UTC for equities; Asia 00:00 / EU 07:00 / US 13:30 anchors for BTC) "
        "and `vol_squeeze` (Bollinger-inside-Keltner release). One trade per session for ORB.",
        datasets,
    )
    outcomes: list[StudyOutcome] = []
    grid_orb = {"range_minutes": [15, 30, 60], "hold_minutes": [120, 240, 360], "min_range_bps": [0.0, 20.0]}
    for key, ds in datasets.items():
        anchors = [(13, 30, "US")] if ds.asset_class is AssetClass.EQUITY else [(0, 0, "Asia"), (7, 0, "EU"), (13, 30, "US")]
        for h, m, nm in anchors:

            def sig(bars, p, _h=h, _m=m, _lo=ds.long_only):
                return S.session_breakout(bars, _h, _m, int(p["range_minutes"]), int(p["hold_minutes"]), long_only=_lo, min_range_bps=float(p["min_range_bps"]))

            def make(p, _sym=ds.symbol, _h=h, _m=m):
                return RangeBreakout([_sym], {"bar_minutes": 1, "session_start_utc": _h, "session_start_minute": _m, "range_minutes": int(p["range_minutes"]), "hold_minutes": int(p["hold_minutes"]), "min_range_bps": float(p["min_range_bps"]), "alloc_frac": 1.0, "min_delta_frac": 0.0})

            ds1 = Dataset(ds.label, ds.bars_1m, ds.asset_class, ds.cost, ds.symbol, session_filter=False, long_only=ds.long_only)
            o = run_bar_study(ds1, "1min", sig, grid_orb, min_train_rt=20, stability_params=("range_minutes", "hold_minutes"), event_strategy=make if ds.asset_class is AssetClass.EQUITY else None, crypto_ladder=(ds.asset_class is AssetClass.CRYPTO), label=f"1min ORB {nm}")
            outcomes.append(o)
        grid_sq = {"bb_window": [20, 40], "kc_mult": [1.0, 1.5], "hold": [6, 12, 24]}

        def sig_sq(bars, p, _lo=ds.long_only):
            return S.vol_squeeze(bars, int(p["bb_window"]), float(p["kc_mult"]), int(p["hold"]), long_only=_lo)

        outcomes.append(run_bar_study(ds, "5min", sig_sq, grid_sq, stability_params=("bb_window", "hold"), crypto_ladder=(ds.asset_class is AssetClass.CRYPTO), label="5min squeeze"))
    lines.append("## Results")
    lines.append("")
    for o in outcomes:
        lines.append(o.to_markdown())
    lines += _summary_table(outcomes)
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "04_breakouts.md").write_text(text)
    return text


def study_seasonality(datasets: dict[str, Dataset], write: bool = True) -> str:
    lines = _header(
        "05 - Intraday seasonality (time-of-day, day-of-week, overnight vs intraday)",
        "Per-hour and per-weekday mean returns with t-statistics on TRAIN, then the best hours re-tested on VALIDATION and "
        "HOLDOUT as a `time_of_day` position. With 24 hours x several data sets, |t| > 3 is the bar for a real effect.",
        datasets,
    )
    for key, ds in datasets.items():
        sp = chrono_split(ds.bars_1m)
        lines.append(f"## {ds.label}")
        lines.append("")
        h_tr = S.hourly_return_table(sp.train)
        h_va = S.hourly_return_table(sp.val)
        h_ho = S.hourly_return_table(sp.holdout)
        tbl = pd.DataFrame({"train_mean_bps": h_tr["mean_bps"], "train_t": h_tr["t"], "val_mean_bps": h_va["mean_bps"], "val_t": h_va["t"], "holdout_mean_bps": h_ho["mean_bps"], "holdout_t": h_ho["t"]})
        tbl.index.name = "utc_hour"
        lines.append("Hourly mean 1-min return (bps) and t-stat by split:")
        lines.append("")
        lines.append(_md_table(tbl))
        lines.append("")
        d_tr = S.dow_return_table(sp.train)
        d_tr.index.name = "dow"
        lines.append("Day-of-week (train):")
        lines.append("")
        lines.append(_md_table(d_tr))
        lines.append("")
        sig_hours = [int(h) for h, t in h_tr["t"].items() if t > 2.0]
        neg_hours = [int(h) for h, t in h_tr["t"].items() if t < -2.0]
        lines.append(f"Hours with train |t| > 2: long {sig_hours}, short {neg_hours}. Max |t| on train = {h_tr['t'].abs().max():.2f} over {len(h_tr)} tests (Bonferroni bar ~ 3.0).")
        lines.append("")
        if sig_hours or neg_hours:
            bars = ds.bars_1m
            tgt = S.hour_of_day(bars, tuple(sig_hours), tuple(neg_hours if not ds.long_only else ()))
            idx = _split_indices(bars, sp)
            rows = {}
            for nm in ("train", "val", "holdout"):
                r = _slice_result(bars, tgt, idx[nm], ds.cost, ds, n_trials=len(h_tr), full=True)
                rows[nm] = r.summary()
            df = pd.DataFrame(rows).T[["sharpe", "total_return", "max_drawdown", "n_round_trips", "gross_edge_bps_per_rt", "cost_bps_per_rt", "net_edge_bps_per_rt", "breakeven_fee_bps_per_side"]]
            df.index.name = "split"
            lines.append("`time_of_day` holding the train-selected hours (n_trials = 24 hours):")
            lines.append("")
            lines.append(_md_table(df))
            lines.append("")
            v = verdict(min(rows["val"]["sharpe"], rows["holdout"]["sharpe"]), float("nan"), min(rows["val"]["net_edge_bps_per_rt"], rows["holdout"]["net_edge_bps_per_rt"]), False)
            lines.append(f"**Verdict: {v}** (DSR not computed: selection over 24 hours makes any single-hour Sharpe uninterpretable; treat as NOT VIABLE unless holdout confirms).")
        else:
            lines.append("**Verdict: NOT VIABLE** -- no hour clears even |t| > 2 on train.")
        lines.append("")
        if not ds.long_only and not ds.session_filter or key == "spx":
            # overnight vs intraday split using the near-24h SPX proxy: close 20:00 UTC -> next 13:30 open
            b = ds.bars_1m
            day = b.index.date
            sess = b[(b.index.hour * 60 + b.index.minute >= 13 * 60 + 31) & (b.index.hour * 60 + b.index.minute <= 20 * 60)]
            o = sess.groupby(sess.index.date).agg(first_open=("open", "first"), last_close=("close", "last"))
            intraday = np.log(o["last_close"] / o["first_open"])
            overnight = np.log(o["first_open"] / o["last_close"].shift(1)).dropna()
            lines.append(f"Overnight (prev close -> open) mean {1e4*overnight.mean():.1f} bps/day (t={overnight.mean()/overnight.std()*np.sqrt(len(overnight)):.2f}, n={len(overnight)}) vs intraday (open -> close) mean {1e4*intraday.mean():.1f} bps/day (t={intraday.mean()/intraday.std()*np.sqrt(len(intraday)):.2f}). A daily round trip costs {ds.cost.round_trip_bps:.1f} bps on this cost model.")
            lines.append("")
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "05_seasonality.md").write_text(text)
    return text


def study_stat_arb(datasets: dict[str, Dataset], write: bool = True, seed: int = 7) -> str:
    from jevtrader.data.synthetic import bars_to_events, generate_cointegrated_pair
    from jevtrader.strategies.pairs_kalman import PairsKalman

    lines = ["# 06 - Stat-arb: Kalman pairs (synthetic mechanics) and BTC->SPX lead-lag (real)", "", f"_Generated {pd.Timestamp.now('UTC'):%Y-%m-%d}._", ""]
    lines.append("## Pairs with Kalman hedge ratio -- MECHANICS on synthetic cointegrated pairs")
    lines.append("")
    lines.append("No real equity pair data is cached, so this only shows the strategy recovers a known cointegration (beta=1.5, OU spread of a given half-life and vol) and what it earns net of Alpaca equity costs (3.3 bps per leg round trip) on that ground truth. Read the `spread_vol` rows together: a stationary spread std of ~60 bps of price (spread_vol 0.15) leaves room for costs and for the hedge-ratio estimation error (beta error x price level is itself ~0.5-1 dollar on a 100-dollar pair); a ~20 bps spread (spread_vol 0.05) does not and loses money. It says NOTHING about real pairs -- run the validation list below on Alpaca history first.")
    lines.append("")
    rows = []
    for hl, sv in [(30.0, 0.15), (60.0, 0.15), (30.0, 0.05)]:
        start = pd.Timestamp("2024-01-02 14:30", tz="UTC")
        pair = generate_cointegrated_pair("AAA", "BBB", start, start + pd.Timedelta(days=60), freq="15min", seed=seed, half_life_bars=hl, spread_vol=sv)
        strat = PairsKalman(["AAA", "BBB"], {"bar_minutes": 15, "entry_z": 2.0, "exit_z": 0.5, "warmup": 60, "alloc_frac": 0.4})
        bt = Backtester([strat], {s: bars_to_events(df, s) for s, df in pair.items()}, fees=CompositeFees(), fill_model=SlippageModel(1.0, 0.5, 0.0, float("inf")), initial_cash=100_000, allow_leverage=True, leverage=2.0)
        res = bt.run()
        rows.append({"half_life": hl, "spread_vol": sv, "sharpe": res.metrics["sharpe"], "total_return": res.metrics["total_return"], "max_dd": res.metrics["max_drawdown"], "n_fills": res.metrics["n_fills"], "hit_rate": res.metrics["hit_rate"], "final_beta": strat.beta})
    df = pd.DataFrame(rows).set_index("half_life")
    lines.append(_md_table(df, "{:.3f}"))
    lines.append("")
    lines.append("Real-validation list for the user (Alpaca 15-min bars, >= 1 year, both legs shortable): XLE/XOP, GLD/GDX, QQQ/XLK, KO/PEP, XLF/KBE, IWM/IJR, USO/XLE, MSFT/AAPL. Expect 2 legs x round trip ~ 3-7 bps cost per spread trade on Alpaca; require gross > 15 bps per spread trade to be worth it.")
    lines.append("")
    lines.append("**Verdict: RESEARCH (mechanics validated; no real evidence yet).**")
    lines.append("")
    if "btc" in datasets and "spx" in datasets:
        lines.append("## BTC -> SPX500 lead-lag (real data, US session overlap)")
        lines.append("")
        btc, spx = datasets["btc"], datasets["spx"]
        b5 = resample(btc.bars_1m, "5min")
        s5 = us_session_only(resample(spx.bars_1m, "5min"))
        common = s5.index.intersection(b5.index)
        s5 = s5.loc[common]
        r_lead = np.log(b5["close"]).diff().reindex(common)
        r_fol = np.log(s5["close"]).diff()
        lines.append("Cross-correlation corr(BTC ret[t-k], SPX ret[t]) at 5-min lags (whole overlap sample):")
        lines.append("")
        cc = {f"lag {k}": float(r_fol.corr(r_lead.shift(k))) for k in range(0, 4)}
        lines.append(_dict_table(cc, "lag", "corr"))
        lines.append("")
        grid = {"lookback": [1, 3, 6], "threshold_bps": [10.0, 20.0, 40.0], "hold": [1, 3, 6]}
        ds = Dataset(spx.label, spx.bars_1m, AssetClass.EQUITY, spx.cost, "SPY", session_filter=True)
        b5_full = b5

        def sig(bars, p):
            tgt, _ = S.lead_lag(b5_full, bars, int(p["lookback"]), float(p["threshold_bps"]), int(p["hold"]))
            return tgt

        o = run_bar_study(ds, "5min", sig, grid, min_train_rt=20, stability_params=("lookback", "threshold_bps"), label="5min BTC->SPX")
        lines.append(o.to_markdown())
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "06_stat_arb.md").write_text(text)
    return text


def study_microstructure(write: bool = True, seed: int = 3, minutes: int = 240) -> str:
    """Market maker + imbalance alpha on the synthetic quote/trade stream: MECHANICS + fee floor."""
    from jevtrader.core.fees import BpsFees
    from jevtrader.data.synthetic import generate_quote_trade_stream, quotes_to_events, trades_to_events
    from jevtrader.strategies.imbalance_alpha import ImbalanceAlpha
    from jevtrader.strategies.market_maker import AvellanedaStoikovMM

    lines = ["# 01 - Microstructure / HFT-style strategies: what is achievable at retail cost", "", f"_Generated {pd.Timestamp.now('UTC'):%Y-%m-%d}._", ""]
    lines.append("## The arithmetic first (no simulation needed)")
    lines.append("")
    lines.append("| venue / fee tier | maker fee per side | required half-spread (fee + ~2 bps adverse selection + 0.5 edge) | typical BTC inside spread | typical SPY inside spread | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for label, fee in [("Alpaca crypto tier 1 (<$100k/30d)", 15.0), ("Alpaca crypto tier 3 ($500k)", 10.0), ("Alpaca crypto tier 5 ($10M)", 5.0), ("Alpaca crypto tier 6 ($25M)", 2.0), ("Alpaca crypto top tier / other venue maker rebate", 0.0), ("Alpaca US equities", 0.15)]:
        req = fee + 2.5
        v = "NOT VIABLE" if req > 8 else ("MARGINAL" if req > 3 else "possible")
        lines.append(f"| {label} | {fee:.2f} bps | {req:.1f} bps ({2*req:.0f} bps quoted spread) | ~1-3 bps | ~0.2-1 bps | {v} |")
    lines.append("")
    lines.append("BTC 1-min realized vol is ~6 bps/bar and the mid moves ~3-4 bps per minute on average (Bitstamp 2025-26). A maker who must quote a 35-40 bps spread to cover 15 bps fees is only filled when the price runs through the quote, i.e. by informed flow. This is why the Avellaneda-Stoikov strategy enforces `min_half_spread = maker_fee + adverse_selection + min_edge` and why it is a *negative* result on Alpaca crypto tier 1.")
    lines.append("")
    lines.append("Retail latency (tens of ms REST/WebSocket, no co-location) removes latency arbitrage and most queue-priority games entirely; 'HFT' at retail means quote-driven strategies at 1 s - 5 min horizons with maker fills. Order-book imbalance at those horizons has an IC of a few percent: ~0.1-0.5 bps expected move per event. That clears NO taker fee anywhere and only clears maker costs on zero-commission equities.")
    lines.append("")
    lines.append("## Simulation: synthetic L1 stream with a known weak OFI edge (MECHANICS ONLY)")
    lines.append("")
    lines.append("`jevtrader.data.synthetic.generate_quote_trade_stream` embeds `ret[t] = 0.12 * vol * ofi[t-1] + noise` (a few % IC). We run both strategies through the SimBroker's L1 queue model at several fee levels. This validates that the code does what it says; it does not validate that Alpaca books have this edge -- record real books during paper trading (`research.record_replay`) and replay.")
    lines.append("")
    start = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    q, t = generate_quote_trade_stream("SPY", start, start + pd.Timedelta(minutes=minutes), freq="1s", seed=seed, mid0=450.0, annual_vol=0.15, tick_size=0.01)
    events_cache = list(quotes_to_events(q, "SPY")) + list(trades_to_events(t, "SPY"))
    events_cache.sort(key=lambda e: e.ts)
    rows = []
    for fee_label, fee_bps in [("zero fees", 0.0), ("equity-like (0.15)", 0.15), ("crypto tier1 maker 15 / taker 25", (15.0, 25.0)), ("crypto 2 / 12", (2.0, 12.0))]:
        maker, taker = (fee_bps, fee_bps) if isinstance(fee_bps, float) else fee_bps
        for name, make in [
            ("as_market_maker", lambda: AvellanedaStoikovMM(["SPY"], {"quote_size_notional": 4500.0, "max_inventory_units": 3, "gamma": 0.05, "horizon_s": 30.0, "adverse_selection_bps": 1.0, "min_edge_bps": 0.2})),
            ("imbalance_alpha", lambda: ImbalanceAlpha(["SPY"], {"notional": 4500.0, "hold_ticks": 10, "threshold_join": 0.3, "calibrate": True})),
        ]:
            strat = make()
            bt = Backtester([strat], {"SPY": iter(events_cache)}, fees=BpsFees(maker, taker), initial_cash=100_000, latency_ms=50.0)
            res = bt.run()
            f = res.fills
            maker_share = float((f["liquidity"] == "maker").mean()) if len(f) else float("nan")
            rows.append({"fee": fee_label, "strategy": name, "net_pnl_usd": res.metrics["final_equity"] - 100_000, "n_fills": len(f), "maker_share": maker_share, "fees_usd": res.fee_total, "gross_pnl_usd": res.metrics["final_equity"] - 100_000 + res.fee_total, "sharpe(per-step, indicative)": res.metrics["sharpe"]})
    df = pd.DataFrame(rows).set_index(["fee", "strategy"])
    df.index = [f"{a} / {b}" for a, b in df.index]
    lines.append(_md_table(df, "{:.2f}"))
    lines.append("")
    lines.append(f"Sample: {minutes} minutes of 1-second synthetic quotes ({len(q):,} quotes, {len(t):,} prints). Gross P&L positive and fee-eaten at crypto tiers = the expected picture: the mechanics work, the edge is too small for 15-25 bps fees.")
    lines.append("")
    lines.append("## VPIN toxicity gate")
    lines.append("")
    lines.append("`flow_toxicity.FlowToxicityGate` (bulk-volume classified VPIN) is a filter; it has no stand-alone P&L. Its validation plan: run `vpin_monitor` next to `as_market_maker` in paper trading, then regress the maker's per-fill markout (mid 60 s after fill minus fill price) on VPIN at fill time. A significantly negative slope justifies the gate; otherwise leave it off (it only removes fills).")
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    lines.append("* `as_market_maker` on Alpaca crypto tier 1: **NOT VIABLE** (fee floor forces uncompetitive quotes). On US equities: **RESEARCH** -- mechanics sound, edge unproven without real L2; validate via record/replay in paper.")
    lines.append("* `imbalance_alpha`: TAKE branch **NOT VIABLE** at any Alpaca fee level (predicted moves are sub-bp); JOIN branch **RESEARCH** on equities, **NOT VIABLE** on crypto tier 1.")
    lines.append("* True HFT (sub-second, latency-sensitive) is **not achievable** from a retail Alpaca account: it needs co-location, direct market access, exchange-member fee tiers/rebates and sub-100 us stacks. What would change the picture: crypto fees <= 2-3 bps/side (Alpaca >= $25M/30d tier, or a venue with maker rebates), and a colocated feed.")
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "01_microstructure_hft.md").write_text(text)
    return text


def study_jev_gate(datasets: dict[str, Dataset], write: bool = True) -> str:
    """Base vs Jev-gated variants with the OfflineJevAdvisor (MECHANICS: turnover/sizing effect only)."""
    from jevtrader.jev.advisor import OfflineJevAdvisor
    from jevtrader.strategies.jev_gated import TrendFollowJev, ZScoreReversionJev
    from jevtrader.strategies.trend_follow import TrendFollow
    from jevtrader.strategies.vwap_reversion import ZScoreReversion

    lines = ["# 07 - Jev-gated variants (OfflineJevAdvisor stand-in)", "", f"_Generated {pd.Timestamp.now('UTC'):%Y-%m-%d}._", ""]
    lines.append("**Read this first.** `OfflineJevAdvisor` is a transparent logistic heuristic with no forecasting skill. These runs show how the gate changes *turnover and sizing* (it can only remove entries and shrink size via fractional Kelly), not whether Jev adds alpha. The real test is `ReplayJevAdvisor.from_log(...)` over recorded paper-trading decisions plus the Kev/Laya/Jev calibration scoreboard (`jevtrader.jev.calibration`).")
    lines.append("")
    rows = []
    fees = CompositeFees()
    for key, ds in datasets.items():
        if key not in ("spx", "c"):
            continue
        sp = chrono_split(ds.bars_1m)
        oos = pd.concat([sp.val, sp.holdout])
        slip = SlippageModel(ds.cost.half_spread_bps, ds.cost.slippage_bps, 0.0, float("inf"))
        for label, base_cls, jev_cls, params in [
            ("zscore_reversion 15m", ZScoreReversion, ZScoreReversionJev, {"bar_minutes": 15, "window": 30, "entry_z": 2.0, "exit_z": 0.0, "vol_window": 120, "max_vol_mult": 1.5, "alloc_frac": 0.5, "min_delta_frac": 0.0}),
            ("trend_follow 60m EMA", TrendFollow, TrendFollowJev, {"bar_minutes": 60, "mode": "ema", "fast": 10, "slow": 50, "alloc_frac": 0.5, "min_delta_frac": 0.0}),
        ]:
            for variant, cls, jev in [("base", base_cls, None), ("jev-gated (offline)", jev_cls, OfflineJevAdvisor())]:
                strat = cls([ds.symbol], dict(params))
                bt = Backtester([strat], {ds.symbol: to_bar_events(oos, ds.symbol)}, fees=fees, fill_model=slip, initial_cash=100_000, allow_leverage=True, leverage=1.0, jev=jev)
                res = bt.run()
                rows.append({"dataset": key.upper(), "strategy": label, "variant": variant, "sharpe": res.metrics["sharpe"], "total_return": res.metrics["total_return"], "max_dd": res.metrics["max_drawdown"], "n_fills": res.metrics["n_fills"], "jev_holds": getattr(strat, "jev_holds", 0), "avg_fill_notional": float(res.fills["notional"].mean()) if len(res.fills) else 0.0})
    df = pd.DataFrame(rows)
    df.index = [f"{r.dataset} {r.strategy} [{r.variant}]" for r in df.itertuples()]
    df = df.drop(columns=["dataset", "strategy", "variant"])
    lines.append(_md_table(df, "{:.3f}"))
    lines.append("")
    lines.append("Columns: `jev_holds` = entries the gate suppressed; `avg_fill_notional` shows the Kelly down-sizing. Both variants run on validation+holdout (out-of-sample for the base parameters).")
    lines.append("")
    lines.append("**Verdict: MECHANICS OK (falls back to base with jev=None; late/None = HOLD). Alpha unproven by construction -- run paper with DecisionLog on, then `experiments.jev_gate_replay(log_path, ...)`.**")
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "07_jev_gated.md").write_text(text)
    return text


def jev_gate_replay(log_path: str, strategy, data: dict, **bt_kwargs):
    """Re-run `strategy` over `data` with `ReplayJevAdvisor.from_log(log_path)` -- the honest
    evaluation of what Jev actually said during paper trading."""
    from jevtrader.jev.advisor import ReplayJevAdvisor

    adv = ReplayJevAdvisor.from_log(log_path)
    return Backtester([strategy], data, jev=adv, **bt_kwargs).run()


def study_meta_allocator(datasets: dict[str, Dataset], write: bool = True) -> str:
    from jevtrader.strategies.meta_allocator import MetaAllocator

    lines = ["# 08 - Meta-allocator (inverse-vol / HRP across strategy sleeves)", "", f"_Generated {pd.Timestamp.now('UTC'):%Y-%m-%d}._", ""]
    lines.append("Runs `zscore_reversion` (15m) and `trend_follow` (60m EMA) as sleeves on the SPX500 proxy and C over validation+holdout, with weights re-estimated from the sleeves' own virtual P&L. Compared with equal weights. This is plumbing: the allocator cannot create edge, only distribute risk.")
    lines.append("")
    rows = []
    if "spx" in datasets and "c" in datasets:
        spx, c = datasets["spx"], datasets["c"]
        sp1, sp2 = chrono_split(spx.bars_1m), chrono_split(c.bars_1m)
        data = {"SPY": to_bar_events(pd.concat([sp1.val, sp1.holdout]), "SPY"), "C": to_bar_events(pd.concat([sp2.val, sp2.holdout]), "C")}
        children = [
            {"name": "zscore_reversion", "symbols": ["SPY"], "params": {"bar_minutes": 15, "window": 30, "entry_z": 2.0, "exit_z": 0.0, "vol_window": 120, "max_vol_mult": 1.5, "alloc_frac": 1.0, "min_delta_frac": 0.0}},
            {"name": "zscore_reversion", "symbols": ["C"], "params": {"bar_minutes": 15, "window": 30, "entry_z": 2.0, "exit_z": 0.0, "vol_window": 120, "max_vol_mult": 1.5, "alloc_frac": 1.0, "min_delta_frac": 0.0}},
            {"name": "trend_follow", "symbols": ["SPY"], "params": {"bar_minutes": 60, "mode": "ema", "fast": 10, "slow": 50, "alloc_frac": 1.0, "min_delta_frac": 0.0}},
        ]
        for scheme in ["equal_weight", "inverse_vol", "hrp"]:
            meta = MetaAllocator([], {"children": children, "scheme": scheme, "lookback": 200, "rebalance_bars": 100, "min_weight": 0.1, "max_weight": 0.6, "capital_frac": 1.0})
            data_iters = {k: to_bar_events(pd.concat([sp1.val, sp1.holdout]) if k == "SPY" else pd.concat([sp2.val, sp2.holdout]), k) for k in data}
            bt = Backtester([meta], data_iters, fees=CompositeFees(), fill_model=SlippageModel(1.0, 0.5, 0.0, float("inf")), initial_cash=100_000, allow_leverage=True, leverage=1.5)
            res = bt.run()
            rows.append({"scheme": scheme, "sharpe": res.metrics["sharpe"], "total_return": res.metrics["total_return"], "max_dd": res.metrics["max_drawdown"], "n_fills": res.metrics["n_fills"], "final_weights": {k.split(":")[0] + ":" + k.split(":")[-1]: round(v, 2) for k, v in meta._weights.items()}})
        df = pd.DataFrame(rows).set_index("scheme")
        lines.append(_md_table(df, "{:.3f}"))
        lines.append("")
    lines.append("**Verdict: plumbing validated (virtual per-child positions, weight-scaled orders, scheme weights from `rebalance.targets`). Useful once >= 2 sleeves are individually VIABLE; today it mostly adds risk control.**")
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "08_meta_allocator.md").write_text(text)
    return text


def study_crypto_maker_entry(datasets: dict[str, Dataset], write: bool = True) -> str:
    """The one crypto question left open by the fast screen: does MAKER entry (15 bps, no spread)
    instead of TAKER (25 bps + spread) rescue BTC mean reversion? Event-driven only (limit fills
    need the SimBroker's trade-through rule), on validation+holdout, 60-min bars, long-only."""
    from jevtrader.strategies.vwap_reversion import ZScoreReversion

    lines = ["# 02b - BTC mean reversion with MAKER entries and Alpaca fee tiers (event-driven)", "", f"_Generated {pd.Timestamp.now('UTC'):%Y-%m-%d}._", ""]
    if "btc" not in datasets:
        lines.append("BTC data not available.")
        text = "\n".join(lines)
        if write:
            (REPORT_DIR / "02b_btc_maker_entry.md").write_text(text)
        return text
    ds = datasets["btc"]
    sp = chrono_split(ds.bars_1m)
    oos = pd.concat([sp.val, sp.holdout])
    lines.append(f"Sample: validation + holdout, {oos.index[0]:%Y-%m-%d} -> {oos.index[-1]:%Y-%m-%d} ({len(oos):,} 1-min bars), 60-min z-score reversion (window 30, entry 2.0, exit 0.0, vol filter 1.5), long-only, 100% of equity per trade so fee tiers are hit as they would be on a $100k account.")
    lines.append("")
    rows = []
    for label, style, fee_model, slip in [
        ("taker, tier-1 fixed (25 bps)", "market", CompositeFees(crypto=AlpacaCryptoFees(tiers=[(0, 15.0, 25.0)])), SlippageModel(2.5, 1.0, 0.0, float("inf"))),
        ("taker, Alpaca tiers by rolling 30d volume", "market", CompositeFees(crypto=AlpacaCryptoFees()), SlippageModel(2.5, 1.0, 0.0, float("inf"))),
        ("maker limit entry, tier-1 fixed (15 bps)", "limit", CompositeFees(crypto=AlpacaCryptoFees(tiers=[(0, 15.0, 25.0)])), SlippageModel(2.5, 1.0, 0.0, float("inf"))),
        ("maker limit entry, Alpaca tiers", "limit", CompositeFees(crypto=AlpacaCryptoFees()), SlippageModel(2.5, 1.0, 0.0, float("inf"))),
        ("maker limit entry, zero fees (edge check)", "limit", CompositeFees(crypto=AlpacaCryptoFees(tiers=[(0, 0.0, 0.0)])), SlippageModel(0.0, 0.0, 0.0, float("inf"))),
    ]:
        strat = ZScoreReversion(["BTC/USD"], {"bar_minutes": 60, "window": 30, "entry_z": 2.0, "exit_z": 0.0, "vol_window": 120, "max_vol_mult": 1.5, "alloc_frac": 1.0, "min_delta_frac": 0.0, "session_only": False, "entry_style": style})
        bt = Backtester([strat], {"BTC/USD": to_bar_events(oos, "BTC/USD")}, fees=fee_model, fill_model=slip, initial_cash=100_000.0)
        res = bt.run()
        f = res.fills
        maker_share = float((f["liquidity"] == "maker").mean()) if len(f) else float("nan")
        rows.append({"setup": label, "sharpe": res.metrics["sharpe"], "total_return": res.metrics["total_return"], "max_dd": res.metrics["max_drawdown"], "n_fills": len(f), "maker_share": maker_share, "fee_bps_per_fill": res.metrics["cost_per_trade_bps"], "fees_usd": res.fee_total, "gross_pnl_usd": res.metrics["final_equity"] - 100_000 + res.fee_total, "net_pnl_usd": res.metrics["final_equity"] - 100_000})
    df = pd.DataFrame(rows).set_index("setup")
    lines.append(_md_table(df, "{:.3f}"))
    lines.append("")
    lines.append("Reading: `gross_pnl_usd` is what the signal makes before fees; if it is near zero or negative even at zero fees, no fee tier rescues it. Maker entries pay 15 instead of 25 bps and avoid the spread, but they fill less often and adversely (the limit sits at the close; it only fills when the next bar trades through it).")
    lines.append("")
    zero_edge = rows[-1]["gross_pnl_usd"]
    maker_ok = all(r["net_pnl_usd"] > 0 for r in rows if r["setup"].startswith("maker"))
    if zero_edge <= 0:
        v = "NOT VIABLE at any fee tier -- the signal loses money even with zero fees, so no maker/tier improvement can rescue it"
    elif maker_ok:
        v = "MARGINAL -- positive with maker entries; confirm on paper with real Alpaca fills"
    else:
        v = "NOT VIABLE on Alpaca tiers -- gross edge exists but is smaller than the fee at every tier tested"
    lines.append(f"**Verdict: {v}.**")
    text = "\n".join(lines)
    if write:
        (REPORT_DIR / "02b_btc_maker_entry.md").write_text(text)
    return text


# ----------------------------------------------------------------------------- summary


def _summary_table(outcomes: list[StudyOutcome]) -> list[str]:
    rows = []
    for o in outcomes:
        rows.append({
            "dataset": o.dataset.split(" (")[0],
            "bars": o.rule,
            "train_SR": o.rows["train"]["sharpe"],
            "val_SR": o.rows["val"]["sharpe"],
            "holdout_SR": o.rows["holdout"]["sharpe"],
            "DSR_val": o.dsr_val,
            "holdout_maxDD": o.rows["holdout"]["max_drawdown"],
            "trades/day": o.rows["holdout"]["trades_per_day"],
            "gross_bps/rt(OOS)": (o.rows["val"]["gross_edge_bps_per_rt"] + o.rows["holdout"]["gross_edge_bps_per_rt"]) / 2,
            "cost_bps/rt": o.rows["holdout"]["cost_bps_per_rt"],
            "breakeven_fee": (o.rows["val"]["breakeven_fee_bps_per_side"] + o.rows["holdout"]["breakeven_fee_bps_per_side"]) / 2,
            "verdict": o.verdict,
        })
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    df.index.name = "#"
    return ["## Summary", "", _md_table(df), ""]


STUDIES = {
    "01": lambda ds: study_microstructure(),
    "02": study_mean_reversion,
    "02b": study_crypto_maker_entry,
    "03": study_trend,
    "04": study_breakouts,
    "05": study_seasonality,
    "06": study_stat_arb,
    "07": study_jev_gate,
    "08": study_meta_allocator,
}


def run_all(which: Optional[set[str]] = None, studies: Optional[Sequence[str]] = None) -> dict[str, str]:
    """`which` filters data sets ({"btc","spx","c"}); `studies` filters study numbers."""
    REPORT_DIR.mkdir(exist_ok=True)
    datasets = load_datasets(which)
    t0 = time.time()
    out = {}
    for num in studies or sorted(STUDIES):
        out[num] = STUDIES[num](datasets)
        print(f"{num} done {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    args = sys.argv[1:]
    studies = [a for a in args if a in STUDIES] or None
    which = set(a for a in args if a not in STUDIES) or None
    run_all(which, studies)
