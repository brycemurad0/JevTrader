"""SmartRebalanceBot: a Strategy that periodically rebalances a mixed equity+crypto book using
the schemes in targets.py, gated by the cost-aware policy in policy.py, and executed via the
planner in planner.py. Lives in jevtrader/rebalance (NOT jevtrader/strategies) per ownership,
but self-registers under the "smart_rebalance" name so it's importable into the strategy registry.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from jevtrader.core.fees import default_fees
from jevtrader.core.registry import register
from jevtrader.core.strategy import Strategy, StrategyContext, StrategySpec
from jevtrader.core.types import AssetClass, Bar, Instrument
from jevtrader.rebalance.planner import RebalancePlanner
from jevtrader.rebalance.policy import RebalancePolicy, calendar_due, decide
from jevtrader.rebalance.targets import cov_matrix, target_weights


class SmartRebalanceBot(Strategy):
    """Cost-aware, Jev-tilt-optional rebalancer across a mixed equity+crypto book.

    Params (see `spec.default_params`):
        scheme: one of jevtrader.rebalance.targets.SCHEMES ("inverse_vol", "hrp", "risk_parity",
            "min_variance", "equal_weight")
        lookback: number of bars used to build the returns DataFrame for the scheme + covariance
        min_weight / max_weight / cash_buffer_pct: passed to the scheme
        dry_run: if True, log the plan but never submit orders (default False - "dry-run first
            always" is achieved by running the bot once with dry_run=True before enabling it live)
        risky_symbols: subset of `symbols` eligible for the Jev regime tilt (default: crypto only)
        limit_offset_bps: passed to RebalancePlanner

    `self.policy` (a RebalancePolicy) governs tolerance bands, calendar cadence, the cost/benefit
    gate and the tilt bounds; replace it after construction to change those.
    """

    spec = StrategySpec(
        name="smart_rebalance",
        description=(
            "Cost-aware smart rebalancer: builds long-only target weights (equal weight, inverse "
            "vol, risk parity, HRP or min-variance) from a rolling returns window, only trades "
            "when a symbol has drifted outside its 5/25 tolerance band AND the expected "
            "tracking-error reduction exceeds the estimated trading cost, and optionally applies "
            "a bounded Jev regime tilt to risky assets."
        ),
        asset_classes=("equity", "crypto"),
        frequency="1d",
        style="rebalance",
        uses_jev=True,
        default_params={
            "scheme": "inverse_vol",
            "lookback": 90,
            "min_weight": 0.0,
            "max_weight": 0.40,
            "cash_buffer_pct": 0.02,
            "dry_run": False,
            "limit_offset_bps": 5.0,
            "risky_symbols": None,  # None -> infer crypto symbols via Instrument.infer
        },
        notes=(
            "NOT a return-seeking strategy - a risk-controlled, low-turnover way to hold a "
            "diversified book cheaply (fee reality makes turnover the enemy). Fails quietly (no "
            "trade) with fewer than `lookback` bars of history or missing quotes. Works with "
            "ctx.jev is None (pure quant); the Jev tilt only ever REDUCES risky-asset weight."
        ),
    )

    def __init__(self, symbols: list[str], params: Optional[dict[str, Any]] = None, strategy_id: Optional[str] = None):
        super().__init__(symbols, params, strategy_id)
        self.policy = RebalancePolicy()
        self._last_rebalance_ts: Optional[pd.Timestamp] = None
        self._fee_model = default_fees()

    # ------------------------------------------------------------------------- helpers

    def _risky_symbols(self) -> list[str]:
        explicit = self.params.get("risky_symbols")
        if explicit is not None:
            return list(explicit)
        return [s for s in self.symbols if Instrument.infer(s).asset_class is AssetClass.CRYPTO]

    def _build_returns(self, ctx: StrategyContext) -> Optional[pd.DataFrame]:
        lookback = int(self.params["lookback"])
        series = {}
        for sym in self.symbols:
            bars = ctx.bars(sym, lookback + 1)
            if bars is None or len(bars) < 2:
                return None
            series[sym] = bars["close"].pct_change().dropna()
        df = pd.DataFrame(series).dropna()
        if len(df) < 2:
            return None
        return df

    def _prices(self, ctx: StrategyContext) -> dict[str, float]:
        prices: dict[str, float] = {}
        for sym in self.symbols:
            quote = ctx.last_quote(sym)
            if quote is not None:
                prices[sym] = quote.mid
            else:
                # Bar-only feeds (daily/hourly backtests, bar streams) have no quotes: use last close.
                bars = ctx.bars(sym, 1)
                if bars is not None and len(bars):
                    prices[sym] = float(bars["close"].iloc[-1])
        return prices

    def _current_weights(self, ctx: StrategyContext, prices: dict[str, float], equity: float) -> dict[str, float]:
        weights: dict[str, float] = {}
        equity = equity or 1.0
        for sym in self.symbols:
            pos = ctx.position(sym)
            price = prices.get(sym)
            qty = pos.qty if pos is not None else 0.0
            weights[sym] = (qty * price) / equity if price else 0.0
        return weights

    # ------------------------------------------------------------------------- lifecycle

    def on_day_end(self, ctx: StrategyContext) -> None:
        if self.spec.frequency == "1d":
            self._maybe_rebalance(ctx)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        # Best-effort hourly cadence: only fire once per bar batch, on the first symbol's bar.
        if self.spec.frequency == "1h" and self.symbols and bar.symbol == self.symbols[0]:
            self._maybe_rebalance(ctx)

    # ------------------------------------------------------------------------- core

    def _maybe_rebalance(self, ctx: StrategyContext) -> None:
        if not calendar_due(self._last_rebalance_ts, ctx.now, self.policy.frequency):
            return

        returns_df = self._build_returns(ctx)
        if returns_df is None:
            ctx.log("smart_rebalance: not enough bar history yet, skipping")
            return

        prices = self._prices(ctx)
        missing = [s for s in self.symbols if s not in prices]
        if missing:
            ctx.log("smart_rebalance: missing quotes, skipping", missing=missing)
            return

        account = ctx.account()
        targets = target_weights(
            self.params["scheme"],
            returns_df,
            min_weight=self.params["min_weight"],
            max_weight=self.params["max_weight"],
            cash_buffer_pct=self.params["cash_buffer_pct"],
        ).to_dict()

        current_weights = self._current_weights(ctx, prices, account.equity)
        cov = cov_matrix(returns_df, shrinkage=True)
        instruments = {s: ctx.instrument(s) for s in self.symbols}

        risky = self._risky_symbols()
        jev_features = {s: {"symbol": s} for s in risky}

        decision = decide(
            self.policy,
            current_weights,
            targets,
            account.equity,
            cov,
            prices,
            instruments,
            self._fee_model,
            spread_bps=self.policy.default_spread_bps,
            jev=ctx.jev,
            risky_symbols=jev_features,
        )

        self._last_rebalance_ts = ctx.now
        for reason in decision.reasons:
            ctx.log(f"smart_rebalance: {reason}")

        if not decision.should_trade:
            return

        # Only hand the planner the symbols the policy actually approved for trading; every other
        # symbol keeps its current weight as its "target" so the planner computes zero delta for
        # it (the 5/25 band, not the planner, decides which symbols move).
        planner_targets = dict(current_weights)
        for sym in decision.trade_weights:
            planner_targets[sym] = decision.target_weights.get(sym, 0.0)

        planner = RebalancePlanner(instruments, self._fee_model, limit_offset_bps=self.params["limit_offset_bps"])
        plan = planner.plan(account, prices, planner_targets, now=ctx.now)
        ctx.log("smart_rebalance plan:\n" + plan.to_markdown())

        if self.params.get("dry_run"):
            ctx.log("smart_rebalance: dry_run=True, not submitting orders")
            return

        for order in plan.to_orders(strategy_id=self.id):
            ctx.submit(order)


register(SmartRebalanceBot)
