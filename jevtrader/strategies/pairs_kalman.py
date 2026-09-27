"""Pairs / cointegration stat-arb with a Kalman-filter hedge ratio.

State x_t = [beta_t, alpha_t] follows a random walk; observation p_b,t = beta_t * p_a,t + alpha_t
+ e_t. The Kalman innovation (p_b - predicted) divided by its predicted std is the spread
z-score. Enter long-spread (buy B, sell beta*A) when z < -entry_z, short-spread when z > entry_z,
exit when |z| < exit_z or after `hold_max` bars. Equities only (needs shorting), dollar-neutral
by construction, both legs sized so the *spread* leg notional is `alloc_frac` x equity.

Why it might work: economically linked names (sector ETFs, dual listings, XLE/USO-type pairs)
share a common factor; idiosyncratic deviations revert. Why it fails: cointegration breaks
(index rebalances, corporate events); the Kalman filter adapts but is slow by design.

Validation status: MECHANICS validated on `jevtrader.data.synthetic.generate_cointegrated_pair`
(known ground truth, see tests + research_reports/06_stat_arb.md). We have no real equity pair
data in the cache: the user must run `experiments.pairs_real_validation_list()` symbols through
Alpaca history before trusting any number. Costs: 2 legs x round trip -> ~2x the single-name
cost; spread z-score entries at 2 sigma on 15-60 min bars typically yield 10-30 bps gross on
synthetic data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from jevtrader.core.registry import register
from jevtrader.core.strategy import StrategyContext, StrategySpec
from jevtrader.core.types import Bar, Order, OrderType, Side, TimeInForce
from jevtrader.risk.sizing import lot_round
from jevtrader.strategies._base import BarStrategy


class KalmanHedge:
    """2-state Kalman filter for y = beta*x + alpha with random-walk coefficients."""

    def __init__(self, delta: float = 1e-4, obs_var: float = 1.0) -> None:
        self.x = np.array([1.0, 0.0])
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * delta / (1.0 - delta)
        self.R = obs_var
        self.n = 0

    def update(self, px_a: float, px_b: float) -> tuple[float, float, float]:
        """Returns (innovation, innovation_std, beta) after processing one observation."""
        H = np.array([px_a, 1.0])
        P_pred = self.P + self.Q
        y_pred = float(H @ self.x)
        S = float(H @ P_pred @ H) + self.R
        innov = px_b - y_pred
        K = P_pred @ H / S
        self.x = self.x + K * innov
        self.P = P_pred - np.outer(K, H) @ P_pred
        self.n += 1
        # adaptive observation variance (EW of squared innovations) keeps z well scaled
        self.R = 0.98 * self.R + 0.02 * innov * innov if self.n > 10 else self.R
        return innov, float(np.sqrt(max(S, 1e-12))), float(self.x[0])


class PairsKalman(BarStrategy):
    spec = StrategySpec(
        name="pairs_kalman",
        description="Dollar-neutral pairs trading with a Kalman-filter hedge ratio and innovation z-score entries.",
        asset_classes=("equity",),
        frequency="15min",
        style="stat_arb",
        uses_jev=False,
        default_params={
            **BarStrategy.BASE_PARAMS,
            "bar_minutes": 15,
            "delta": 1e-4,
            "warmup": 60,
            "entry_z": 2.0,
            "exit_z": 0.5,
            "hold_max": 96,
            "alloc_frac": 0.25,
        },
        notes=(
            "MECHANICS validated on synthetic cointegrated pairs only (no real pair data cached). Equities only "
            "(needs a short leg). Two legs double the cost per spread round trip. Candidate real pairs for the "
            "user to validate with Alpaca data: XLE/XOP, GLD/GDX, SPY/IVV (too tight after costs), QQQ/XLK, "
            "KO/PEP, XLF/KBE. Verdict: RESEARCH until real validation."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None, **kw):
        super().__init__(symbols, params, strategy_id, **kw)
        if len(self.symbols) != 2:
            raise ValueError("pairs_kalman needs exactly two symbols [A, B]")
        self.kf = KalmanHedge(delta=float(self.params["delta"]))
        self._last: dict[str, tuple] = {}
        self._pending_ts = None
        self._state = 0  # +1 long spread (long B short A), -1 short spread
        self._held = 0
        self.last_z: Optional[float] = None
        self.beta: float = 1.0

    def _leg_orders(self, ctx: StrategyContext, direction: int, px_a: float, px_b: float) -> None:
        """direction +1: buy B / sell A; -1: sell B / buy A; 0: flatten both."""
        a, b = self.symbols
        equity = ctx.account().equity
        notional_b = float(self.params["alloc_frac"]) * equity
        tgt_b = direction * notional_b / px_b
        tgt_a = -direction * self.beta * notional_b / px_a if direction != 0 else 0.0
        for sym, tgt, px in ((b, tgt_b, px_b), (a, tgt_a, px_a)):
            cur = ctx.position(sym).qty
            delta = lot_round(tgt - cur, ctx.instrument(sym))
            if abs(delta) * px < ctx.instrument(sym).min_notional:
                continue
            side = Side.BUY if delta > 0 else Side.SELL
            reduce_only = direction == 0
            ctx.submit(Order(sym, side, abs(delta), OrderType.MARKET, tif=TimeInForce.IOC, reduce_only=reduce_only, strategy_id=self.id, tag="pair"))

    def on_agg_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        self._last[bar.symbol] = (bar.ts, bar.close)
        a, b = self.symbols
        if a not in self._last or b not in self._last or self._last[a][0] != self._last[b][0]:
            return  # wait until both legs have a bar at the same timestamp
        px_a, px_b = self._last[a][1], self._last[b][1]
        innov, sd, beta = self.kf.update(px_a, px_b)
        self.beta = beta
        if self.kf.n < int(self.params["warmup"]) or sd <= 0:
            return
        z = innov / sd
        self.last_z = z
        p = self.params
        if self._state != 0:
            self._held += 1
            if abs(z) < float(p["exit_z"]) or self._held >= int(p["hold_max"]) or (self._state > 0 and z > 0) or (self._state < 0 and z < 0):
                self._leg_orders(ctx, 0, px_a, px_b)
                self._state, self._held = 0, 0
            return
        if z < -float(p["entry_z"]):
            self._leg_orders(ctx, +1, px_a, px_b)
            self._state, self._held = +1, 0
        elif z > float(p["entry_z"]):
            self._leg_orders(ctx, -1, px_a, px_b)
            self._state, self._held = -1, 0


register(PairsKalman)
