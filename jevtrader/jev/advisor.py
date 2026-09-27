"""`JevAdvisorProtocol` implementations.

- `JevAdvisor` / `AsyncJevAdvisor`: wrap `typesafe_sdk`'s sync/async clients with a latency
  budget, a short TTL cache, a token-bucket rate limiter and a circuit breaker, and NEVER raise
  into the trading loop -- every failure mode returns `None` (skip this cycle) or a `JevView`
  with `late=True` (treat as HOLD), and every real call is written to a `DecisionLog` if one is
  attached.
- `OfflineJevAdvisor`: a deterministic, offline, NOT-Jev logistic heuristic over momentum /
  mean-reversion / order-book-imbalance features, so strategies can be built and backtested
  without a network call or an API key.
- `ReplayJevAdvisor`: replays real (paper-trading) Jev decisions from a `DecisionLog` so a
  backtest can honestly evaluate what Jev actually said, instead of hindsight or a heuristic.
- `ShadowAdvisor`: answers with one primary advisor while firing the same question at other
  backends in the background purely to log a head-to-head comparison, for free, during paper
  trading.
- `make_advisor`: picks the right one for a given `Settings` + trading mode + decision backend.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping, Optional

from typesafe_sdk import TypeSafeError

from jevtrader.core.interfaces import JevView
from jevtrader.jev.backends import DecisionBackend, make_client
from jevtrader.jev.log import DecisionLog, read_jsonl
from jevtrader.jev.questions import direction_questions, regime_questions
from jevtrader.jev.state import to_json

logger = logging.getLogger("jevtrader.jev")

_DEFAULT_LATENCY_BUDGET_MS = 400
_DEFAULT_CACHE_TTL_S = 2.0
_DEFAULT_CACHE_MAXSIZE = 512
_DEFAULT_RATE_LIMIT_PER_SEC = 5.0
_DEFAULT_RATE_LIMIT_BURST = 10.0
_DEFAULT_CIRCUIT_MAX_ERRORS = 5
_DEFAULT_CIRCUIT_COOLDOWN_S = 30.0


# --------------------------------------------------------------------------- shared helpers


def _qfield(question: Any, name: str, default: Any = None) -> Any:
    """Read a field off either a `typesafe_sdk` question object or a raw dict question."""
    if isinstance(question, Mapping):
        return question.get(name, default)
    return getattr(question, name, default)


def _question_type(question: Any) -> str:
    return str(_qfield(question, "type", ""))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [v / total for v in exps]


def _cache_key(symbol: str, question_key: str, state: Mapping[str, Any]) -> tuple[str, str, str]:
    digest = hashlib.sha1(to_json(state).encode("utf-8")).hexdigest()[:16]
    return (symbol, question_key, digest)


def _view_from_response(resp: Any, latency_ms: float, *, late: bool = False) -> JevView:
    """Convert a `typesafe_sdk.SystemOneResponse` into a `JevView`.

    Noul -> `{"yes": p, "no": 1-p}`. Score -> `{"0": p0, "1": p1, ...}` with the expected
    (probability-weighted) value and rubric legend stashed in `raw[name]`.
    """
    probs: dict[str, dict[str, float]] = {}
    top: dict[str, str] = {}
    confidence: dict[str, float] = {}
    raw: dict[str, Any] = {"usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}}

    for name, answer in resp.choices.items():
        probs[name] = dict(answer.probabilities)
        top[name] = answer.choice
        confidence[name] = float(answer.confidence)

    for name, answer in resp.scores.items():
        probs[name] = {str(k): float(v) for k, v in answer.probabilities.items()}
        top[name] = str(round(answer.score))
        confidence[name] = float(answer.confidence)
        raw[name] = {"expected": float(answer.score), "legend": dict(answer.legend)}

    for name, answer in resp.nouls.items():
        p_yes = float(answer.noul)
        probs[name] = {"yes": p_yes, "no": 1.0 - p_yes}
        top[name] = "yes" if p_yes >= 0.5 else "no"
        confidence[name] = abs(p_yes - 0.5) * 2.0
        raw[name] = {"noul": p_yes}

    return JevView(probs=probs, top=top, confidence=confidence, latency_ms=latency_ms, late=late, model=resp.model, raw=raw)


# --------------------------------------------------------------------------- rate limit / breaker / cache


@dataclass
class _TokenBucket:
    rate_per_sec: float = _DEFAULT_RATE_LIMIT_PER_SEC
    capacity: float = _DEFAULT_RATE_LIMIT_BURST
    _tokens: float = field(init=False, default=0.0)
    _updated: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._updated = time.monotonic()

    def try_acquire(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self._updated
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
        self._updated = now
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False


@dataclass
class _CircuitBreaker:
    max_consecutive_errors: int = _DEFAULT_CIRCUIT_MAX_ERRORS
    cooldown_s: float = _DEFAULT_CIRCUIT_COOLDOWN_S
    consecutive_errors: int = field(init=False, default=0)
    _opened_at: Optional[float] = field(init=False, default=None)

    @property
    def open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            return False  # half-open: let the next call through as a trial
        return True

    def record_success(self) -> None:
        self.consecutive_errors = 0
        self._opened_at = None

    def record_error(self) -> None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors:
            # (Re)start the cooldown from now. A failure during a half-open trial (cooldown
            # already elapsed once) re-opens the breaker instead of leaving a stale timestamp
            # that would make every subsequent call look like a fresh trial.
            self._opened_at = time.monotonic()


class _TTLCache:
    def __init__(self, ttl_s: float, maxsize: int) -> None:
        self.ttl_s = ttl_s
        self.maxsize = maxsize
        self._store: dict[Any, tuple[float, JevView]] = {}

    def get(self, key: Any) -> Optional[JevView]:
        item = self._store.get(key)
        if item is None:
            return None
        stored_at, view = item
        if time.monotonic() - stored_at > self.ttl_s:
            del self._store[key]
            return None
        return view

    def set(self, key: Any, view: JevView) -> None:
        if len(self._store) >= self.maxsize and key not in self._store:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        self._store[key] = (time.monotonic(), view)


# --------------------------------------------------------------------------- real advisors


class JevAdvisor:
    """Synchronous `JevAdvisorProtocol` backed by `typesafe_sdk.TypeSafeClient`.

    Guarantees:
    - Never raises into the trading loop: SDK errors and unexpected exceptions are caught,
      logged, and turned into `None` (skip this cycle).
    - Enforces `latency_budget_ms`: if a call takes longer than that, the returned `JevView`
      has `late=True`; callers (and `Strategy` code) MUST treat `late=True` as a HOLD.
    - Caches answers for `cache_ttl_s` seconds, keyed by `(symbol, question_key, hash(state))`
      -- `question_key` (e.g. `"direction:5min"`, `"regime"`) is included so different
      question sets asked against the same state never collide.
    - Rate-limits outgoing calls with a token bucket; a call made with no tokens available is
      skipped (returns `None`) rather than blocking the trading loop.
    - Opens a circuit breaker after `circuit_max_errors` consecutive failures and skips calls
      (returns `None`) for `circuit_cooldown_s` before trying again (half-open).
    - Logs every real call to `decision_log`, if one is attached.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: Optional[str] = None,
        latency_budget_ms: int = _DEFAULT_LATENCY_BUDGET_MS,
        cache_ttl_s: float = _DEFAULT_CACHE_TTL_S,
        cache_maxsize: int = _DEFAULT_CACHE_MAXSIZE,
        rate_limit_per_sec: float = _DEFAULT_RATE_LIMIT_PER_SEC,
        rate_limit_burst: float = _DEFAULT_RATE_LIMIT_BURST,
        circuit_max_errors: int = _DEFAULT_CIRCUIT_MAX_ERRORS,
        circuit_cooldown_s: float = _DEFAULT_CIRCUIT_COOLDOWN_S,
        decision_log: Optional[DecisionLog] = None,
    ) -> None:
        self._client = client
        self._model = model
        self.latency_budget_ms = latency_budget_ms
        self._cache = _TTLCache(cache_ttl_s, cache_maxsize)
        self._bucket = _TokenBucket(rate_limit_per_sec, rate_limit_burst)
        self._breaker = _CircuitBreaker(circuit_max_errors, circuit_cooldown_s)
        self._decision_log = decision_log

    def direction(self, symbol: str, features: Mapping[str, Any], horizon: str) -> Optional[JevView]:
        cost_bps = float(features.get("cost_bps", 0.0)) if isinstance(features, Mapping) else 0.0
        questions = direction_questions(horizon, cost_bps)
        return self._ask_cached(symbol, f"direction:{horizon}", dict(features), questions)

    def regime(self, symbol: str, features: Mapping[str, Any]) -> Optional[JevView]:
        questions = regime_questions()
        return self._ask_cached(symbol, "regime", dict(features), questions)

    def ask(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        question_key = ",".join(sorted(questions.keys()))
        return self._ask_cached("_", question_key, dict(state), questions)

    def _ask_cached(self, symbol: str, question_key: str, state: dict[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        key = _cache_key(symbol, question_key, state)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        view = self._call(symbol, question_key, state, questions)
        if view is not None:
            self._cache.set(key, view)
        return view

    def _call(self, symbol: str, question_key: str, state: dict[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        if self._breaker.open:
            logger.warning("jev circuit breaker open; skipping %s/%s", symbol, question_key)
            return None
        if not self._bucket.try_acquire():
            logger.warning("jev rate limit exceeded; skipping %s/%s", symbol, question_key)
            return None

        t0 = time.perf_counter()
        try:
            resp = self._client.system_one(state=state, questions=questions, model=self._model)
        except TypeSafeError as exc:
            self._breaker.record_error()
            logger.error("jev call failed for %s/%s: %s", symbol, question_key, exc)
            return None
        except Exception:  # never raise into the trading loop
            self._breaker.record_error()
            logger.exception("jev call raised an unexpected error for %s/%s", symbol, question_key)
            return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._breaker.record_success()
        late = elapsed_ms > self.latency_budget_ms
        if late:
            logger.warning("jev call for %s/%s took %.0fms > budget %dms; marking late", symbol, question_key, elapsed_ms, self.latency_budget_ms)
        view = _view_from_response(resp, elapsed_ms, late=late)
        if self._decision_log is not None:
            try:
                self._decision_log.record(symbol=symbol, state=state, questions=questions, view=view, question_key=question_key)
            except Exception:
                logger.exception("failed to write jev decision log entry for %s/%s", symbol, question_key)
        return view

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class AsyncJevAdvisor:
    """Asynchronous counterpart of `JevAdvisor`, backed by `typesafe_sdk.AsyncTypeSafeClient`.

    Same cache/rate-limit/breaker/latency-budget/never-raise guarantees as `JevAdvisor`. The
    async client additionally lets the latency budget be enforced as an actual cutoff
    (`asyncio.wait_for`) rather than only a post-hoc flag, and supports "speculative fan-out"
    via `speculate` for hiding request latency behind other async work.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: Optional[str] = None,
        latency_budget_ms: int = _DEFAULT_LATENCY_BUDGET_MS,
        cache_ttl_s: float = _DEFAULT_CACHE_TTL_S,
        cache_maxsize: int = _DEFAULT_CACHE_MAXSIZE,
        rate_limit_per_sec: float = _DEFAULT_RATE_LIMIT_PER_SEC,
        rate_limit_burst: float = _DEFAULT_RATE_LIMIT_BURST,
        circuit_max_errors: int = _DEFAULT_CIRCUIT_MAX_ERRORS,
        circuit_cooldown_s: float = _DEFAULT_CIRCUIT_COOLDOWN_S,
        decision_log: Optional[DecisionLog] = None,
    ) -> None:
        self._client = client
        self._model = model
        self.latency_budget_ms = latency_budget_ms
        self._cache = _TTLCache(cache_ttl_s, cache_maxsize)
        self._bucket = _TokenBucket(rate_limit_per_sec, rate_limit_burst)
        self._breaker = _CircuitBreaker(circuit_max_errors, circuit_cooldown_s)
        self._decision_log = decision_log

    async def direction(self, symbol: str, features: Mapping[str, Any], horizon: str) -> Optional[JevView]:
        cost_bps = float(features.get("cost_bps", 0.0)) if isinstance(features, Mapping) else 0.0
        questions = direction_questions(horizon, cost_bps)
        return await self._ask_cached(symbol, f"direction:{horizon}", dict(features), questions)

    async def regime(self, symbol: str, features: Mapping[str, Any]) -> Optional[JevView]:
        questions = regime_questions()
        return await self._ask_cached(symbol, "regime", dict(features), questions)

    async def ask(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        question_key = ",".join(sorted(questions.keys()))
        return await self._ask_cached("_", question_key, dict(state), questions)

    async def _ask_cached(self, symbol: str, question_key: str, state: dict[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        key = _cache_key(symbol, question_key, state)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        view = await self._call(symbol, question_key, state, questions)
        if view is not None:
            self._cache.set(key, view)
        return view

    async def _call(self, symbol: str, question_key: str, state: dict[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        if self._breaker.open:
            logger.warning("jev circuit breaker open; skipping %s/%s", symbol, question_key)
            return None
        if not self._bucket.try_acquire():
            logger.warning("jev rate limit exceeded; skipping %s/%s", symbol, question_key)
            return None

        t0 = time.perf_counter()
        budget_s = self.latency_budget_ms / 1000.0
        try:
            resp = await asyncio.wait_for(
                self._client.system_one(state=state, questions=questions, model=self._model), timeout=budget_s
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # A latency-budget breach is not treated as an error for circuit-breaker purposes:
            # the request may well still be in flight server-side, it's just too slow to use.
            logger.warning("jev call for %s/%s exceeded latency budget (%dms); marking late", symbol, question_key, self.latency_budget_ms)
            return JevView(probs={}, top={}, confidence={}, latency_ms=elapsed_ms, late=True, model=self._model or "")
        except TypeSafeError as exc:
            self._breaker.record_error()
            logger.error("jev call failed for %s/%s: %s", symbol, question_key, exc)
            return None
        except Exception:  # never raise into the trading loop
            self._breaker.record_error()
            logger.exception("jev call raised an unexpected error for %s/%s", symbol, question_key)
            return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._breaker.record_success()
        late = elapsed_ms > self.latency_budget_ms
        if late:
            logger.warning("jev call for %s/%s took %.0fms > budget %dms; marking late", symbol, question_key, elapsed_ms, self.latency_budget_ms)
        view = _view_from_response(resp, elapsed_ms, late=late)
        if self._decision_log is not None:
            try:
                self._decision_log.record(symbol=symbol, state=state, questions=questions, view=view, question_key=question_key)
            except Exception:
                logger.exception("failed to write jev decision log entry for %s/%s", symbol, question_key)
        return view

    def speculate(self, symbol: str, features: Mapping[str, Any], horizon: str) -> "SpeculativeDirection":
        """Fire a `direction()` request now for a not-yet-confirmed next state, so its latency
        is hidden behind other async work (order management, data fetches, ...). Call
        `.get(max_age_ms)` when the state is confirmed and actually needed; it returns the
        answer only if it is still fresh enough, else `None` (fall back to a fresh call)."""
        spec = SpeculativeDirection(self)
        spec.fire(symbol, features, horizon)
        return spec

    async def aclose(self) -> None:
        aclose = getattr(self._client, "aclose", None)
        if callable(aclose):
            await aclose()


class SpeculativeDirection:
    """Holds one in-flight speculative `AsyncJevAdvisor.direction()` call.

    NOT part of `JevAdvisorProtocol` -- it's an optimization helper for async live/paper
    runners. Typical use: as soon as the current bar closes and a plausible "next state" can be
    estimated (e.g. carrying forward the last price), `fire()` the request; once the real next
    bar arrives, call `get(max_age_ms)` and use the answer if it's still fresh, else fall back
    to `advisor.direction(...)` for a synchronous-feeling call.
    """

    def __init__(self, advisor: AsyncJevAdvisor) -> None:
        self._advisor = advisor
        self._task: Optional["asyncio.Task[Optional[JevView]]"] = None
        self._fired_at: float = 0.0

    def fire(self, symbol: str, features: Mapping[str, Any], horizon: str) -> None:
        self._task = asyncio.ensure_future(self._advisor.direction(symbol, features, horizon))
        self._fired_at = time.monotonic()

    async def get(self, max_age_ms: float) -> Optional[JevView]:
        task, self._task = self._task, None
        if task is None:
            return None
        age_ms = (time.monotonic() - self._fired_at) * 1000.0
        if age_ms > max_age_ms and not task.done():
            task.cancel()
            try:
                await task  # let cancellation actually propagate so nothing is left dangling
            except (asyncio.CancelledError, Exception):
                pass
            return None
        try:
            view = await task
        except asyncio.CancelledError:
            return None
        except Exception:
            logger.exception("speculative jev call failed")
            return None
        if (time.monotonic() - self._fired_at) * 1000.0 > max_age_ms:
            return None
        return view


# --------------------------------------------------------------------------- offline heuristic


@dataclass(frozen=True)
class OfflineWeights:
    """Configurable weights for `OfflineJevAdvisor`'s logistic heuristic."""

    momentum: float = 1.0
    mean_reversion: float = -0.6
    imbalance: float = 0.5
    trend: float = 0.8
    bias: float = 0.0


class OfflineJevAdvisor:
    """Deterministic, offline stand-in for Jev. **THIS IS NOT JEV.**

    It answers with a transparent logistic heuristic over a handful of features (momentum,
    a mean-reversion z-score, order-book imbalance, trend slope), with configurable weights,
    so that strategies written against `JevAdvisorProtocol` can be built and backtested
    without a network call or an API key. It has no real forecasting skill -- it exists so the
    rest of the system exercises the exact same code path in backtest as in paper/live, and so
    `calibration.py` has a documented baseline to compare a real Jev decision log against.
    Never point this at real capital.
    """

    def __init__(self, weights: Optional[OfflineWeights] = None, *, model: str = "offline-heuristic-v1") -> None:
        self.weights = weights or OfflineWeights()
        self.model = model

    def direction(self, symbol: str, features: Mapping[str, Any], horizon: str) -> Optional[JevView]:
        score = self._score(features)
        p_up, p_down, p_flat = self._three_way(score, features)
        probs = {"direction": {"up": p_up, "down": p_down, "flat": p_flat}}
        top = {"direction": max(probs["direction"], key=probs["direction"].get)}
        confidence = {"direction": max(probs["direction"].values())}
        return JevView(probs=probs, top=top, confidence=confidence, latency_ms=0.0, late=False, model=self.model, raw={"score": round(score, 4)})

    def regime(self, symbol: str, features: Mapping[str, Any]) -> Optional[JevView]:
        vol = float(features.get("vol_bps", 0.0) or 0.0)
        trend = float(features.get("trend_slope_bps", 0.0) or 0.0)
        z = float(features.get("z_vwap", 0.0) or 0.0)
        w = self.weights
        logits = {
            "trending_up": w.trend * trend + w.momentum * max(0.0, trend),
            "trending_down": -w.trend * trend + w.momentum * max(0.0, -trend),
            "mean_reverting": abs(z) * 1.5 - vol / 50.0,
            "volatile_chop": vol / 30.0 - abs(trend) / 5.0,
            "quiet": -vol / 20.0,
        }
        names = list(logits.keys())
        raw_probs = _softmax([logits[n] for n in names])
        regime_probs = {n: round(p, 4) for n, p in zip(names, raw_probs)}
        risk_off_p = round(_sigmoid(vol / 40.0 - 1.0), 4)
        probs = {"regime": regime_probs, "risk_off": {"yes": risk_off_p, "no": round(1.0 - risk_off_p, 4)}}
        top = {"regime": max(regime_probs, key=regime_probs.get), "risk_off": "yes" if risk_off_p >= 0.5 else "no"}
        confidence = {"regime": max(regime_probs.values()), "risk_off": round(abs(risk_off_p - 0.5) * 2, 4)}
        return JevView(probs=probs, top=top, confidence=confidence, latency_ms=0.0, late=False, model=self.model, raw={"vol_bps": vol, "trend_slope_bps": trend, "z_vwap": z})

    def ask(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        """Generic fallback for arbitrary question sets: routes each question by its `type`
        through the same score, so the offline advisor can stand in for `ask()` calls too.
        Coarser than `direction`/`regime` -- prefer those where possible."""
        score = self._score(state)
        probs: dict[str, dict[str, float]] = {}
        top: dict[str, str] = {}
        confidence: dict[str, float] = {}
        raw: dict[str, Any] = {"score": round(score, 4)}
        for name, question in questions.items():
            qtype = _question_type(question)
            if qtype == "noul":
                p_yes = round(_sigmoid(score), 4)
                probs[name] = {"yes": p_yes, "no": round(1.0 - p_yes, 4)}
                top[name] = "yes" if p_yes >= 0.5 else "no"
                confidence[name] = round(abs(p_yes - 0.5) * 2, 4)
            elif qtype == "choice":
                labels = list(_qfield(question, "criteria", {}).keys())
                if not labels:
                    continue
                # bias the softmax toward the first label as "score increases" (deterministic, arbitrary but stable ordering).
                logits = [score * (1.0 - 2.0 * i / max(1, len(labels) - 1)) for i in range(len(labels))]
                dist = _softmax(logits)
                probs[name] = {label: round(p, 4) for label, p in zip(labels, dist)}
                top[name] = max(probs[name], key=probs[name].get)
                confidence[name] = round(max(probs[name].values()), 4)
            elif qtype == "score":
                criteria = list(_qfield(question, "criteria", []))
                n = len(criteria)
                if n == 0:
                    continue
                center = _sigmoid(score) * (n - 1)
                logits = [-((i - center) ** 2) for i in range(n)]
                dist = _softmax(logits)
                probs[name] = {str(i): round(p, 4) for i, p in enumerate(dist)}
                expected = sum(i * p for i, p in enumerate(dist))
                top[name] = str(round(expected))
                confidence[name] = round(max(dist), 4)
                raw[name] = {"expected": round(expected, 4)}
        return JevView(probs=probs, top=top, confidence=confidence, latency_ms=0.0, late=False, model=self.model, raw=raw)

    def _score(self, features: Mapping[str, Any]) -> float:
        ret_bps = features.get("ret_bps") if isinstance(features, Mapping) else None
        momentum = 0.0
        if isinstance(ret_bps, Mapping) and ret_bps:
            values = [v for v in ret_bps.values() if v is not None]
            if values:
                momentum = float(sum(values) / len(values)) / 50.0  # normalize bps into a logit-scale term
        mean_rev = float(features.get("z_vwap", 0.0) or 0.0)
        imbalance = float(features.get("book_imbalance", 0.0) or 0.0)
        trend = float(features.get("trend_slope_bps", 0.0) or 0.0) / 20.0
        w = self.weights
        return w.bias + w.momentum * momentum + w.mean_reversion * mean_rev + w.imbalance * imbalance + w.trend * trend

    def _three_way(self, score: float, features: Mapping[str, Any]) -> tuple[float, float, float]:
        cost_bps = float(features.get("cost_bps", 0.0) or 0.0)
        vol_bps = float(features.get("vol_bps", 0.0) or 0.0)
        p_move_up = _sigmoid(score)
        edge = abs(score)
        p_flat = min(max(_sigmoid(cost_bps / max(vol_bps, 1.0) - edge), 0.05), 0.9)
        remaining = 1.0 - p_flat
        p_up = remaining * p_move_up
        p_down = remaining * (1.0 - p_move_up)
        return round(p_up, 4), round(p_down, 4), round(p_flat, 4)


# --------------------------------------------------------------------------- replay advisor


def _view_from_entry(entry: Mapping[str, Any]) -> JevView:
    answers = entry.get("answers", {})
    return JevView(
        probs=answers.get("probs", {}),
        top=answers.get("top", {}),
        confidence=answers.get("confidence", {}),
        latency_ms=float(entry.get("latency_ms", 0.0) or 0.0),
        late=bool(entry.get("late", False)),
        model=str(entry.get("model", "")),
        raw={"replayed_ts": entry.get("ts"), "extra": entry.get("extra", {})},
    )


class ReplayJevAdvisor:
    """Replays previously recorded Jev decisions from a `DecisionLog` (JSONL) so a backtest can
    honestly evaluate what Jev actually said during paper trading, instead of relying on
    hindsight or `OfflineJevAdvisor`'s heuristic.

    Entries are served back in recorded order per `(symbol, question_key)` FIFO queue -- the
    backtest driving this advisor must replay events in the same order the log was recorded
    in. Once a queue for a given `(symbol, question_key)` is exhausted, further calls for it
    return `None` (same as "no signal this cycle").
    """

    def __init__(self, entries: Iterable[Mapping[str, Any]]) -> None:
        self._queues: dict[tuple[str, str], Deque[Mapping[str, Any]]] = defaultdict(deque)
        for entry in entries:
            key = (entry.get("symbol", ""), entry.get("question_key", ""))
            self._queues[key].append(entry)

    @classmethod
    def from_log(cls, path: str | Path) -> "ReplayJevAdvisor":
        return cls(read_jsonl(path))

    def _next(self, symbol: str, question_key: str) -> Optional[JevView]:
        queue = self._queues.get((symbol, question_key))
        if not queue:
            return None
        return _view_from_entry(queue.popleft())

    def direction(self, symbol: str, features: Mapping[str, Any], horizon: str) -> Optional[JevView]:
        return self._next(symbol, f"direction:{horizon}")

    def regime(self, symbol: str, features: Mapping[str, Any]) -> Optional[JevView]:
        return self._next(symbol, "regime")

    def ask(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        question_key = ",".join(sorted(questions.keys()))
        return self._next("_", question_key)


# --------------------------------------------------------------------------- shadow advisor


class ShadowAdvisor:
    """Answers with `primary` while firing the identical question at each backend in `shadows`
    off to the side, purely for comparison logging -- trading decisions never depend on a
    shadow's answer or latency.

    In paper trading this collects Jev vs Kev vs Laya (or any other `JevAdvisorProtocol`,
    including `OfflineJevAdvisor`/`ForecastAdvisor`-style in-process advisors) answers on
    IDENTICAL states for free: `jevtrader.jev.finetune.scoreboard` can later score every
    backend's decision log against the same realized outcomes with no restated state.

    Guarantees:
    - `direction`/`regime`/`ask` always return `primary`'s result immediately -- a shadow is
      never on the critical path and never blocks the caller (each fires on a background
      thread).
    - A shadow that raises, hangs, or answers `None` is caught and logged; it never surfaces to
      the caller and never affects `primary`'s answer.
    - Every shadow answer that DOES come back is written to `decision_log` (if attached) tagged
      `extra={"backend": name, "shadow": True}`, under the same `(symbol, question_key)` the
      primary would use, so its rows line up with the primary's for calibration/scoreboard
      analysis.
    - Call `wait(timeout=None)` to block until in-flight shadow calls finish -- for tests and a
      clean shutdown; never required on the trading hot path.
    """

    def __init__(
        self,
        primary: Any,
        shadows: Mapping[str, Any],
        *,
        decision_log: Optional[DecisionLog] = None,
        max_workers: int = 4,
    ) -> None:
        self._primary = primary
        self._shadows = dict(shadows)
        self._decision_log = decision_log
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="jev-shadow")
        self._futures: list[concurrent.futures.Future] = []

    def direction(self, symbol: str, features: Mapping[str, Any], horizon: str) -> Optional[JevView]:
        result = self._primary.direction(symbol, features, horizon)
        self._fan_out("direction", (symbol, features, horizon), symbol=symbol, question_key=f"direction:{horizon}", state=dict(features))
        return result

    def regime(self, symbol: str, features: Mapping[str, Any]) -> Optional[JevView]:
        result = self._primary.regime(symbol, features)
        self._fan_out("regime", (symbol, features), symbol=symbol, question_key="regime", state=dict(features))
        return result

    def ask(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        result = self._primary.ask(state, questions)
        question_key = ",".join(sorted(questions.keys()))
        self._fan_out("ask", (state, questions), symbol="_", question_key=question_key, state=dict(state))
        return result

    def _fan_out(self, method_name: str, args: tuple[Any, ...], *, symbol: str, question_key: str, state: dict[str, Any]) -> None:
        if not self._shadows:
            return
        self._futures = [f for f in self._futures if not f.done()]  # drop finished futures so this list can't grow unbounded
        for name, shadow in self._shadows.items():
            self._futures.append(self._executor.submit(self._run_shadow, name, shadow, method_name, args, symbol, question_key, state))

    def _run_shadow(self, name: str, shadow: Any, method_name: str, args: tuple[Any, ...], symbol: str, question_key: str, state: dict[str, Any]) -> None:
        try:
            view = getattr(shadow, method_name)(*args)
        except Exception:
            logger.exception("shadow backend %r failed on %s/%s", name, symbol, question_key)
            return
        if view is None or self._decision_log is None:
            return
        try:
            self._decision_log.record(symbol=symbol, state=state, questions={}, view=view, question_key=question_key, extra={"backend": name, "shadow": True})
        except Exception:
            logger.exception("failed to log shadow decision for backend %r on %s/%s", name, symbol, question_key)

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until every currently in-flight shadow call finishes (or `timeout` elapses)."""
        futures, self._futures = self._futures, []
        for f in futures:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    def close(self) -> None:
        self.wait()
        self._executor.shutdown(wait=False)


# --------------------------------------------------------------------------- factory


def make_advisor(
    settings: Any,
    mode: Any,
    *,
    decision_log: Optional[DecisionLog] = None,
    client: Optional[Any] = None,
    backend: Optional[Any] = None,
) -> Any:
    """Build the right `JevAdvisorProtocol` for `settings` + trading `mode` (+ decision `backend`).

    `mode` gates network use at all: anything other than `"paper"`/`"live"` (accepts either a
    `jevtrader.core.broker.TradingMode` or a plain string) always returns `OfflineJevAdvisor` --
    a backtest never depends on a remote API or a locally running server being up.

    In `paper`/`live`, `backend` picks the implementation (a `DecisionBackend` or one of its
    string values `"jev"`/`"kev"`/`"laya"`/`"offline"`). It defaults to `$DECISION_BACKEND`,
    and if that's unset too, to `"jev"` -- i.e. unchanged pre-existing behavior: a real
    `JevAdvisor` against hosted Jev when `settings.has_jev`, else `OfflineJevAdvisor`. Setting
    `DECISION_BACKEND=kev` or `DECISION_BACKEND=laya` (or passing `backend=` explicitly) instead
    points `JevAdvisor` at a locally served Kev/Laya checkpoint via `backends.make_client` --
    no API key required, and never a hard failure: an unresolvable backend still fails open to
    `OfflineJevAdvisor` rather than raising into the trading loop.
    """
    mode_value = getattr(mode, "value", mode)
    if mode_value not in ("paper", "live"):
        return OfflineJevAdvisor()

    resolved_backend = DecisionBackend.coerce(backend) if backend is not None else DecisionBackend.from_env(default=DecisionBackend.JEV)

    if resolved_backend is DecisionBackend.OFFLINE:
        return OfflineJevAdvisor()
    if resolved_backend is DecisionBackend.JEV and not getattr(settings, "has_jev", False):
        return OfflineJevAdvisor()

    c = client
    advisor_model = None
    if c is None:
        try:
            advisor_model = getattr(settings, "jev_model", None) if resolved_backend is DecisionBackend.JEV else None
            c = make_client(resolved_backend, settings=settings, timeout=max(1.0, settings.jev_latency_budget_ms / 1000.0))
        except Exception:
            logger.exception("failed to build a %s decision client; falling back to OfflineJevAdvisor", resolved_backend.value)
            return OfflineJevAdvisor()
    return JevAdvisor(c, model=advisor_model, latency_budget_ms=settings.jev_latency_budget_ms, decision_log=decision_log)
