from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pytest
from typesafe_sdk import SystemOneResponse

from jevtrader.config import Settings


def make_response(answers: Mapping[str, Mapping[str, Any]], *, model: str = "jev-test", input_tokens: int = 100, output_tokens: int = 10) -> SystemOneResponse:
    """Build a real `typesafe_sdk.SystemOneResponse` from a plain answers dict, going through
    JSON (as the wire format does) so score-question integer keys coerce correctly."""
    payload = {"model": model, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}, "answers": dict(answers)}
    return SystemOneResponse.model_validate_json(json.dumps(payload))


def choice_answer(choice: str, probabilities: Mapping[str, float], confidence: float = 0.8) -> dict[str, Any]:
    return {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": dict(probabilities)}


def score_answer(score: float, probabilities: Mapping[int, float], legend: Optional[Mapping[int, str]] = None, confidence: float = 0.7) -> dict[str, Any]:
    legend = legend or {i: str(i) for i in probabilities}
    return {
        "type": "score",
        "score": score,
        "confidence": confidence,
        "legend": {str(k): v for k, v in legend.items()},
        "probabilities": {str(k): v for k, v in probabilities.items()},
    }


def noul_answer(noul: float) -> dict[str, Any]:
    return {"type": "noul", "noul": noul}


class FakeClient:
    """Duck-typed stand-in for `typesafe_sdk.TypeSafeClient`.

    Pass `responses` for a list of successful answers, `error` for a client that always
    raises, or `sequence` (a list mixing `SystemOneResponse`s and exception instances) for
    per-call control, e.g. to test a circuit breaker recovering after a run of failures. The
    last item repeats once the sequence is exhausted.
    """

    def __init__(
        self,
        responses: Optional[list[SystemOneResponse]] = None,
        *,
        sleep_s: float = 0.0,
        error: Optional[BaseException] = None,
        sequence: Optional[list[Any]] = None,
    ):
        if sequence is not None:
            self.sequence: list[Any] = list(sequence)
        elif error is not None:
            self.sequence = [error]
        else:
            self.sequence = list(responses) if responses is not None else [make_response({})]
        self.sleep_s = sleep_s
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def system_one(self, *, state, questions, model=None, **kw):
        self.calls.append({"state": state, "questions": dict(questions), "model": model})
        if self.sleep_s:
            time.sleep(self.sleep_s)
        idx = len(self.calls) - 1
        item = self.sequence[idx] if idx < len(self.sequence) else self.sequence[-1]
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


class FakeAsyncClient:
    """Async counterpart of `FakeClient`; see its docstring for `sequence` semantics."""

    def __init__(
        self,
        responses: Optional[list[SystemOneResponse]] = None,
        *,
        sleep_s: float = 0.0,
        error: Optional[BaseException] = None,
        sequence: Optional[list[Any]] = None,
    ):
        if sequence is not None:
            self.sequence: list[Any] = list(sequence)
        elif error is not None:
            self.sequence = [error]
        else:
            self.sequence = list(responses) if responses is not None else [make_response({})]
        self.sleep_s = sleep_s
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def system_one(self, *, state, questions, model=None, **kw):
        import asyncio

        self.calls.append({"state": state, "questions": dict(questions), "model": model})
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        idx = len(self.calls) - 1
        item = self.sequence[idx] if idx < len(self.sequence) else self.sequence[-1]
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self):
        self.closed = True


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        alpaca_key="",
        alpaca_secret="",
        alpaca_paper=True,
        alpaca_live_key="",
        alpaca_live_secret="",
        alpaca_data_feed="iex",
        typesafe_api_key="",
        jev_model="jev-latest",
        jev_latency_budget_ms=400,
        live_confirm="",
        data_dir=Path("/tmp/jevtrader-test/data"),
        runs_dir=Path("/tmp/jevtrader-test/runs"),
        state_dir=Path("/tmp/jevtrader-test/state"),
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def tmp_log_path(tmp_path: Path) -> Path:
    return tmp_path / "decisions.jsonl"
