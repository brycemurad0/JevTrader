"""Question library: typed `typesafe_sdk` questions for the shapes of judgment JevTrader needs.

Each builder returns a `dict[str, Question]` ready to pass straight to
`TypeSafeClient.system_one(state=..., questions=...)` or to `JevAdvisorProtocol.ask`. Bundling
several atomic questions into one call follows the community best practice ("multiple atomic
questions in one call") and keeps latency down to a single round trip.

Label contract (must match `jevtrader.core.interfaces.JevAdvisorProtocol`'s docstring):

- `direction_questions` -> `"direction"` Choice with labels `{"up", "down", "flat"}`.
- `regime_questions` -> `"regime"` Choice with labels
  `{"trending_up", "trending_down", "mean_reverting", "volatile_chop", "quiet"}`, plus a
  `"risk_off"` Noul.
"""

from __future__ import annotations

from typing import Sequence

from typesafe_sdk import Choice, Noul, Question, Score

DIRECTION_LABELS: tuple[str, ...] = ("up", "down", "flat")
REGIME_LABELS: tuple[str, ...] = ("trending_up", "trending_down", "mean_reverting", "volatile_chop", "quiet")
ENTRY_QUALITY_RUBRIC: tuple[str, ...] = (
    "poor: no discernible edge, or trading costs likely dominate any expected move",
    "weak: a slight edge exists but with high uncertainty or elevated cost/spread",
    "fair: a modest edge with acceptable cost and volatility",
    "good: a clear directional or mean-reversion edge with manageable risk",
    "excellent: a strong, well-supported edge with favorable cost and liquidity",
)
TILT_LABELS: tuple[str, ...] = ("strong underweight", "underweight", "neutral", "overweight", "strong overweight")


def direction_questions(horizon: str, cost_bps: float, *, include_conviction: bool = True) -> dict[str, Question]:
    """`"direction"` Choice (up/down/flat), where up/down explicitly mean "moves more than
    `cost_bps` in that direction within `horizon`" -- so "up"/"down" already net out the
    round-trip cost of trading on the signal, and Jev isn't asked to forecast noise it can't
    profitably act on. Optionally bundles a `"conviction"` Score (0-3) rating the strength of
    the signal, independent of its direction, as a second atomic question in the same call.
    """
    cost_txt = f"{float(cost_bps):.1f} bps"
    questions: dict[str, Question] = {
        "direction": Choice(
            instructions=(
                f"Given the market state, over the next {horizon} will this instrument's price move "
                f"up, down, or stay flat, net of a {cost_txt} round-trip trading cost? Only call "
                f"'up' or 'down' if the expected move is large enough to clear that cost."
            ),
            criteria={
                "up": f"Price moves up by more than {cost_txt} within {horizon}.",
                "down": f"Price moves down by more than {cost_txt} within {horizon}.",
                "flat": f"Price stays within +/-{cost_txt} of current within {horizon} (any move is too small to clear costs).",
            },
        )
    }
    if include_conviction:
        questions["conviction"] = Score(
            instructions=(
                "Independent of which direction, how strong is the directional signal in the "
                "state (momentum, trend, imbalance, regime all agreeing vs. contradicting)?"
            ),
            criteria=["none: signals are contradictory or absent", "weak: one weak signal", "moderate: multiple agreeing signals", "strong: multiple strong, agreeing signals"],
        )
    return questions


def regime_questions() -> dict[str, Question]:
    """`"regime"` Choice over the five market regimes JevTrader's strategies key off of, plus
    a `"risk_off"` Noul for "should exposure be cut across the board right now"."""
    return {
        "regime": Choice(
            instructions="Given the market state, which regime best describes current price action?",
            criteria={
                "trending_up": "A sustained directional move higher with reasonably consistent momentum.",
                "trending_down": "A sustained directional move lower with reasonably consistent momentum.",
                "mean_reverting": "Price oscillates around a stable level, snapping back after moves away from it.",
                "volatile_chop": "Erratic, high-volatility price action with no reliable direction or reversion.",
                "quiet": "Low volatility, low volume, no meaningful directional or mean-reverting behavior.",
            },
        ),
        "risk_off": Noul(
            instructions="Should exposure to this instrument be reduced across the board right now, independent of any specific signal (e.g. elevated volatility, thin liquidity, or a stressed regime)?",
            criteria={"true": "Conditions call for cutting risk (e.g. volatile chop with widening spreads).", "false": "Conditions are normal; no blanket risk reduction is warranted."},
        ),
    }


def entry_quality_questions() -> dict[str, Question]:
    """`"entry_quality"` Score, a 0-4 rubric rating whether *now* is a good time to open a new
    position at all (edge vs. cost/volatility/liquidity), separate from direction."""
    return {
        "entry_quality": Score(
            instructions=(
                "Rate the overall quality of opening a new position right now, given the trend, "
                "momentum, volatility, spread and cost context in the state."
            ),
            criteria=list(ENTRY_QUALITY_RUBRIC),
        )
    }


def rebalance_tilt_questions(assets: Sequence[str]) -> dict[str, Question]:
    """One `"tilt:<asset>"` Score (0-4, centered at neutral=2) per asset: how far its target
    portfolio weight should be tilted this rebalance, based on regime/momentum evidence in the
    shared state. Bundling one Score per asset keeps each question atomic while still getting
    a full cross-sectional read in a single `system_one` call."""
    if not assets:
        raise ValueError("rebalance_tilt_questions requires at least one asset")
    questions: dict[str, Question] = {}
    for asset in assets:
        questions[f"tilt:{asset}"] = Score(
            instructions=(
                f"Relative to its current target weight, how should {asset}'s weight be tilted "
                "over the coming rebalance period, based on the regime and momentum evidence in state?"
            ),
            criteria=list(TILT_LABELS),
        )
    return questions
