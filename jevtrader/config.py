"""Runtime settings loaded from environment / .env. Keys never leave this machine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from jevtrader.core.broker import TradingMode

ROOT = Path(__file__).resolve().parent.parent
LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_THIS_IS_REAL_MONEY"


def _bool(v: str | None, default: bool) -> bool:
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    alpaca_key: str
    alpaca_secret: str
    alpaca_paper: bool
    alpaca_live_key: str
    alpaca_live_secret: str
    alpaca_data_feed: str
    typesafe_api_key: str
    jev_model: str
    jev_latency_budget_ms: int
    live_confirm: str
    data_dir: Path
    runs_dir: Path
    state_dir: Path

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_key and self.alpaca_secret)

    @property
    def has_jev(self) -> bool:
        return bool(self.typesafe_api_key)

    def live_allowed(self) -> bool:
        return self.live_confirm == LIVE_CONFIRM_PHRASE

    def keys_for(self, mode: TradingMode) -> tuple[str, str]:
        if mode is TradingMode.LIVE:
            return (self.alpaca_live_key or self.alpaca_key, self.alpaca_live_secret or self.alpaca_secret)
        return self.alpaca_key, self.alpaca_secret


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env", override=False)
    e = os.environ.get
    return Settings(
        alpaca_key=e("ALPACA_API_KEY", ""),
        alpaca_secret=e("ALPACA_SECRET_KEY", ""),
        alpaca_paper=_bool(e("ALPACA_PAPER"), True),
        alpaca_live_key=e("ALPACA_LIVE_API_KEY", ""),
        alpaca_live_secret=e("ALPACA_LIVE_SECRET_KEY", ""),
        alpaca_data_feed=e("ALPACA_DATA_FEED", "iex"),
        typesafe_api_key=e("TYPESAFE_API_KEY", ""),
        jev_model=e("TYPESAFE_DEFAULT_MODEL", "jev-latest"),
        jev_latency_budget_ms=int(e("JEV_LATENCY_BUDGET_MS", "400")),
        live_confirm=e("JEV_LIVE_CONFIRM", ""),
        data_dir=Path(e("JEV_DATA_DIR", str(ROOT / "data" / "cache"))),
        runs_dir=Path(e("JEV_RUNS_DIR", str(ROOT / "runs"))),
        state_dir=Path(e("JEV_STATE_DIR", str(ROOT / "state"))),
    )
