"""Decision backend selection: hosted Jev, or a locally served Kev/Laya checkpoint.

Kev (github.com/jaredpalmer/kev) and Laya (github.com/NandhaKishorM/laya) both serve the exact
same `/v1/systemone` wire protocol TypeSafe's hosted Jev does, so `typesafe_sdk.TypeSafeClient`
talks to any of them unmodified -- only `base_url` (and, for a server with no auth configured,
a non-empty placeholder API key, since the SDK rejects an empty one) differ. That means every
other file in this package (`advisor.py`, `questions.py`, `state.py`, ...) is backend-agnostic:
point the same `JevAdvisor` at a different client and it's now running on-prem, offline of the
real Jev API, with no code change.

Local server defaults (override via the env vars below, or start them yourself):
    kev:  `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009`
    laya: `LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve` (defaults to 0.0.0.0:8000)
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional

DEFAULT_KEV_BASE_URL = "http://127.0.0.1:8009"
DEFAULT_LAYA_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_KEV_MODEL = "kev-latest"
DEFAULT_LAYA_MODEL = "laya-latest"

# The SDK requires a non-empty, printable-ASCII, whitespace-free API key even when talking to a
# local server that was started with no auth configured (KEV_API_KEY / LAYA_API_KEY unset). This
# is never sent anywhere but the local process on the other end of KEV_BASE_URL/LAYA_BASE_URL.
_LOCAL_PLACEHOLDER_API_KEY = "local-no-auth-required"

DECISION_BACKEND_ENV = "DECISION_BACKEND"
KEV_BASE_URL_ENV = "KEV_BASE_URL"
LAYA_BASE_URL_ENV = "LAYA_BASE_URL"
LOCAL_MODEL_API_KEY_ENV = "LOCAL_MODEL_API_KEY"


class DecisionBackend(str, Enum):
    """Which System One implementation answers Jev-shaped questions.

    `OFFLINE` has no HTTP endpoint at all -- it means "use `OfflineJevAdvisor` in-process",
    never a client. The other three are interchangeable `/v1/systemone` servers.
    """

    JEV = "jev"
    KEV = "kev"
    LAYA = "laya"
    OFFLINE = "offline"

    @classmethod
    def from_env(cls, default: "DecisionBackend | str" = "offline", *, env_var: str = DECISION_BACKEND_ENV) -> "DecisionBackend":
        """Read `env_var` (default `DECISION_BACKEND`); fall back to `default` when unset/blank."""
        raw = os.environ.get(env_var, "").strip().lower()
        value = raw or (default.value if isinstance(default, cls) else str(default).strip().lower())
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(b.value for b in cls)
            raise ValueError(f"Unknown decision backend {value!r} (from ${env_var}); expected one of: {valid}") from None

    @classmethod
    def coerce(cls, value: "DecisionBackend | str") -> "DecisionBackend":
        return value if isinstance(value, cls) else cls(str(value).strip().lower())


def base_url_for(backend: "DecisionBackend | str") -> Optional[str]:
    """The `/v1/systemone` base URL for a backend, or `None` for `JEV` (the SDK's own default,
    honoring `TYPESAFE_BASE_URL` if set). Raises for `OFFLINE`, which has no HTTP endpoint."""
    backend = DecisionBackend.coerce(backend)
    if backend is DecisionBackend.JEV:
        return None
    if backend is DecisionBackend.KEV:
        return os.environ.get(KEV_BASE_URL_ENV, "").strip() or DEFAULT_KEV_BASE_URL
    if backend is DecisionBackend.LAYA:
        return os.environ.get(LAYA_BASE_URL_ENV, "").strip() or DEFAULT_LAYA_BASE_URL
    raise ValueError("DecisionBackend.OFFLINE has no HTTP endpoint; use OfflineJevAdvisor directly")


def _api_key_for(backend: "DecisionBackend | str", *, settings: Any = None, api_key: Optional[str] = None) -> str:
    if api_key:
        return api_key
    backend = DecisionBackend.coerce(backend)
    if backend is DecisionBackend.JEV:
        settings_key = getattr(settings, "typesafe_api_key", "") if settings is not None else ""
        return settings_key or os.environ.get("TYPESAFE_API_KEY", "")
    # Kev/Laya: a locally served checkpoint usually has no auth configured at all.
    return os.environ.get(LOCAL_MODEL_API_KEY_ENV, "").strip() or _LOCAL_PLACEHOLDER_API_KEY


def _default_model_for(backend: "DecisionBackend | str", *, settings: Any = None) -> Optional[str]:
    backend = DecisionBackend.coerce(backend)
    if backend is DecisionBackend.JEV:
        return getattr(settings, "jev_model", None) if settings is not None else None
    if backend is DecisionBackend.KEV:
        return DEFAULT_KEV_MODEL
    if backend is DecisionBackend.LAYA:
        return DEFAULT_LAYA_MODEL
    return None


def make_client(
    backend: "DecisionBackend | str",
    *,
    settings: Any = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    is_async: bool = False,
    **client_kwargs: Any,
) -> Any:
    """Build a `TypeSafeClient` (or `AsyncTypeSafeClient` if `is_async`) pointed at `backend`.

    - `jev`: the SDK's own default base URL (hosted Jev), keyed by `settings.typesafe_api_key`
      or `$TYPESAFE_API_KEY`.
    - `kev` / `laya`: `$KEV_BASE_URL` / `$LAYA_BASE_URL` (defaulting to the servers' own default
      ports, 8009 / 8000), keyed by `$LOCAL_MODEL_API_KEY` or a harmless local placeholder.
    - `offline` is not valid here -- raises `ValueError` (use `OfflineJevAdvisor` in-process).

    `base_url`, if given, overrides the env-derived default for `kev`/`laya` (or the SDK
    default for `jev`) -- e.g. `scripts/finetune/scoreboard.py --kev http://gpu-box:8009`
    pointing at a specific server for one run without touching `$KEV_BASE_URL`.
    """
    backend = DecisionBackend.coerce(backend)
    if backend is DecisionBackend.OFFLINE:
        raise ValueError("DecisionBackend.OFFLINE has no HTTP client; use OfflineJevAdvisor directly")

    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

    kwargs: dict[str, Any] = {
        "api_key": _api_key_for(backend, settings=settings, api_key=api_key),
        "model": model or _default_model_for(backend, settings=settings),
    }
    resolved_base_url = base_url if base_url is not None else base_url_for(backend)
    if resolved_base_url is not None:
        kwargs["base_url"] = resolved_base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    kwargs.update(client_kwargs)
    cls = AsyncTypeSafeClient if is_async else TypeSafeClient
    return cls(**kwargs)
