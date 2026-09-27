"""Strategy registry: `@register` a Strategy subclass and it becomes available to the CLI by name."""

from __future__ import annotations

import importlib
import pkgutil

from jevtrader.core.strategy import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    name = cls.spec.name
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"duplicate strategy name {name!r}")
    _REGISTRY[name] = cls
    return cls


def _autoload() -> None:
    importlib.import_module("jevtrader.rebalance.bot")  # SmartRebalanceBot lives outside strategies/
    try:
        pkg = importlib.import_module("jevtrader.strategies")
    except ModuleNotFoundError:
        return
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)


def get(name: str) -> type[Strategy]:
    _autoload()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(_REGISTRY)}") from None


def all_strategies() -> dict[str, type[Strategy]]:
    _autoload()
    return dict(_REGISTRY)
