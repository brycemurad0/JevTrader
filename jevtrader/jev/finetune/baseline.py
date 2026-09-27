"""An honest quant baseline: multinomial logistic regression on `build_state`'s numeric
features, fit with plain numpy + scipy (no sklearn), answering the SAME `direction` question a
Kev/Laya/Jev checkpoint does. It costs microseconds per call and has no learned "understanding"
of anything -- it's a straight line through a dozen numbers.

This exists to keep everyone honest: a fine-tuned local model earns its keep only if it beats
BOTH the offline heuristic and this baseline, on Brier score AND after-cost trading edge, on an
untouched test split (`scoreboard.beats` encodes exactly that rule). If it can't beat a
regularized softmax over rounded technical features, the model isn't adding anything an LLM's
extra cost and latency would justify -- say so plainly rather than paper over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize

DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "ret_bps.5",
    "ret_bps.15",
    "ret_bps.30",
    "ret_bps.60",
    "vol_bps",
    "z_vwap",
    "rsi",
    "atr_pct",
    "vol_ratio",
    "trend_slope_bps",
    "spread_bps",
    "book_imbalance",
    "microprice_offset_bps",
    "unrealized_pnl_bps",
    "cost_bps",
)


def _get_path(state: Mapping[str, Any], path: str) -> Any:
    cur: Any = state
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def extract_features(state: Mapping[str, Any], feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES) -> np.ndarray:
    """A fixed-order numeric feature vector from a `build_state` dict. Dotted paths index into
    nested dicts (e.g. `"ret_bps.30"`); a missing, `None`, or non-numeric value becomes `0.0`
    (a neutral value for every feature this module uses -- all are roughly zero-centered)."""
    values = []
    for name in feature_names:
        v = _get_path(state, name)
        values.append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0)
    return np.asarray(values, dtype=float)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class LogisticBaseline:
    """A fitted multinomial logistic regression: `K` classes over standardized features plus an
    intercept, with an optional post-hoc temperature (`calibrate_temperature`). `weights` has
    shape `(K, len(feature_names) + 1)`, column 0 the intercept."""

    classes: tuple[str, ...]
    feature_names: tuple[str, ...]
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    temperature: float = 1.0
    label_key: str = "direction"
    model: str = "quant-baseline-logit"

    def _design(self, states: Sequence[Mapping[str, Any]]) -> np.ndarray:
        x = np.stack([extract_features(s, self.feature_names) for s in states]) if states else np.zeros((0, len(self.feature_names)))
        xs = (x - self.feature_mean) / self.feature_std
        return np.hstack([np.ones((xs.shape[0], 1)), xs])

    def predict_proba(self, states: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """`(N, K)` calibrated class probabilities, rows summing to 1."""
        logits = self._design(states) @ self.weights.T / max(self.temperature, 1e-6)
        return _softmax_rows(logits)

    def predict_one(self, state: Mapping[str, Any]) -> dict[str, float]:
        probs = self.predict_proba([state])[0]
        return {c: float(p) for c, p in zip(self.classes, probs)}


def fit_logistic_baseline(
    examples: Sequence[Mapping[str, Any]],
    *,
    label_key: str = "direction",
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    l2: float = 1.0,
    max_iter: int = 200,
) -> LogisticBaseline:
    """Fit on `examples` (from `dataset.build_examples`/`build_dataset`'s `train` split) via
    L-BFGS-B on the L2-regularized multinomial cross-entropy (intercept unregularized). Softmax
    is shift-invariant, so without regularization the `K`-way (rather than `K-1`-way) weight
    matrix would be underdetermined; `l2 > 0` breaks that degeneracy and is standard practice.
    """
    states = [ex["state"] for ex in examples]
    labels = [ex["labels"][label_key] for ex in examples]
    if not states:
        raise ValueError("fit_logistic_baseline requires at least one example")
    classes = tuple(sorted(set(labels)))
    class_index = {c: i for i, c in enumerate(classes)}
    y = np.array([class_index[label] for label in labels], dtype=int)

    x = np.stack([extract_features(s, feature_names) for s in states])
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    xs = (x - mean) / std
    design = np.hstack([np.ones((xs.shape[0], 1)), xs])

    n, d1 = design.shape
    k = len(classes)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0

    def loss_and_grad(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
        w = w_flat.reshape(k, d1)
        logits = design @ w.T
        probs = _softmax_rows(logits)
        nll = -np.log(np.clip(probs[np.arange(n), y], 1e-12, None)).mean()
        reg = 0.5 * l2 * np.sum(w[:, 1:] ** 2) / n
        grad = (probs - onehot).T @ design / n
        grad[:, 1:] += l2 * w[:, 1:] / n
        return nll + reg, grad.ravel()

    result = optimize.minimize(loss_and_grad, np.zeros(k * d1), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    weights = result.x.reshape(k, d1)
    return LogisticBaseline(classes=classes, feature_names=tuple(feature_names), weights=weights, feature_mean=mean, feature_std=std, label_key=label_key)


def calibrate_temperature(baseline: LogisticBaseline, examples: Sequence[Mapping[str, Any]], *, bounds: tuple[float, float] = (0.05, 20.0)) -> LogisticBaseline:
    """Fit a scalar temperature (logits /= T before softmax) minimizing cross-entropy on
    `examples` (the CALIBRATION split -- never train, never test), via bounded 1-D search.
    Returns a new `LogisticBaseline`; `examples` with an unseen label are ignored."""
    class_index = {c: i for i, c in enumerate(baseline.classes)}
    states, y_list = [], []
    for ex in examples:
        label = ex["labels"].get(baseline.label_key)
        if label in class_index:
            states.append(ex["state"])
            y_list.append(class_index[label])
    if not states:
        return baseline
    y = np.asarray(y_list, dtype=int)
    raw_logits = baseline._design(states) @ baseline.weights.T  # temperature-1 logits

    def nll(t: float) -> float:
        t = max(float(t), 1e-3)
        probs = _softmax_rows(raw_logits / t)
        return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())

    result = optimize.minimize_scalar(nll, bounds=bounds, method="bounded")
    return replace(baseline, temperature=float(result.x))


def save_baseline(baseline: LogisticBaseline, path: str | Path) -> None:
    """Persist a fitted (and, usually, temperature-calibrated) baseline as JSON."""
    payload = {
        "classes": list(baseline.classes),
        "feature_names": list(baseline.feature_names),
        "weights": baseline.weights.tolist(),
        "feature_mean": baseline.feature_mean.tolist(),
        "feature_std": baseline.feature_std.tolist(),
        "temperature": baseline.temperature,
        "label_key": baseline.label_key,
        "model": baseline.model,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def load_baseline(path: str | Path) -> LogisticBaseline:
    """Load a baseline saved by `save_baseline`."""
    payload = json.loads(Path(path).read_text())
    return LogisticBaseline(
        classes=tuple(payload["classes"]),
        feature_names=tuple(payload["feature_names"]),
        weights=np.asarray(payload["weights"], dtype=float),
        feature_mean=np.asarray(payload["feature_mean"], dtype=float),
        feature_std=np.asarray(payload["feature_std"], dtype=float),
        temperature=float(payload["temperature"]),
        label_key=payload.get("label_key", "direction"),
        model=payload.get("model", "quant-baseline-logit"),
    )
