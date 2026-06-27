"""Safe formula evaluator using a restricted pandas namespace."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_formula(
    formula: str,
    cross_section: dict[str, pd.Series],
) -> pd.Series:
    """Evaluate a formula string against a dict of feature Series (one per stock).

    Returns a signal Series indexed by TICKER.
    Raises ValueError if the formula references unknown identifiers.
    Raises NotImplementedError if time-series operators (delta, ts_mean, ts_std) are used.
    """
    namespace = _build_namespace(cross_section)
    try:
        result = eval(formula, {"__builtins__": {}}, namespace)  # noqa: S307
    except NameError as exc:
        raise ValueError(
            f"Formula references unknown identifier: {exc}. "
            "Check that all features are declared in the alpha config."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Formula evaluation failed: {exc}") from exc

    if not isinstance(result, pd.Series):
        raise ValueError(
            f"Formula must evaluate to a Series of per-stock values, got {type(result)}"
        )
    return result.dropna()


def _build_namespace(cross_section: dict[str, pd.Series]) -> dict:
    """Build the evaluation namespace: feature Series + allowed operator functions."""

    def _ts_error(*args, **kwargs):
        raise NotImplementedError(
            "delta(), ts_mean(), ts_std() require time-series context and cannot be used "
            "directly in the formula string. Pre-compute these as named features instead."
        )

    ns: dict = {
        # Operators
        "rank": lambda s: s.rank(pct=True),
        "zscore": lambda s: (s - s.mean()) / (s.std() + 1e-9),
        "log": lambda s: np.log(s.clip(lower=1e-9)),
        "abs": lambda s: s.abs(),
        "sign": lambda s: np.sign(s),
        # Time-series operators — not supported in cross-sectional formula
        "delta": _ts_error,
        "ts_mean": _ts_error,
        "ts_std": _ts_error,
    }
    ns.update(cross_section)
    return ns
