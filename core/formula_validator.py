"""Alpha formula validator, namespace builder, and cross-sectional evaluator."""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from core.types import AlphaConfig, ValidationResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All raw data columns available in the processed panel (output of data_process.py).
# Prices are joined back as-is; fundamentals are winsorised + standardised + ffilled.
AVAILABLE_RAW_COLUMNS: frozenset[str] = frozenset({
    # Price (raw, not standardised)
    "ADJUSTED_PRICE",
    "ADJUSTED_VOLUME",
    # Income statement — TTM, cross-sectionally winsorised + standardised
    "SALES_LTM",
    "COGS_LTM",
    "NET_INCOME_LTM",
    "OPER_INCOME_LTM",
    # Cash flow — TTM, cross-sectionally winsorised + standardised
    "DA_LTM",
    "SHARES_DILUTED",
})

# Function names the LLM may use in a raw_formula.
# delta/ts_mean/ts_std are listed so the validator recognises them and warns;
# they raise NotImplementedError at evaluation time.
ALLOWED_FUNCTION_NAMES: set[str] = {
    "rank", "zscore", "log", "abs", "sign", "delta", "ts_mean", "ts_std",
}

ALLOWED_UNIVERSES: set[str] = {"sp500"}

REQUIRED_KEYS: list[str] = [
    "alpha_id", "raw_formula", "universe", "start_date", "end_date",
]

_DATE_FMT = "%Y-%m-%d"
_NON_DATA_COLS = {"SECTOR", "COUNTRY", "INDUSTRY"}

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_alpha(alpha: AlphaConfig) -> ValidationResult:
    """Validate an alpha config. Returns a ValidationResult with errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Required keys
    for key in REQUIRED_KEYS:
        if key not in alpha:
            errors.append(f"Missing required field: '{key}'")

    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # 2. Raw formula validation
    raw_formula = alpha.get("raw_formula", "")
    _validate_raw_formula_tokens(raw_formula, errors, warnings)

    # 3. Universe
    if alpha.get("universe") not in ALLOWED_UNIVERSES:
        errors.append(
            f"Universe '{alpha.get('universe')}' not supported. "
            f"Choose from: {ALLOWED_UNIVERSES}"
        )

    # 4. Date format and ordering
    try:
        start = datetime.strptime(alpha["start_date"], _DATE_FMT)
        end = datetime.strptime(alpha["end_date"], _DATE_FMT)
        if start >= end:
            errors.append("start_date must be before end_date")
    except ValueError as exc:
        errors.append(f"Invalid date format (expected YYYY-MM-DD): {exc}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Panel namespace — used by signal_calculation.compute_signal
# ---------------------------------------------------------------------------

def build_panel_namespace(processed_df: pd.DataFrame) -> dict:
    """Build the eval namespace for raw_formula evaluation.

    Each raw data column in processed_df is pivoted to a DATE × TICKER wide
    DataFrame, so the formula can use:
      - Time-series pandas methods directly: ADJUSTED_PRICE.shift(21)
      - Cross-sectional operators row-wise:  rank(ADJUSTED_PRICE.shift(21) / ...)
    """
    ns: dict = {
        "rank":   lambda df: df.rank(axis=1, pct=True),
        "zscore": lambda df: (
            df.sub(df.mean(axis=1), axis=0)
            .div(
                df.std(axis=1, ddof=1).replace(0, float("nan")).fillna(1.0) + 1e-9,
                axis=0,
            )
        ),
        "log":    lambda df: np.log(df.clip(lower=1e-9)),
        "abs":    lambda df: df.abs(),
        "sign":   lambda df: np.sign(df),
        "delta":   _ts_error,
        "ts_mean": _ts_error,
        "ts_std":  _ts_error,
        "np":    np,
        "float": float,
        "nan":   float("nan"),
    }

    data_cols = [c for c in processed_df.columns if c not in _NON_DATA_COLS]
    for col in data_cols:
        ns[col] = processed_df[col].unstack(level="TICKER")

    return ns


# ---------------------------------------------------------------------------
# Cross-sectional evaluator (legacy path used by backtest.py directly)
# ---------------------------------------------------------------------------

def evaluate_formula(
    formula: str,
    cross_section: dict[str, pd.Series],
) -> pd.Series:
    """Evaluate a formula against a dict of per-stock Series.

    Returns a signal Series indexed by TICKER.
    """
    ns = _build_cross_sectional_namespace(cross_section)
    try:
        result = eval(formula, {"__builtins__": {}}, ns)  # noqa: S307
    except NameError as exc:
        raise ValueError(
            f"Formula references unknown identifier: {exc}."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Formula evaluation failed: {exc}") from exc

    if not isinstance(result, pd.Series):
        raise ValueError(
            f"Formula must evaluate to a Series of per-stock values, got {type(result)}"
        )
    return result.dropna()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ts_error(*args, **kwargs):
    raise NotImplementedError(
        "delta(), ts_mean(), ts_std() are not available as formula functions. "
        "Use pandas DataFrame methods directly in the raw_formula instead:\n"
        "  shift:   ADJUSTED_PRICE.shift(21)\n"
        "  rolling: ADJUSTED_PRICE.rolling(20).std()\n"
        "  diff:    ADJUSTED_PRICE.diff(252)"
    )


def _build_cross_sectional_namespace(cross_section: dict[str, pd.Series]) -> dict:
    ns: dict = {
        "rank":   lambda s: s.rank(pct=True),
        "zscore": lambda s: (s - s.mean()) / (s.std() + 1e-9),
        "log":    lambda s: np.log(s.clip(lower=1e-9)),
        "abs":    lambda s: s.abs(),
        "sign":   lambda s: np.sign(s),
        "delta":   _ts_error,
        "ts_mean": _ts_error,
        "ts_std":  _ts_error,
    }
    ns.update(cross_section)
    return ns


def _tokenize(formula: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)


def _validate_raw_formula_tokens(
    formula: str, errors: list[str], warnings: list[str]
) -> None:
    if not formula.strip():
        errors.append("raw_formula must not be empty")
        return

    _check_parens(formula, errors)
    if errors:
        return

    tokens = set(_tokenize(formula))
    if not (tokens & AVAILABLE_RAW_COLUMNS):
        errors.append(
            "raw_formula does not reference any known data column. "
            f"Available columns: {sorted(AVAILABLE_RAW_COLUMNS)}"
        )

    known_identifiers = (
        AVAILABLE_RAW_COLUMNS
        | ALLOWED_FUNCTION_NAMES
        | {"np"}
        | {"shift", "rolling", "diff", "pct_change", "mean", "std",
           "fillna", "clip", "replace", "abs", "sum", "min", "max",
           "float", "nan"}
    )
    for token in tokens:
        if token.isupper() and "_" in token and token not in known_identifiers:
            warnings.append(
                f"'{token}' looks like a column name but is not in AVAILABLE_RAW_COLUMNS — "
                "will raise NameError at evaluation time."
            )


def _check_parens(formula: str, errors: list[str]) -> None:
    depth = 0
    max_depth = 0
    for ch in formula:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                errors.append("Unbalanced parentheses in formula (unexpected ')')")
                return
    if depth != 0:
        errors.append("Unbalanced parentheses in formula (unclosed '(')")
    if max_depth > 5:
        errors.append(f"Formula nesting depth {max_depth} exceeds maximum of 5")
