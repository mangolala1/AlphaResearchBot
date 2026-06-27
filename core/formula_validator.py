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

# Raw data columns that can appear in a raw_formula.
# These are the columns produced by data_loader and processed by data_process.
AVAILABLE_RAW_COLUMNS: frozenset[str] = frozenset({
    # Price
    "ADJUSTED_PRICE", "ADJUSTED_VOLUME",
    # Income statement (TTM)
    "SALES_LTM", "COGS_LTM", "NET_INCOME_LTM", "OPER_INCOME_LTM",
    # Cash flow (TTM)
    "DA_LTM", "SHARES_DILUTED", "INV_CHANGE_LTM",
})

# Named feature labels used in the display formula (formula field).
# These are the human-readable shorthand names shown in the UI.
ALLOWED_FEATURES: set[str] = {
    # Raw pass-throughs (same name as raw column)
    "ADJUSTED_PRICE", "ADJUSTED_VOLUME",
    "SALES_LTM", "COGS_LTM", "NET_INCOME_LTM", "SHARES_DILUTED", "INV_CHANGE_LTM",
    # Forward-looking (paywalled — warn if used)
    "EPS_NTM", "SALES_NTM", "EBITDA_NTM", "COGS_NTM",
    # Metadata
    "SECTOR", "INDUSTRY", "TICKER",
    # Named derived features (for display only)
    "EPS_LTM", "EBITDA_LTM",
    "EBITDA_MARGIN", "NET_MARGIN",
    "MOM12_1", "MOM6_1",
    "SALES_GROWTH", "EPS_GROWTH",
    "PRICE_TO_SALES", "VOL_20D", "LIQUIDITY",
}

EVALUATOR_FEATURES: frozenset[str] = frozenset(ALLOWED_FEATURES) - {
    "SECTOR", "INDUSTRY", "TICKER",
    "EPS_NTM", "SALES_NTM", "EBITDA_NTM", "COGS_NTM",
}

FUTURE_LOOKING_FIELDS: set[str] = {
    "EPS_NTM", "SALES_NTM", "EBITDA_NTM", "COGS_NTM",
}

# All function names the LLM may write in a formula string.
# delta/ts_mean/ts_std are listed so the validator can identify them;
# they raise NotImplementedError at evaluation time.
ALLOWED_FUNCTION_NAMES: set[str] = {
    "rank", "zscore", "log", "abs", "sign", "delta", "ts_mean", "ts_std",
}

ALLOWED_UNIVERSES: set[str] = {"sp500", "russell1000", "russell3000"}

REQUIRED_KEYS: list[str] = [
    "alpha_id", "formula", "raw_formula", "universe", "start_date", "end_date",
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

    # 2. Display formula token validation (formula field uses named feature labels)
    display_formula = alpha.get("formula", "")
    _validate_display_formula(display_formula, errors, warnings)

    # 3. Raw formula validation (raw_formula uses actual data column names)
    raw_formula = alpha.get("raw_formula", "")
    _validate_raw_formula_tokens(raw_formula, errors, warnings)

    # 4. Universe
    if alpha.get("universe") not in ALLOWED_UNIVERSES:
        errors.append(
            f"Universe '{alpha.get('universe')}' not supported. "
            f"Choose from: {ALLOWED_UNIVERSES}"
        )

    # 5. Date format and ordering
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

    Args:
        processed_df: (DATE, TICKER) MultiIndex DataFrame from data_process.process().

    Returns:
        dict mapping column names and operator names to their values/callables.
    """
    ns: dict = {
        # Cross-sectional operators — row-wise on DATE × TICKER DataFrames
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
        # Blocked — pandas methods should be used directly
        "delta":   _ts_error,
        "ts_mean": _ts_error,
        "ts_std":  _ts_error,
        # Numpy available for math constants / functions
        "np": np,
    }

    # Pivot every data column to a wide DATE × TICKER DataFrame
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
    """Evaluate a display formula against a dict of per-stock Series.

    Returns a signal Series indexed by TICKER.
    """
    ns = _build_cross_sectional_namespace(cross_section)
    try:
        result = eval(formula, {"__builtins__": {}}, ns)  # noqa: S307
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
    """Namespace for cross-sectional (per-date Series) formula evaluation."""
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
    """Split formula into identifier tokens (words only, not method chains)."""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)


def _validate_display_formula(
    formula: str, errors: list[str], warnings: list[str]
) -> None:
    """Validate the display formula (uses named feature labels)."""
    if not formula.strip():
        errors.append("formula (display) must not be empty")
        return

    _check_parens(formula, errors)
    if errors:
        return

    for token in _tokenize(formula):
        if token not in ALLOWED_FEATURES and token not in ALLOWED_FUNCTION_NAMES:
            errors.append(
                f"Unknown identifier '{token}' in display formula — "
                "not a recognised feature label or operator"
            )

    for feat in _tokenize(formula):
        if feat in FUTURE_LOOKING_FIELDS:
            warnings.append(
                f"Feature '{feat}' is forward-looking (NTM) — "
                "not available from free SimFin data. Use LTM variants instead."
            )


def _validate_raw_formula_tokens(
    formula: str, errors: list[str], warnings: list[str]
) -> None:
    """Validate the raw_formula (uses actual data column names + pandas methods)."""
    if not formula.strip():
        errors.append("raw_formula must not be empty")
        return

    _check_parens(formula, errors)
    if errors:
        return

    # Check that at least one known raw column is referenced
    tokens = set(_tokenize(formula))
    referenced_cols = tokens & AVAILABLE_RAW_COLUMNS
    if not referenced_cols:
        errors.append(
            "raw_formula does not reference any known data column. "
            f"Available columns: {sorted(AVAILABLE_RAW_COLUMNS)}"
        )

    # Warn about tokens that look like column names but aren't available
    known_identifiers = (
        AVAILABLE_RAW_COLUMNS
        | ALLOWED_FUNCTION_NAMES
        | {"np"}
        # common pandas method names used inline
        | {"shift", "rolling", "diff", "pct_change", "mean", "std",
           "fillna", "clip", "replace", "abs", "sum", "min", "max"}
    )
    for token in tokens:
        if token.isupper() and "_" in token and token not in known_identifiers:
            warnings.append(
                f"'{token}' looks like a column name but is not in AVAILABLE_RAW_COLUMNS. "
                "It will raise a NameError at evaluation time if not present in the data."
            )


def _check_parens(formula: str, errors: list[str]) -> None:
    """Check parenthesis balance and max nesting depth."""
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
