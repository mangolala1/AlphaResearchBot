"""Alpha formula and config validator."""

from __future__ import annotations

import re
from datetime import datetime

from core.types import AlphaConfig, ValidationResult

ALLOWED_FEATURES: set[str] = {
    # Raw price / fundamental columns
    "EPS_LTM", "EPS_NTM",
    "SALES_LTM", "SALES_NTM",
    "EBITDA_LTM", "EBITDA_NTM",
    "COGS_LTM", "COGS_NTM",
    "NET_INCOME_LTM", "SHARES_DILUTED", "INV_CHANGE_LTM",
    "ADJUSTED_PRICE", "ADJUSTED_VOLUME",
    "SECTOR", "INDUSTRY", "TICKER",
    # Derived / computed features
    "EBITDA_MARGIN", "NET_MARGIN", "MOM12_1", "MOM6_1",
    "SALES_GROWTH", "EPS_GROWTH",
    "PRICE_TO_SALES", "VOL_20D", "LIQUIDITY",
}

EVALUATOR_FEATURES: frozenset[str] = ALLOWED_FEATURES - {
    "SECTOR", "INDUSTRY", "TICKER",
    "EPS_NTM", "SALES_NTM", "EBITDA_NTM", "COGS_NTM",
}

FUTURE_LOOKING_FIELDS: set[str] = {
    "EPS_NTM", "SALES_NTM", "EBITDA_NTM", "COGS_NTM",
}

ALLOWED_FUNCTION_NAMES: set[str] = {
    "rank", "zscore", "log", "abs", "sign", "delta", "ts_mean", "ts_std",
}

ALLOWED_UNIVERSES: set[str] = {"sp500", "russell1000", "russell3000"}

REQUIRED_KEYS: list[str] = [
    "alpha_id", "formula", "features", "universe", "start_date", "end_date",
]

_DATE_FMT = "%Y-%m-%d"


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

    # 2. Features in allowed set
    for feat in alpha.get("features", []):
        if feat not in ALLOWED_FEATURES:
            errors.append(f"Feature '{feat}' is not in the allowed feature set")
        elif feat in FUTURE_LOOKING_FIELDS:
            warnings.append(
                f"Feature '{feat}' is an NTM (forward-looking) estimate — "
                "these fields are not available in V2 free data sources (no analyst consensus). "
                "Use LTM variants (EPS_LTM, SALES_LTM, etc.) instead."
            )

    # 3. Formula token validation
    formula = alpha.get("formula", "")
    _validate_formula_tokens(formula, errors, warnings)

    # 4. Features referenced in formula match declared features list
    declared = set(alpha.get("features", []))
    referenced = {tok for tok in _tokenize(formula) if tok in ALLOWED_FEATURES}
    missing_from_list = referenced - declared
    missing_from_formula = declared - referenced
    if missing_from_list:
        warnings.append(
            f"Features referenced in formula but not in features list: {missing_from_list}"
        )
    if missing_from_formula:
        warnings.append(
            f"Features declared in features list but not found in formula: {missing_from_formula}"
        )

    # 5. Universe
    if alpha.get("universe") not in ALLOWED_UNIVERSES:
        errors.append(
            f"Universe '{alpha.get('universe')}' not supported. "
            f"Choose from: {ALLOWED_UNIVERSES}"
        )

    # 6. Date format and ordering
    try:
        start = datetime.strptime(alpha["start_date"], _DATE_FMT)
        end = datetime.strptime(alpha["end_date"], _DATE_FMT)
        if start >= end:
            errors.append("start_date must be before end_date")
    except ValueError as exc:
        errors.append(f"Invalid date format (expected YYYY-MM-DD): {exc}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def _tokenize(formula: str) -> list[str]:
    """Split formula into identifier tokens."""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)


def _validate_formula_tokens(
    formula: str, errors: list[str], warnings: list[str]
) -> None:
    """Check that formula only uses allowed tokens and has reasonable nesting depth."""
    if not formula.strip():
        errors.append("Formula must not be empty")
        return

    # Check parenthesis balance and max depth
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
        return
    if max_depth > 5:
        errors.append(f"Formula nesting depth {max_depth} exceeds maximum of 5")

    # Check identifiers — every word token must be a known feature or function name
    for token in _tokenize(formula):
        if token not in ALLOWED_FEATURES and token not in ALLOWED_FUNCTION_NAMES:
            errors.append(
                f"Unknown identifier '{token}' in formula — "
                "not a recognized feature or operator"
            )
