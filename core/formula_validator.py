"""Alpha formula validator, namespace builder, and cross-sectional evaluator."""

from __future__ import annotations

import ast
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
    "REVENUE_LTM",
    "COGS_LTM",
    "GROSS_PROFIT_LTM",
    "OPERATING_EXPENSES_LTM",
    "SGA_EXPENSE_LTM",
    "OPERATING_INCOME_LTM",
    "NON_OPERATING_INCOME_LTM",
    "NET_INTEREST_EXPENSE_LTM",
    "PRETAX_INCOME_ADJ_LTM",
    "PRETAX_INCOME_LTM",
    "INCOME_TAX_LTM",
    "CONTINUING_INCOME_LTM",
    "NET_INCOME_LTM",
    "NET_INCOME_COMMON_LTM",
    "EPS_DILUTED",
    "SHARES_BASIC",
    "SHARES_DILUTED",
    # Cash flow statement — TTM, cross-sectionally winsorised + standardised
    "NET_INCOME_START_LTM",
    "DA_LTM",
    "NON_CASH_ITEMS_LTM",
    "WORKING_CAPITAL_CHANGE_LTM",
    "CFO_LTM",
    "FIXED_ASSET_CHANGE_LTM",
    "CFI_LTM",
    "DEBT_FINANCING_CF_LTM",
    "EQUITY_FINANCING_CF_LTM",
    "CFF_LTM",
    "NET_CHANGE_CASH_LTM",
})

# All operator names available in the panel eval namespace.
ALLOWED_FUNCTION_NAMES: set[str] = {
    # Cross-sectional (operate across tickers per date)
    "rank", "zscore", "sign", "log", "abs", "scale",
    "tanh", "sigmoid", "exp", "sqrt",
    "power", "sign_power", "max", "min", "clip", "where",
    "group_rank", "group_zscore", "indneutralize",
    # Time-series (operate along date axis per ticker)
    "delta", "ts_delta", "ts_shift",
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum",
    "ts_rank", "ts_argmax", "ts_argmin",
    "ts_corr", "ts_cov", "ts_av_diff", "ts_zscore",
    "decay_linear", "product",
    # Technical indicators
    "ema", "sma", "wma", "rsi", "macd",
    "boll_upper", "boll_lower", "boll_mid",
}

ALLOWED_UNIVERSES: set[str] = {"sp500"}

# Canonical operator reference injected into every LLM prompt that may produce a formula.
FORMULA_CONSTRAINT: str = (
    "IMPORTANT — `formula` uses raw DataFrame column names directly.\n"
    "Each column is a full DATE × TICKER pandas DataFrame.\n"
    "All fundamental columns are already winsorized and standardized cross-sectionally "
    "(z-scored per date) — do NOT apply zscore() or rank() as a first step on raw "
    "fundamentals; use them to combine or transform signals.\n"
    f"Available columns: {', '.join(sorted(AVAILABLE_RAW_COLUMNS))}\n"
    "\n"
    "Cross-sectional operators (across tickers per date):\n"
    "  rank(X)  zscore(X)  sign(X)  log(X)  abs(X)  scale(X)  tanh(X)  sigmoid(X)  exp(X)  sqrt(X)\n"
    "  power(X, n)  sign_power(X, n)  max(A, B)  min(A, B)  clip(X, lo, hi)  where(cond, t, f)\n"
    "  group_rank(X, SECTOR)  group_zscore(X, SECTOR)  indneutralize(X, SECTOR)\n"
    "\n"
    "Time-series operators (along date axis per ticker):\n"
    "  ts_mean(X, n)  ts_std(X, n)  ts_max(X, n)  ts_min(X, n)  ts_sum(X, n)\n"
    "  ts_shift(X, n)  ts_delta(X, n)  delta(X, n)\n"
    "  ts_rank(X, n)  ts_argmax(X, n)  ts_argmin(X, n)\n"
    "  ts_corr(X, Y, n)  ts_cov(X, Y, n)\n"
    "  decay_linear(X, n)  product(X, n)\n"
    "  ts_av_diff(X, n)  ts_zscore(X, n)\n"
    "\n"
    "Technical indicators:\n"
    "  ema(X, n)  sma(X, n)  wma(X, n)  rsi(X, n)  macd(X, n)\n"
    "  boll_upper(X, n)  boll_lower(X, n)  boll_mid(X, n)\n"
    "\n"
    "Pandas methods work inline: X.shift(n)  X.pct_change()  X.diff(n)  X.rolling(n).mean()\n"
    "Standard arithmetic: +  -  *  /  **\n"
    "All fundamental columns are TTM (trailing twelve months), not point-in-time.\n"
    "All fundamental columns are already clean with no NaN values — do NOT use .fillna() or .replace()."
)

REQUIRED_KEYS: list[str] = [
    "alpha_id", "formula", "universe", "start_date", "end_date",
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

    # 2. Formula validation
    formula = alpha.get("formula", "")
    _validate_formula_tokens(formula, errors, warnings)

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

# --- operator helpers -------------------------------------------------------

def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sigma = df.std(axis=1, ddof=1).replace(0, float("nan")).fillna(1.0) + 1e-9
    return df.sub(mu, axis=0).div(sigma, axis=0)

def _cs_scale(df: pd.DataFrame) -> pd.DataFrame:
    lo = df.min(axis=1)
    hi = df.max(axis=1)
    rng = (hi - lo).replace(0, float("nan"))
    return df.sub(lo, axis=0).div(rng, axis=0)

def _ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
    )

def _decay_linear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    w = np.arange(1, n + 1, dtype=float); w /= w.sum()
    return df.rolling(n).apply(lambda x: (x * w).sum(), raw=True)

def _rsi(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d = df.diff()
    gain = d.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    loss = (-d).clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def _macd(df: pd.DataFrame, n: int) -> pd.DataFrame:
    fast = df.ewm(span=max(n // 2, 1), adjust=False).mean()
    slow = df.ewm(span=n, adjust=False).mean()
    line = fast - slow
    return line - line.ewm(span=max(n // 4, 1), adjust=False).mean()

def _group_rank(col: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    col_s = col.stack(future_stack=True)
    grp_s = group.stack(future_stack=True).reindex(col_s.index)
    ranked = col_s.groupby([col_s.index.get_level_values(0), grp_s]).rank(pct=True)
    return ranked.unstack()

def _group_zscore(col: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    col_s = col.stack(future_stack=True)
    grp_s = group.stack(future_stack=True).reindex(col_s.index)
    zs = col_s.groupby([col_s.index.get_level_values(0), grp_s]).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    return zs.unstack()

def _where(cond: pd.DataFrame, t, f) -> pd.DataFrame:
    c = cond.astype(bool)
    tv = t.values if hasattr(t, "values") else t
    fv = f.values if hasattr(f, "values") else f
    return pd.DataFrame(np.where(c.values, tv, fv), index=c.index, columns=c.columns)

# ---------------------------------------------------------------------------

def build_panel_namespace(processed_df: pd.DataFrame) -> dict:
    """Build the eval namespace for formula evaluation.

    Each data column is pivoted to a DATE × TICKER wide DataFrame so formulas
    can use pandas time-series methods and the operators defined below.
    """
    ns: dict = {
        # --- cross-sectional (row-wise across tickers per date) ---
        "rank":         lambda df: df.rank(axis=1, pct=True),
        "zscore":       _cs_zscore,
        "sign":         lambda df: np.sign(df),
        "log":          lambda df: np.log(df.clip(lower=1e-9)),
        "abs":          lambda df: df.abs(),
        "scale":        _cs_scale,
        "tanh":         lambda df: np.tanh(df),
        "sigmoid":      lambda df: 1.0 / (1.0 + np.exp(-df.clip(-500, 500))),
        "exp":          lambda df: np.exp(df.clip(upper=500)),
        "sqrt":         lambda df: df.clip(lower=0).pow(0.5),
        "power":        lambda base, exp: base ** exp,
        "sign_power":   lambda base, exp: np.sign(base) * (base.abs() ** exp),
        "max":          lambda a, b: np.maximum(a, b),
        "min":          lambda a, b: np.minimum(a, b),
        "clip":         lambda df, lo, hi: df.clip(lower=lo, upper=hi),
        "where":        _where,
        "group_rank":   _group_rank,
        "group_zscore": _group_zscore,
        "indneutralize": _group_zscore,
        # --- time-series (column-wise along date axis per ticker) ---
        "delta":        lambda df, n: df.diff(n),
        "ts_delta":     lambda df, n: df.diff(n),
        "ts_shift":     lambda df, n: df.shift(n),
        "ts_mean":      lambda df, n: df.rolling(n).mean(),
        "ts_std":       lambda df, n: df.rolling(n).std(),
        "ts_max":       lambda df, n: df.rolling(n).max(),
        "ts_min":       lambda df, n: df.rolling(n).min(),
        "ts_sum":       lambda df, n: df.rolling(n).sum(),
        "ts_rank":      _ts_rank,
        "ts_argmax":    lambda df, n: df.rolling(n).apply(lambda x: float(np.argmax(x)), raw=True),
        "ts_argmin":    lambda df, n: df.rolling(n).apply(lambda x: float(np.argmin(x)), raw=True),
        "ts_corr":      lambda df1, df2, n: df1.rolling(n).corr(df2),
        "ts_cov":       lambda df1, df2, n: df1.rolling(n).cov(df2),
        "decay_linear": _decay_linear,
        "product":      lambda df, n: df.rolling(n).apply(np.prod, raw=True),
        "ts_av_diff":   lambda df, n: df - df.rolling(n).mean(),
        "ts_zscore":    lambda df, n: (df - df.rolling(n).mean()) / (df.rolling(n).std() + 1e-9),
        # --- technical indicators ---
        "ema":          lambda df, n: df.ewm(span=n, adjust=False).mean(),
        "sma":          lambda df, n: df.rolling(n).mean(),
        "wma":          _decay_linear,
        "rsi":          _rsi,
        "macd":         _macd,
        "boll_upper":   lambda df, n: df.rolling(n).mean() + 2 * df.rolling(n).std(),
        "boll_lower":   lambda df, n: df.rolling(n).mean() - 2 * df.rolling(n).std(),
        "boll_mid":     lambda df, n: df.rolling(n).mean(),
        # --- python builtins ---
        "np":    np,
        "float": float,
        "nan":   float("nan"),
    }

    data_cols = [c for c in processed_df.columns if c not in _NON_DATA_COLS]
    for col in data_cols:
        ns[col] = processed_df[col].unstack(level="TICKER")

    # Make SECTOR available for group_rank / group_zscore / indneutralize
    if "SECTOR" in processed_df.columns:
        ns["SECTOR"] = processed_df["SECTOR"].unstack(level="TICKER")

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

def _build_cross_sectional_namespace(cross_section: dict[str, pd.Series]) -> dict:
    def _ts_unsupported(*args, **kwargs):
        raise NotImplementedError(
            "delta(), ts_mean(), ts_std() require a full DATE × TICKER panel — "
            "they are not available in the legacy cross-sectional evaluator."
        )

    ns: dict = {
        "rank":    lambda s: s.rank(pct=True),
        "zscore":  lambda s: (s - s.mean()) / (s.std() + 1e-9),
        "log":     lambda s: np.log(s.clip(lower=1e-9)),
        "abs":     lambda s: s.abs(),
        "sign":    lambda s: np.sign(s),
        "delta":   _ts_unsupported,
        "ts_mean": _ts_unsupported,
        "ts_std":  _ts_unsupported,
    }
    ns.update(cross_section)
    return ns


def _tokenize(formula: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)


# ---------------------------------------------------------------------------
# Formula complexity — used by decision.score_alpha's simplicity sub-score
# ---------------------------------------------------------------------------

_COMPLEXITY_NODE_TYPES = (ast.Call, ast.BinOp, ast.UnaryOp, ast.Compare)


def formula_complexity(formula: str) -> int:
    """Structural complexity: n_calls + max_expr_depth + n_distinct_columns.

    Calibration: `rank(X) * -1` → 4; the quality+value fallback formula
    (`rank(A/B) + rank(C/(D/E)) * -1`) → ~11; heavily nested LLM-gamed
    formulas land ≥ 20. Falls back to token count // 2 on SyntaxError.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return len(_tokenize(formula)) // 2

    n_calls = sum(isinstance(node, ast.Call) for node in ast.walk(tree))
    depth = _expr_depth(tree.body)
    n_cols = len(set(_tokenize(formula)) & AVAILABLE_RAW_COLUMNS)
    return n_calls + depth + n_cols


def _expr_depth(node: ast.AST) -> int:
    """Max nesting depth counting only Call/BinOp/UnaryOp/Compare nodes."""
    own = 1 if isinstance(node, _COMPLEXITY_NODE_TYPES) else 0
    child_depths = [_expr_depth(child) for child in ast.iter_child_nodes(node)]
    return own + (max(child_depths) if child_depths else 0)


def _validate_formula_tokens(
    formula: str, errors: list[str], warnings: list[str]
) -> None:
    if not formula.strip():
        errors.append("formula must not be empty")
        return

    _check_parens(formula, errors)
    if errors:
        return

    tokens = set(_tokenize(formula))
    if not (tokens & AVAILABLE_RAW_COLUMNS):
        errors.append(
            "formula does not reference any known data column. "
            f"Available columns: {sorted(AVAILABLE_RAW_COLUMNS)}"
        )

    pandas_methods = {"shift", "rolling", "diff", "pct_change", "mean", "std",
                      "clip", "abs", "sum", "min", "max", "float", "nan"}
    known_identifiers = AVAILABLE_RAW_COLUMNS | ALLOWED_FUNCTION_NAMES | {"np"} | pandas_methods

    for token in tokens:
        if token.isupper() and "_" in token and token not in known_identifiers:
            warnings.append(
                f"'{token}' looks like a column name but is not in AVAILABLE_RAW_COLUMNS — "
                "will raise NameError at evaluation time."
            )
        elif (token.islower() or "_" in token) and token not in known_identifiers \
                and not token[0].isupper() and len(token) > 2:
            # Likely an unknown function call (e.g. ts_stddev instead of ts_std)
            errors.append(
                f"'{token}' is not a recognised operator. "
                f"Did you mean one of: {sorted(fn for fn in ALLOWED_FUNCTION_NAMES if fn.startswith(token[:4]))}? "
                f"See ALLOWED_FUNCTION_NAMES for the full list."
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
