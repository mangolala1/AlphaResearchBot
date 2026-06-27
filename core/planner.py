"""Research planner — reads experiment memory and uses DeepSeek to suggest next directions.

Falls back to rule-based suggestions if the LLM call fails or returns invalid JSON.
"""

from __future__ import annotations

import json
import os
import re

from core.types import AlphaConfig, MemorySummary, ResearchSuggestion
from core.formula_validator import (
    ALLOWED_FUNCTION_NAMES, AVAILABLE_RAW_COLUMNS, validate_alpha,
)

_SYSTEM_PROMPT = (
    "You are a quantitative research director reviewing a portfolio of alpha factor experiments. "
    "Based on what has been tried and what failed, suggest new research directions. "
    "Be specific about formulas and features. Think about diversifying signal sources."
)

_FORMULA_CONSTRAINT = f"""IMPORTANT — two formula fields are required:

1. `formula` (display): free-form human-readable description of the signal for the UI.
   Example: "EBITDA margin quality + 12-month momentum, low-vol damped"

2. `raw_formula` (execution): each column is a full DATE × TICKER pandas DataFrame.
   Available columns: {', '.join(sorted(AVAILABLE_RAW_COLUMNS))}

   Cross-sectional operators (across tickers per date):
     rank(X)  zscore(X)  sign(X)  log(X)  abs(X)  scale(X)  tanh(X)  sigmoid(X)  exp(X)  sqrt(X)
     power(X, n)  sign_power(X, n)  max(A, B)  min(A, B)  clip(X, lo, hi)  where(cond, t, f)
     group_rank(X, SECTOR)  group_zscore(X, SECTOR)  indneutralize(X, SECTOR)

   Time-series operators (along date axis per ticker):
     ts_mean(X, n)  ts_std(X, n)  ts_max(X, n)  ts_min(X, n)  ts_sum(X, n)
     ts_shift(X, n)  ts_delta(X, n)  delta(X, n)
     ts_rank(X, n)  ts_argmax(X, n)  ts_argmin(X, n)
     ts_corr(X, Y, n)  ts_cov(X, Y, n)
     decay_linear(X, n)  product(X, n)
     ts_av_diff(X, n)  ts_zscore(X, n)

   Technical indicators:
     ema(X, n)  sma(X, n)  wma(X, n)  rsi(X, n)  macd(X, n)
     boll_upper(X, n)  boll_lower(X, n)  boll_mid(X, n)

   Pandas methods work inline: X.shift(n)  X.pct_change()  X.rolling(n).mean()
   Standard arithmetic: +  -  *  /  **
   All fundamental columns are TTM (trailing twelve months), not point-in-time — do NOT treat them as quarterly snapshots.
   All fundamental columns are already clean with no NaN values — do NOT use .fillna() or .replace()."""


def plan_next_research(
    store: "ExperimentStore",  # noqa: F821
    n: int = 3,
) -> list[ResearchSuggestion]:
    """Read all experiments from memory and suggest the next N research directions.

    Tries DeepSeek first; fills remaining slots with rule-based suggestions on failure.
    """
    from core.memory_analyzer import analyze_memory

    summary = analyze_memory(store)

    try:
        suggestions = _llm_plan(summary, n)
    except Exception as exc:
        print(f"  [planner] LLM planning failed ({exc}), using rule-based fallback.")
        suggestions = []

    # Fill remaining slots with rule-based suggestions
    if len(suggestions) < n:
        fallback = _rule_based_plan(summary, n - len(suggestions))
        suggestions.extend(fallback)

    return suggestions[:n]


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _llm_plan(summary: MemorySummary, n: int) -> list[ResearchSuggestion]:
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = _build_plan_prompt(summary, n)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()
    suggestions = _parse_suggestions(raw, summary)
    return suggestions


def _build_plan_prompt(summary: MemorySummary, n: int) -> str:
    vc = summary["verdict_counts"]
    status_line = (
        f"{summary['total_experiments']} total: "
        + ", ".join(f"{v} {k}" for k, v in sorted(vc.items()))
        if vc else "0 total"
    )

    fcc = summary["failure_category_counts"]
    failure_lines = (
        "  " + ", ".join(f"{k}: {v}" for k, v in sorted(fcc.items(), key=lambda x: -x[1]))
        if fcc else "  (none)"
    )

    best_lines = "\n".join(
        f"  {e['alpha_id']} | {e['formula']} | Sharpe={e['Sharpe']:.3f} | ICIR={e['ICIR']:.3f}"
        for e in summary["best_experiments"]
    ) or "  (none)"

    trend_lines = "\n".join(f"  - {obs}" for obs in summary["trend_observations"])

    explored = ", ".join(summary["explored_features"]) or "(none)"
    unexplored = ", ".join(summary["unexplored_features"]) or "(none)"

    return f"""You are planning the next round of alpha factor research.

== Memory Summary ==
Total: {status_line}
Failure patterns: {failure_lines}
Best experiments:
{best_lines}
Explored features: {explored}
Unexplored features: {unexplored}
Trend observations:
{trend_lines}

{_FORMULA_CONSTRAINT}

Suggest {n} NEW alpha research directions that are meaningfully different from what has been tried.
Prioritize unexplored signal types (value, quality, growth, volatility, liquidity).

Return ONLY a JSON array of {n} objects (no markdown, no explanation):
[
  {{
    "direction": "short label (3-5 words)",
    "hypothesis": "one sentence investment thesis",
    "formula": "display formula using named feature labels",
    "raw_formula": "execution formula using raw column DataFrames and pandas methods",
    "features": ["list", "of", "named", "features"],
    "parent_id": "alpha_id to branch from, or null",
    "rationale": "one sentence: why this direction given prior results"
  }},
  ...
]"""


def _parse_suggestions(raw: str, summary: MemorySummary) -> list[ResearchSuggestion]:
    data = _extract_json(raw)
    if not isinstance(data, list):
        raise ValueError("LLM returned a dict instead of a list")

    valid: list[ResearchSuggestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        raw_formula = item.get("raw_formula", "")
        if not raw_formula:
            print(f"  [planner] Skipping '{item.get('direction')}': missing raw_formula")
            continue

        test_config = AlphaConfig(
            alpha_id=f"plan_test_{len(valid)}",
            formula=item.get("formula", ""),
            raw_formula=raw_formula,
            features=item.get("features", []),
            universe="sp500",
            start_date="2021-01-01",
            end_date="2026-06-01",
        )
        result = validate_alpha(test_config)
        if not result.valid:
            print(f"  [planner] Skipping invalid suggestion '{item.get('direction')}': {result.errors}")
            continue

        valid.append(ResearchSuggestion(
            direction=item.get("direction", ""),
            hypothesis=item.get("hypothesis", ""),
            formula=item.get("formula", ""),
            raw_formula=raw_formula,
            features=item.get("features", []),
            parent_id=item.get("parent_id"),
            rationale=item.get("rationale", ""),
        ))

    return valid


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_FALLBACK_SUGGESTIONS: list[ResearchSuggestion] = [
    ResearchSuggestion(
        direction="value screen",
        hypothesis="Cheap stocks (low P/S) outperform expensive ones in the cross-section.",
        formula="rank(PRICE_TO_SALES) * -1",
        raw_formula="rank(ADJUSTED_PRICE / (REVENUE_LTM / SHARES_DILUTED)) * -1",
        features=["PRICE_TO_SALES"],
        parent_id=None,
        rationale="No value-based alpha has been tested yet.",
    ),
    ResearchSuggestion(
        direction="quality + value",
        hypothesis="High-quality cheap stocks outperform: profitable companies trading at low valuations.",
        formula="rank(EBITDA_MARGIN) + rank(PRICE_TO_SALES) * -1",
        raw_formula=(
            "rank((OPERATING_INCOME_LTM + DA_LTM) / REVENUE_LTM)"
            " + rank(ADJUSTED_PRICE / (REVENUE_LTM / SHARES_DILUTED)) * -1"
        ),
        features=["EBITDA_MARGIN", "PRICE_TO_SALES"],
        parent_id=None,
        rationale="Combining profitability and valuation is a classic Piotroski-style factor.",
    ),
    ResearchSuggestion(
        direction="earnings growth momentum",
        hypothesis="Companies with accelerating EPS growth attract institutional buying.",
        formula="rank(EPS_GROWTH) + 0.5 * rank(SALES_GROWTH)",
        raw_formula=(
            "rank(NET_INCOME_LTM / NET_INCOME_LTM.shift(252) - 1)"
            " + 0.5 * rank(SALES_LTM / SALES_LTM.shift(252) - 1)"
        ),
        features=["EPS_GROWTH", "SALES_GROWTH"],
        parent_id=None,
        rationale="Growth factors have not yet been tested in this experiment set.",
    ),
    ResearchSuggestion(
        direction="low volatility",
        hypothesis="Low-volatility stocks generate better risk-adjusted returns due to investor preference for lotteries.",
        formula="rank(VOL_20D) * -1",
        raw_formula="rank(ADJUSTED_PRICE.pct_change().rolling(20).std()) * -1",
        features=["VOL_20D"],
        parent_id=None,
        rationale="The low-vol anomaly is well-documented and unexplored here.",
    ),
    ResearchSuggestion(
        direction="liquidity + momentum",
        hypothesis="Liquid momentum stocks outperform because they are easier to trade and attract trend-following flows.",
        formula="rank(MOM6_1) * rank(LIQUIDITY)",
        raw_formula=(
            "rank(ADJUSTED_PRICE.shift(21) / ADJUSTED_PRICE.shift(126) - 1)"
            " * rank(ADJUSTED_VOLUME * ADJUSTED_PRICE)"
        ),
        features=["MOM6_1", "LIQUIDITY"],
        parent_id=None,
        rationale="Combining liquidity with shorter-term momentum may reduce turnover vs MOM12_1.",
    ),
]


def _rule_based_plan(summary: MemorySummary, n: int) -> list[ResearchSuggestion]:
    tried_features = set(summary["explored_features"])

    # Prefer suggestions whose features haven't been tried
    untried = [s for s in _FALLBACK_SUGGESTIONS if not set(s["features"]) & tried_features]
    tried = [s for s in _FALLBACK_SUGGESTIONS if set(s["features"]) & tried_features]
    ordered = untried + tried

    result = []
    for s in ordered[:n]:
        result.append(ResearchSuggestion(**dict(s)))  # type: ignore[arg-type]

    return result


# ---------------------------------------------------------------------------
# Shared JSON helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | list:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)
