"""Research planner — reads experiment memory and uses DeepSeek to suggest next directions.

Falls back to rule-based suggestions if the LLM call fails or returns invalid JSON.
"""

from __future__ import annotations

import json
import os
import re

from core.types import AlphaConfig, MemorySummary, ResearchSuggestion
from core.formula_validator import ALLOWED_FUNCTION_NAMES, EVALUATOR_FEATURES, validate_alpha

_SYSTEM_PROMPT = (
    "You are a quantitative research director reviewing a portfolio of alpha factor experiments. "
    "Based on what has been tried and what failed, suggest new research directions. "
    "Be specific about formulas and features. Think about diversifying signal sources."
)

_SAFE_OPERATORS: frozenset[str] = ALLOWED_FUNCTION_NAMES - {"delta", "ts_mean", "ts_std"}

_FORMULA_CONSTRAINT = (
    f"IMPORTANT — supported formula operators ONLY: {', '.join(sorted(_SAFE_OPERATORS))}() "
    "and standard arithmetic (+, -, *, /, **). "
    "DO NOT use ts_mean(), ts_std(), or delta() — they raise NotImplementedError at runtime. "
    f"Supported features: {', '.join(sorted(EVALUATOR_FEATURES))}."
)


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
    "formula": "valid formula string",
    "features": ["list", "of", "features"],
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
        test_config = AlphaConfig(
            alpha_id=f"plan_test_{len(valid)}",
            formula=item.get("formula", ""),
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
        features=["PRICE_TO_SALES"],
        parent_id=None,
        rationale="No value-based alpha has been tested yet.",
    ),
    ResearchSuggestion(
        direction="quality + value",
        hypothesis="High-quality cheap stocks outperform: profitable companies trading at low valuations.",
        formula="rank(EBITDA_MARGIN) + rank(PRICE_TO_SALES) * -1",
        features=["EBITDA_MARGIN", "PRICE_TO_SALES"],
        parent_id=None,
        rationale="Combining profitability and valuation is a classic Piotroski-style factor.",
    ),
    ResearchSuggestion(
        direction="earnings growth momentum",
        hypothesis="Companies with accelerating EPS growth attract institutional buying.",
        formula="rank(EPS_GROWTH) + 0.5 * rank(SALES_GROWTH)",
        features=["EPS_GROWTH", "SALES_GROWTH"],
        parent_id=None,
        rationale="Growth factors have not yet been tested in this experiment set.",
    ),
    ResearchSuggestion(
        direction="low volatility",
        hypothesis="Low-volatility stocks generate better risk-adjusted returns due to investor preference for lotteries.",
        formula="rank(VOL_20D) * -1",
        features=["VOL_20D"],
        parent_id=None,
        rationale="The low-vol anomaly is well-documented and unexplored here.",
    ),
    ResearchSuggestion(
        direction="liquidity + momentum",
        hypothesis="Liquid momentum stocks outperform because they are easier to trade and attract trend-following flows.",
        formula="rank(MOM6_1) * rank(LIQUIDITY)",
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
