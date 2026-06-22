"""Research planner — reads experiment memory and uses DeepSeek to suggest next directions.

Falls back to rule-based suggestions if the LLM call fails or returns invalid JSON.
"""

from __future__ import annotations

import json
import os
import re

from core.types import AlphaConfig, ResearchSuggestion
from core.validator import ALLOWED_FEATURES, validate_alpha

_SYSTEM_PROMPT = (
    "You are a quantitative research director reviewing a portfolio of alpha factor experiments. "
    "Based on what has been tried and what failed, suggest new research directions. "
    "Be specific about formulas and features. Think about diversifying signal sources."
)

_FORMULA_CONSTRAINT = (
    "IMPORTANT — supported formula operators ONLY: rank(), zscore(), log(), abs(), sign() "
    "and standard arithmetic (+, -, *, /, **). "
    "DO NOT use ts_mean(), ts_std(), or delta() — they are not implemented."
)

_FEATURE_DESCRIPTIONS = {
    "MOM12_1": "12-month momentum excluding last month",
    "MOM6_1": "6-month momentum excluding last month",
    "VOL_20D": "20-day rolling volatility of log returns",
    "LIQUIDITY": "20-day average dollar volume",
    "EBITDA_MARGIN": "EBITDA / Revenue (profitability)",
    "SALES_GROWTH": "Year-over-year revenue growth",
    "EPS_GROWTH": "Year-over-year EPS growth",
    "PRICE_TO_SALES": "Price / Revenue per share proxy",
    "EPS_LTM": "Trailing twelve months EPS",
    "SALES_LTM": "Trailing twelve months revenue",
    "EBITDA_LTM": "Trailing twelve months EBITDA",
    "COGS_LTM": "Trailing twelve months cost of goods sold",
    "ADJUSTED_PRICE": "Split/dividend-adjusted close price",
    "ADJUSTED_VOLUME": "Adjusted trading volume",
}

_AVAILABLE_FEATURES_DESC = "\n".join(
    f"  - {k}: {v}" for k, v in _FEATURE_DESCRIPTIONS.items()
)


def plan_next_research(
    store: "ExperimentStore",  # noqa: F821
    n: int = 3,
) -> list[ResearchSuggestion]:
    """Read all experiments from memory and suggest the next N research directions.

    Tries DeepSeek first; fills remaining slots with rule-based suggestions on failure.
    """
    from core.memory import ExperimentStore  # local import

    records = store.load_all()

    try:
        suggestions = _llm_plan(records, n)
    except Exception as exc:
        print(f"  [planner] LLM planning failed ({exc}), using rule-based fallback.")
        suggestions = []

    # Fill remaining slots with rule-based suggestions
    if len(suggestions) < n:
        fallback = _rule_based_plan(records, n - len(suggestions))
        suggestions.extend(fallback)

    return suggestions[:n]


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _llm_plan(records: list[dict], n: int) -> list[ResearchSuggestion]:
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = _build_plan_prompt(records, n)

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
    suggestions = _parse_suggestions(raw, records)
    return suggestions


def _build_plan_prompt(records: list[dict], n: int) -> str:
    if records:
        summary_rows = []
        for r in records:
            m = r.get("metrics", {})
            summary_rows.append(
                f"  - {r['alpha_id']}: formula={r['formula']} | "
                f"verdict={r['verdict']} | Sharpe={m.get('Sharpe', 0):.3f} | "
                f"ICIR={m.get('ICIR', 0):.3f} | turnover={m.get('turnover', 0):.0f}bps | "
                f"failure={r.get('failure_reason') or 'N/A'}"
            )
        history = "\n".join(summary_rows)
    else:
        history = "  (no experiments run yet)"

    # Include last reflection's Next Mutation hint
    reflections = ""
    for r in records[-3:]:
        refl = r.get("reflection", "")
        if "Next Mutation" in refl:
            line = next((l for l in refl.splitlines() if "Next Mutation" in l), "")
            if line:
                reflections += f"  - {r['alpha_id']}: {line.strip()}\n"

    return f"""You are planning the next round of alpha factor research.

Experiment history:
{history}

Recent mutation suggestions from prior reflections:
{reflections or "  (none)"}

Available features:
{_AVAILABLE_FEATURES_DESC}

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


def _parse_suggestions(raw: str, records: list[dict]) -> list[ResearchSuggestion]:
    data = _extract_json(raw)
    if not isinstance(data, list):
        raise ValueError("LLM returned a dict instead of a list")

    # Use the first record's config as a template for validation
    base_config = records[0]["config"] if records else {}

    valid: list[ResearchSuggestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Build a minimal AlphaConfig to validate formula + features
        test_config = AlphaConfig(
            alpha_id=f"plan_test_{len(valid)}",
            formula=item.get("formula", ""),
            features=item.get("features", []),
            universe=base_config.get("universe", "sp500"),
            start_date=base_config.get("start_date", "2018-01-01"),
            end_date=base_config.get("end_date", "2024-12-31"),
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


def _rule_based_plan(records: list[dict], n: int) -> list[ResearchSuggestion]:
    tried_features: set[str] = set()
    for r in records:
        tried_features.update(r.get("features") or [])

    # Prefer suggestions whose features haven't been tried
    untried = [s for s in _FALLBACK_SUGGESTIONS if not set(s["features"]) & tried_features]
    tried = [s for s in _FALLBACK_SUGGESTIONS if set(s["features"]) & tried_features]
    ordered = untried + tried

    # Wire parent_id to the most recent experiment if available
    last_id = records[-1]["alpha_id"] if records else None
    result = []
    for s in ordered[:n]:
        s = dict(s)  # type: ignore[assignment]
        if s["parent_id"] is None and last_id:
            s["parent_id"] = last_id
        result.append(ResearchSuggestion(**s))  # type: ignore[arg-type]

    return result


# ---------------------------------------------------------------------------
# Shared JSON helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | list:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)
