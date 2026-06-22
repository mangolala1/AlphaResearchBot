"""Alpha mutation generator — uses DeepSeek to produce a child AlphaConfig from a parent.

Falls back to rule-based mutation if the LLM call fails or returns invalid JSON.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from core.types import AlphaConfig
from core.validator import ALLOWED_FEATURES, ALLOWED_FUNCTION_NAMES, EVALUATOR_FEATURES, validate_alpha

_SYSTEM_PROMPT = (
    "You are a quantitative research analyst designing alpha factors. "
    "You will be given a parent alpha experiment that failed or underperformed. "
    "Your job is to propose one specific mutation that addresses the failure mode. "
    "Be precise and concrete. Only suggest formulas using supported operators."
)

_SAFE_OPERATORS: frozenset[str] = ALLOWED_FUNCTION_NAMES - {"delta", "ts_mean", "ts_std"}
_AVAILABLE_FEATURES = ", ".join(sorted(EVALUATOR_FEATURES))

_FORMULA_CONSTRAINT = (
    f"IMPORTANT — supported formula operators ONLY: {', '.join(sorted(_SAFE_OPERATORS))}() "
    "and standard arithmetic (+, -, *, /, **). "
    "ts_mean(), ts_std(), delta() raise NotImplementedError — never use them. "
    f"Only use these features: {_AVAILABLE_FEATURES}."
)


def generate_mutation(
    parent_id: str,
    store: "ExperimentStore",  # noqa: F821
) -> AlphaConfig:
    """Generate a mutated child AlphaConfig from a parent experiment record.

    Tries DeepSeek first; falls back to rule-based mutation on any failure.
    """
    from core.memory import ExperimentStore  # local import

    record = store.load_by_id(parent_id)
    if record is None:
        raise ValueError(f"Parent alpha '{parent_id}' not found in store.")

    try:
        return _llm_mutation(record)
    except Exception as exc:
        print(f"  [mutator] LLM mutation failed ({exc}), using rule-based fallback.")
        return _rule_based_mutation(record)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _llm_mutation(record: dict) -> AlphaConfig:
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = _build_mutation_prompt(record)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    raw = response.choices[0].message.content.strip()
    child = _parse_and_validate(raw, record)
    return child


def _build_mutation_prompt(record: dict) -> str:
    metrics = record.get("metrics", {})
    robustness = record.get("robustness", {})
    config = record.get("config", {})

    return f"""A parent alpha experiment has completed. Propose ONE mutation to improve it.

Parent Alpha:
- alpha_id: {record["alpha_id"]}
- Formula: {record["formula"]}
- Features: {record["features"]}
- Universe: {config.get("universe")} | Rebalance: {config.get("rebalance")} | Neutralization: {config.get("neutralization")}
- Period: {config.get("start_date")} to {config.get("end_date")}

Results:
- Verdict: {record["verdict"].upper()}
- Failure Reason: {record.get("failure_reason") or "N/A"}
- IC_mean: {metrics.get("IC_mean", 0):.4f}
- ICIR: {metrics.get("ICIR", 0):.4f}
- Sharpe: {metrics.get("Sharpe", 0):.4f}
- Turnover: {metrics.get("turnover", 0):.1f} bps/yr
- Sector Stability: {robustness.get("sector_stability", 0):.4f}
- Subperiod Stability: {robustness.get("subperiod_stability", 0):.4f}

Prior Reflection:
{record.get("reflection", "N/A")}

Available features: {_AVAILABLE_FEATURES}

{_FORMULA_CONSTRAINT}

Return ONLY a JSON object (no markdown, no explanation) with these fields:
{{
  "hypothesis": "one sentence investment thesis",
  "formula": "valid formula string",
  "features": ["list", "of", "features", "used"],
  "mutation": "one sentence describing what changed vs parent",
  "universe": "{config.get("universe", "sp500")}",
  "start_date": "{config.get("start_date", "2018-01-01")}",
  "end_date": "{config.get("end_date", "2024-12-31")}",
  "neutralization": "{config.get("neutralization", "sector")}",
  "rebalance": "monthly or quarterly",
  "transaction_cost_bps": {config.get("transaction_cost_bps", 5)},
  "holding_period_days": {config.get("holding_period_days", 20)}
}}"""


def _parse_and_validate(raw: str, parent_record: dict) -> AlphaConfig:
    parent_id = parent_record["alpha_id"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    new_alpha_id = f"{parent_id}_mut_{timestamp}"

    data = _extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM returned a list instead of a dict")

    data["alpha_id"] = new_alpha_id
    data["parent_id"] = parent_id

    result = validate_alpha(data)
    if not result.valid:
        # Retry: try to fix common issue — formula referencing undeclared features
        raise ValueError(f"Generated alpha failed validation: {result.errors}")

    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_based_mutation(record: dict) -> AlphaConfig:
    config = record.get("config", {})
    metrics = record.get("metrics", {})
    features: list[str] = list(record.get("features") or [])
    formula: str = record.get("formula") or ""

    parent_id = record["alpha_id"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    new_alpha_id = f"{parent_id}_mut_{timestamp}"

    turnover = metrics.get("turnover", 0)
    sharpe = metrics.get("Sharpe", 0)
    icir = metrics.get("ICIR", 0)

    has_quality = any(f in features for f in ("EBITDA_MARGIN", "EBITDA_LTM", "EPS_LTM"))
    has_momentum = any("MOM" in f for f in features)

    if turnover > 300:
        new_formula = formula
        new_features = features
        new_rebalance = "quarterly"
        mutation_desc = "Switched rebalance frequency to quarterly to reduce turnover."
        hypothesis = record.get("hypothesis", "") + " (quarterly rebalance)"
    elif sharpe < 0.3 and not has_quality:
        new_features = features + ["EBITDA_MARGIN"]
        new_formula = f"({formula}) + 0.3 * rank(EBITDA_MARGIN)"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Added EBITDA_MARGIN quality overlay to strengthen weak signal."
        hypothesis = "Adding profitability overlay to improve signal strength."
    elif icir < 0.2:
        new_features = features + ["VOL_20D"]
        new_formula = f"({formula}) * (1 - rank(VOL_20D))"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Added low-volatility damper (1 - rank(VOL_20D)) to reduce noise."
        hypothesis = "Filtering out high-volatility stocks to improve IC consistency."
    elif has_momentum and not has_quality:
        new_features = features + ["EBITDA_MARGIN"]
        new_formula = f"({formula}) + 0.3 * rank(EBITDA_MARGIN)"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Added quality overlay (EBITDA_MARGIN) to complement momentum."
        hypothesis = "Quality + momentum combination to improve signal stability."
    else:
        new_features = features + ["LIQUIDITY"]
        new_formula = f"({formula}) * rank(LIQUIDITY)"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Added liquidity screen to concentrate in tradeable stocks."
        hypothesis = "Restricting universe to liquid stocks to reduce noise."

    return AlphaConfig(
        alpha_id=new_alpha_id,
        parent_id=parent_id,
        hypothesis=hypothesis,
        formula=new_formula,
        features=list(set(new_features)),
        mutation=mutation_desc,
        universe=config.get("universe", "sp500"),
        start_date=config.get("start_date", "2018-01-01"),
        end_date=config.get("end_date", "2024-12-31"),
        neutralization=config.get("neutralization", "sector"),
        rebalance=new_rebalance,
        transaction_cost_bps=config.get("transaction_cost_bps", 5),
        holding_period_days=config.get("holding_period_days", 20),
    )


# ---------------------------------------------------------------------------
# Shared JSON helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | list:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)
