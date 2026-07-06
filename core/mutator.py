"""Alpha mutation generator — uses DeepSeek to produce a child AlphaConfig from a parent.

Falls back to rule-based mutation if the LLM call fails or returns invalid JSON.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from core.types import AlphaConfig
from core.formula_validator import ALLOWED_FUNCTION_NAMES, FORMULA_CONSTRAINT, validate_alpha

_SYSTEM_PROMPT = (
    "You are a quantitative research analyst designing alpha factors. "
    "You will be given a parent alpha experiment that failed or underperformed. "
    "Your job is to propose one specific mutation that addresses the failure mode. "
    "Be precise and concrete. Only suggest formulas using supported operators and raw data columns."
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
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    last_error = None
    for attempt in range(3):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        try:
            child = _parse_and_validate(raw, record)
            return child
        except ValueError as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your formula failed validation: {exc}\n"
                    "Fix ONLY the formula field and return the corrected JSON. "
                    "Use only the exact operator names from the list provided earlier."
                ),
            })

    raise ValueError(f"LLM failed after 3 attempts: {last_error}")


def _score_lines(record: dict) -> str:
    """Composite-score block for the mutation prompt (empty for pre-V4 parents)."""
    score = record.get("score")
    if score is None:
        return ""
    sub = record.get("sub_scores") or {}
    lines = [
        f"- Composite Score (directional): {score:.1f} / 100",
        f"- Signal Strength: {record.get('signal_strength', score):.1f} / 100",
    ]
    if sub:
        lines.append(
            "- Sub-scores: " + " | ".join(f"{k} {v:.2f}" for k, v in sub.items())
        )
    instructions = []
    if record.get("preferred_direction") == -1:
        instructions.append(
            "The signal direction is INVERTED — the primary mutation is to negate the "
            "formula (multiply by -1) AND restate the hypothesis accordingly; do not add complexity."
        )
    if sub.get("simplicity", 1.0) < 0.5:
        instructions.append(
            "Simplicity is below 0.5 — the mutation MUST reduce formula complexity "
            "(fewer operators/features), not add terms."
        )
    if sub.get("novelty", 1.0) < 0.4:
        instructions.append(
            "Novelty is low — the mutation must change the signal source, not just re-weight terms."
        )
    if instructions:
        lines.append("IMPORTANT: " + " ".join(instructions))
    return "\n".join(lines) + "\n"


def _build_mutation_prompt(record: dict) -> str:
    metrics = record.get("metrics", {})
    robustness = record.get("robustness", {})
    config = record.get("config", {})

    return f"""A parent alpha experiment has completed. Propose ONE mutation to improve it.

Parent Alpha:
- alpha_id: {record["alpha_id"]}
- Formula: {config.get("formula", "N/A")}
- Universe: {config.get("universe")} | Rebalance: {config.get("rebalance")} | Neutralization: {config.get("neutralization")}
- Period: {config.get("start_date")} to {config.get("end_date")}

Results:
- Verdict: {record["verdict"].upper()}
- Failure Reason: {record.get("failure_reason") or "N/A"}
{_score_lines(record)}- IC_mean: {metrics.get("IC_mean", 0):.4f}
- ICIR: {metrics.get("ICIR", 0):.4f}
- Sharpe: {metrics.get("Sharpe", 0):.4f}
- Turnover: {metrics.get("turnover", 0):.4f}
- Subperiod Stability: {robustness.get("subperiod_stability", 0):.4f}

Prior Reflection (for context only — do NOT copy function names from it verbatim):
{record.get("reflection", "N/A")}

{FORMULA_CONSTRAINT}
CRITICAL: Use ONLY the operator names listed above. Any operator name in the reflection that is not in the list above is incorrect — do not use it.

Return ONLY a JSON object (no markdown, no explanation) with these fields:
{{
  "hypothesis": "one sentence investment thesis",
  "formula": "execution formula using raw column DataFrames",
  "mutation": "one sentence describing what changed vs parent",
  "universe": "{config.get("universe", "sp500")}",
  "start_date": "{config.get("start_date", "2021-01-01")}",
  "end_date": "{config.get("end_date", "2026-06-01")}",
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
        raise ValueError(f"Generated alpha failed validation: {result.errors}")

    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_based_mutation(record: dict) -> AlphaConfig:
    config = record.get("config", {})
    metrics = record.get("metrics", {})
    parent_formula: str = config.get("formula", "rank(ADJUSTED_PRICE.pct_change().rolling(20).std()) * -1")

    parent_id = record["alpha_id"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    new_alpha_id = f"{parent_id}_mut_{timestamp}"

    turnover = metrics.get("turnover", 0)
    sharpe = metrics.get("Sharpe", 0)
    icir = metrics.get("ICIR", 0)

    has_quality = "OPERATING_INCOME_LTM" in parent_formula or "NET_INCOME_LTM" in parent_formula
    has_momentum = "shift" in parent_formula and "ADJUSTED_PRICE" in parent_formula

    if record.get("preferred_direction") == -1:
        new_formula = f"-1 * ({parent_formula})"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Negated the formula — signal was real but the hypothesis direction was inverted."
        hypothesis = f"Inverted hypothesis: the opposite of '{record.get('hypothesis', '')}' holds."

    elif turnover > 0.7:
        new_formula = parent_formula
        new_rebalance = "quarterly"
        mutation_desc = "Switched rebalance frequency to quarterly to reduce turnover."
        hypothesis = record.get("hypothesis", "") + " (quarterly rebalance)"

    elif sharpe < 0.3 and not has_quality:
        quality_raw = "rank((OPERATING_INCOME_LTM + DA_LTM) / REVENUE_LTM)"
        new_formula = f"({parent_formula}) + 0.3 * {quality_raw}"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Added EBITDA_MARGIN quality overlay to strengthen weak signal."
        hypothesis = "Adding profitability overlay to improve signal strength."

    elif icir < 0.2:
        vol_raw = "rank(ADJUSTED_PRICE.pct_change().rolling(20).std())"
        new_formula = f"({parent_formula}) * (1 - {vol_raw})"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Added low-volatility damper to reduce noise."
        hypothesis = "Filtering out high-volatility stocks to improve IC consistency."

    elif has_momentum and not has_quality:
        quality_raw = "rank((OPERATING_INCOME_LTM + DA_LTM) / REVENUE_LTM)"
        new_formula = f"({parent_formula}) + 0.3 * {quality_raw}"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Added quality overlay to complement momentum."
        hypothesis = "Quality + momentum combination to improve signal stability."

    else:
        liquidity_raw = "rank(ADJUSTED_VOLUME * ADJUSTED_PRICE)"
        new_formula = f"({parent_formula}) * {liquidity_raw}"
        new_rebalance = config.get("rebalance", "monthly")
        mutation_desc = "Added liquidity screen to concentrate in tradeable stocks."
        hypothesis = "Restricting universe to liquid stocks to reduce noise."

    return AlphaConfig(
        alpha_id=new_alpha_id,
        parent_id=parent_id,
        hypothesis=hypothesis,
        formula=new_formula,
        mutation=mutation_desc,
        universe=config.get("universe", "sp500"),
        start_date=config.get("start_date", "2021-01-01"),
        end_date=config.get("end_date", "2026-06-01"),
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
