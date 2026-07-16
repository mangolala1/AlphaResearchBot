"""LLM-powered reflection generator using DeepSeek API.

Falls back to rule-based output if the API call fails.
Every output is labeled as LLM-generated hypothesis, not validated evidence.
"""

from __future__ import annotations

import os

from core.formula_validator import FORMULA_CONSTRAINT
from core.types import AlphaConfig, AlphaScore, BacktestMetrics, RobustnessResult, Verdict

_DISCLAIMER = "[DISCLAIMER: LLM-generated hypothesis, not validated evidence]"


def generate_reflection(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
    verdict: Verdict,
    failure_reason: str | None,
    alpha_score: AlphaScore | None = None,
) -> str:
    """Generate a structured reflection using DeepSeek LLM, with rule-based fallback."""
    try:
        return _llm_reflection(alpha, metrics, robustness, verdict, failure_reason, alpha_score)
    except Exception as exc:
        print(f"  [reflection] LLM call failed ({exc}), using rule-based fallback.")
        return _rule_based_reflection(alpha, metrics, robustness, verdict, failure_reason, alpha_score)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _llm_reflection(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
    verdict: Verdict,
    failure_reason: str | None,
    alpha_score: AlphaScore | None = None,
) -> str:
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = _build_prompt(alpha, metrics, robustness, verdict, failure_reason, alpha_score)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a quantitative research analyst reviewing alpha factor backtests. "
                    "Be concise and specific. Never make claims about future performance. "
                    "Always frame observations as hypotheses to be tested further."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=400,
    )

    content = response.choices[0].message.content.strip()
    return f"{_DISCLAIMER}\n\n{content}"


def _score_block(alpha_score: AlphaScore | None) -> str:
    if alpha_score is None:
        return ""
    sub = alpha_score.sub_scores
    return f"""
Composite Score:
- Score (hypothesis as stated): {alpha_score.total:.1f} / 100
- Predictive magnitude (direction-blind): {alpha_score.predictive_magnitude:.1f} / 100
- Direction status: {alpha_score.direction_status}
- Sub-scores: performance {sub['performance']:.2f} | implementation {sub['implementation']:.2f} | robustness {sub['robustness']:.2f} | simplicity {sub['simplicity']:.2f} | novelty {sub['novelty']:.2f}
The score rewards simplicity and novelty; sub-scores below 0.4 are the priority to fix.
"""


def _build_prompt(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
    verdict: Verdict,
    failure_reason: str | None,
    alpha_score: AlphaScore | None = None,
) -> str:
    return f"""You are reviewing a quantitative alpha factor backtest. Provide a structured analysis in exactly this format:

Observation: <1-2 sentences on what the numbers show>
Failure Reason: {failure_reason if failure_reason else "N/A"}
Possible Explanation: <1-2 sentences on why the alpha performed this way>
Next Mutation: <1 concrete, specific change to the formula or config to try next>

IMPORTANT: NEVER suggest a sign-flip-only mutation (multiplying the formula by -1). The backtest already measures both directions; a sign flip adds no information and will be rejected as a near-duplicate. If the direction was contradicted, restate the economic hypothesis in the opposite direction in 'Possible Explanation', but the Next Mutation must be a structurally different formula.
{_score_block(alpha_score)}
Backtest details:
- Formula: {alpha.get("formula")}  [execution]
- Features: {alpha.get("features")}
- Universe: {alpha.get("universe")} | Rebalance: {alpha.get("rebalance")} | Neutralization: {alpha.get("neutralization")}
- Period: {alpha.get("start_date")} to {alpha.get("end_date")}

Results:
- Verdict: {verdict.upper()}
- IC_mean: {metrics["IC_mean"]:.4f}
- ICIR: {metrics["ICIR"]:.4f}
- Sharpe: {metrics["Sharpe"]:.4f}
- Deflated Sharpe: {metrics["deflated_sharpe"]:.4f}
- Turnover: {metrics["turnover"]:.1f} bps/yr
- Max Drawdown: {metrics["max_drawdown"]:.4f}
- Noise Risk: {metrics["noise_risk"]}

Robustness:
- Sector IC (mean per sector): {", ".join(f"{s}: {v:.3f}" for s, v in robustness["sector_stability"].items()) or "N/A"}
- Subperiod Stability: {robustness["subperiod_stability"]:.4f}
- Market Regime Sharpe: {", ".join(f"{r}: {v:.3f}" for r, v in robustness["market_regime_sharpe"].items()) or "N/A"}
- Placebo Score: {robustness["placebo_score"]:.4f}

{FORMULA_CONSTRAINT}
"""


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_based_reflection(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
    verdict: Verdict,
    failure_reason: str | None,
    alpha_score: AlphaScore | None = None,
) -> str:
    features = alpha.get("features", [])
    formula = alpha.get("formula", "")

    observation = _observation(metrics, verdict)
    possible_explanation = _possible_explanation(features, formula, metrics, robustness)
    failure_line = f"Failure Reason: {failure_reason}" if failure_reason else "Failure Reason: N/A"
    next_mutation = _next_mutation(alpha, metrics, verdict)

    return (
        f"{_DISCLAIMER}\n\n"
        f"Observation: {observation}\n"
        f"{failure_line}\n"
        f"Possible Explanation: {possible_explanation}\n"
        f"Next Mutation: {next_mutation}"
    )


def _observation(metrics: BacktestMetrics, verdict: Verdict) -> str:
    sharpe = metrics["Sharpe"]
    icir = metrics["ICIR"]
    turnover = metrics["turnover"]

    if verdict == "promising":
        return (
            f"Alpha shows a promising Sharpe of {sharpe:.2f} and ICIR of {icir:.2f} "
            f"with manageable turnover of {turnover:.0f} bps/yr."
        )
    elif verdict == "failed":
        if turnover > 300:
            return (
                f"Alpha produced extremely high turnover of {turnover:.0f} bps/yr, "
                f"which would erode returns after transaction costs."
            )
        return (
            f"Alpha underperformed thresholds with Sharpe {sharpe:.2f} "
            f"and ICIR {icir:.2f}."
        )
    else:
        return (
            f"Alpha shows raw Sharpe {sharpe:.2f} but deflated Sharpe "
            f"{metrics['deflated_sharpe']:.2f} suggests significant noise contamination."
        )


def _possible_explanation(
    features: list[str],
    formula: str,
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
) -> str:
    has_momentum = any("MOM" in f for f in features)
    has_quality = any(f in features for f in ("EBITDA_MARGIN", "EBITDA_LTM", "EPS_LTM"))
    has_growth = any("GROWTH" in f for f in features)

    if metrics["turnover"] > 300:
        return (
            "High rebalancing frequency combined with momentum features may be generating "
            "excessive churn. Consider extending the holding period or dampening the signal."
        )
    sector_ics = robustness["sector_stability"]
    if sector_ics and all(v < 0 for v in sector_ics.values()):
        return (
            "Negative IC across all sectors suggests the signal direction may be inverted "
            "or the factor lacks breadth needed for a robust alpha."
        )
    if robustness["subperiod_stability"] == 0.0:
        return (
            "Zero subperiod stability indicates the alpha's performance is not consistent "
            "across time — likely a regime-specific effect."
        )
    if has_momentum and has_quality:
        return (
            "The combination of quality (profitability) and momentum signals may exhibit "
            "regime-dependent correlation, performing well in trending markets but degrading "
            "during mean-reversion episodes."
        )
    if has_growth:
        return (
            "Growth-based features can be noisy near earnings announcements and may "
            "introduce look-ahead bias if NTM estimates are used."
        )
    if metrics["noise_risk"] == "high":
        return (
            "The signal may be over-fit to the in-sample period. "
            "Deflated Sharpe suggests the raw metric is upward-biased."
        )
    return (
        f"The formula '{formula}' captures cross-sectional variation, "
        "but signal persistence may be limited without additional filters."
    )


def _next_mutation(
    alpha: AlphaConfig, metrics: BacktestMetrics, verdict: Verdict
) -> str:
    features = alpha.get("features", [])
    has_momentum = any("MOM" in f for f in features)
    has_quality = any(f in features for f in ("EBITDA_MARGIN", "EBITDA_LTM", "EPS_LTM"))

    if metrics["turnover"] > 300:
        return (
            "Switch rebalance frequency from monthly to quarterly to reduce turnover."
        )
    if verdict == "failed" and metrics["Sharpe"] < 0.3:
        return (
            "Signal is too weak. Try adding a value dimension: "
            "incorporate rank(PRICE_TO_SALES) * -1 as a cheap-stock screen."
        )
    if verdict == "failed" and metrics["ICIR"] < 0.2:
        return (
            "Low ICIR indicates inconsistent signal. "
            "Add rank(VOL_20D) * -1 as a low-volatility screen to filter noisy periods."
        )
    if verdict == "revise" and metrics["noise_risk"] == "high":
        # (was a dead `verdict == "inconclusive"` branch — that value never existed)
        return (
            "Add a liquidity filter: multiply signal by rank(ADJUSTED_VOLUME * ADJUSTED_PRICE) "
            "to avoid trading illiquid stocks where noise dominates."
        )
    if has_momentum and not has_quality:
        return "Add a quality overlay: combine with rank(EBITDA_MARGIN) weighted at 0.3."
    if has_quality and not has_momentum:
        return "Add a momentum overlay: combine with rank(MOM6_1) weighted at 0.3."
    return (
        "Experiment with alternative neutralization: switch from sector to no neutralization "
        "to test whether sector effects are suppressing the signal."
    )
