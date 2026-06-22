"""LLM-powered reflection generator using DeepSeek API.

Falls back to rule-based output if the API call fails.
Every output is labeled as LLM-generated hypothesis, not validated evidence.
"""

from __future__ import annotations

import os

from core.types import AlphaConfig, BacktestMetrics, RobustnessResult, Verdict

_DISCLAIMER = "[DISCLAIMER: LLM-generated hypothesis, not validated evidence]"


def generate_reflection(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
    verdict: Verdict,
    failure_reason: str | None,
) -> str:
    """Generate a structured reflection using DeepSeek LLM, with rule-based fallback."""
    try:
        return _llm_reflection(alpha, metrics, robustness, verdict, failure_reason)
    except Exception as exc:
        print(f"  [reflection] LLM call failed ({exc}), using rule-based fallback.")
        return _rule_based_reflection(alpha, metrics, robustness, verdict, failure_reason)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _llm_reflection(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
    verdict: Verdict,
    failure_reason: str | None,
) -> str:
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = _build_prompt(alpha, metrics, robustness, verdict, failure_reason)

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


def _build_prompt(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
    verdict: Verdict,
    failure_reason: str | None,
) -> str:
    return f"""You are reviewing a quantitative alpha factor backtest. Provide a structured analysis in exactly this format:

Observation: <1-2 sentences on what the numbers show>
Failure Reason: {failure_reason if failure_reason else "N/A"}
Possible Explanation: <1-2 sentences on why the alpha performed this way>
Next Mutation: <1 concrete, specific change to the formula or config to try next>

Backtest details:
- Formula: {alpha.get("formula")}
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
- Sector Stability: {robustness["sector_stability"]:.4f}
- Subperiod Stability: {robustness["subperiod_stability"]:.4f}
- Market Regime Sharpe: {robustness["market_regime_sharpe"]:.4f}
- Placebo Score: {robustness["placebo_score"]:.4f}
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
    if robustness["sector_stability"] == 0.0:
        return (
            "Zero sector stability suggests the signal only works in a subset of sectors "
            "and lacks the breadth needed for a robust factor."
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
    if verdict == "inconclusive":
        return (
            "Add a liquidity filter: multiply signal by rank(LIQUIDITY) "
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
