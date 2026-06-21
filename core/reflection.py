"""Rule-based mock reflection generator.

Every output is labeled as LLM-generated hypothesis, not validated evidence.
"""

from __future__ import annotations

from core.types import AlphaConfig, BacktestMetrics, Verdict

_DISCLAIMER = (
    "[DISCLAIMER: LLM-generated hypothesis, not validated evidence]"
)


def generate_reflection(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    verdict: Verdict,
    failure_reason: str | None,
) -> str:
    """Generate a structured reflection string for an experiment."""
    features = alpha.get("features", [])
    formula = alpha.get("formula", "")

    observation = _observation(metrics, verdict)
    possible_reason = _possible_reason(features, formula, metrics)
    failure_line = f"Failure Reason: {failure_reason}" if failure_reason else "Failure Reason: N/A"
    next_mutation = _next_mutation(alpha, metrics, verdict)

    return (
        f"{_DISCLAIMER}\n\n"
        f"Observation: {observation}\n"
        f"Possible Reason: {possible_reason}\n"
        f"{failure_line}\n"
        f"Next Mutation Suggestion: {next_mutation}"
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


def _possible_reason(
    features: list[str], formula: str, metrics: BacktestMetrics
) -> str:
    has_momentum = any("MOM" in f for f in features)
    has_quality = any(f in features for f in ("EBITDA_MARGIN", "EBITDA_LTM", "EPS_LTM"))
    has_growth = any("GROWTH" in f for f in features)

    if metrics["turnover"] > 300:
        return (
            "High rebalancing frequency combined with momentum features may be generating "
            "excessive churn. Consider extending the holding period or dampening the signal."
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
            "Reduce turnover by adding a signal smoothing layer: "
            "replace rank() with ts_mean(rank(), 3) to average signal over 3 periods."
        )
    if verdict == "failed" and metrics["Sharpe"] < 0.3:
        return (
            "Signal is too weak. Try adding a value dimension: "
            "incorporate SALES_LTM / ADJUSTED_PRICE as a price-to-sales anchor."
        )
    if verdict == "failed" and metrics["ICIR"] < 0.2:
        return (
            "Low ICIR indicates inconsistent signal. "
            "Try sector-neutralizing more aggressively or adding VOL_20D as a risk screen."
        )
    if verdict == "inconclusive":
        return (
            "Apply a volatility filter: multiply signal by sign(1 - VOL_20D / ts_mean(VOL_20D, 60)) "
            "to avoid trading in high-volatility regimes where noise dominates."
        )
    if has_momentum and not has_quality:
        return "Add a quality overlay: combine with rank(EBITDA_MARGIN) weighted at 0.3."
    if has_quality and not has_momentum:
        return "Add a momentum overlay: combine with rank(MOM6_1) weighted at 0.3."
    return (
        "Experiment with alternative neutralization: switch from sector to industry "
        "neutralization to capture finer cross-sectional variation."
    )
