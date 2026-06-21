"""Rule-based decision logic — maps metrics to a verdict."""

from __future__ import annotations

from core.types import BacktestMetrics, RobustnessResult, Verdict

SHARPE_THRESHOLD = 0.5
ICIR_THRESHOLD = 0.3
TURNOVER_THRESHOLD = 300.0


def decide(
    metrics: BacktestMetrics, robustness: RobustnessResult
) -> tuple[Verdict, str | None]:
    """Apply decision rules and return (verdict, failure_reason).

    failure_reason is None when verdict is 'promising'.
    """
    if metrics["turnover"] > TURNOVER_THRESHOLD:
        reason = (
            f"Excessive turnover: {metrics['turnover']:.1f} bps/yr "
            f"exceeds threshold of {TURNOVER_THRESHOLD:.0f}"
        )
        return "failed", reason

    if metrics["Sharpe"] < SHARPE_THRESHOLD or metrics["ICIR"] < ICIR_THRESHOLD:
        reason = (
            f"Sharpe {metrics['Sharpe']:.2f} or ICIR {metrics['ICIR']:.2f} "
            f"below minimum thresholds ({SHARPE_THRESHOLD} / {ICIR_THRESHOLD})"
        )
        return "failed", reason

    if metrics["noise_risk"] == "high":
        reason = (
            f"High noise risk: deflated Sharpe {metrics['deflated_sharpe']:.2f} "
            f"is significantly below raw Sharpe {metrics['Sharpe']:.2f}"
        )
        return "inconclusive", reason

    return "promising", None
