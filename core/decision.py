"""Rule-based decision logic — tiered pass / revise / fail system."""

from __future__ import annotations

from core.types import BacktestMetrics, RobustnessResult, Verdict

# ---------------------------------------------------------------------------
# Tier 1 — Predictive Power
# Hard fail: signal is dead.  Soft fail: signal is weak but alive.
# ---------------------------------------------------------------------------
IC_MEAN_HARD = 0.0     # no positive edge at all
IC_MEAN_SOFT = 0.02    # real but weak
ICIR_HARD    = 0.0     # net-negative IC across periods
ICIR_SOFT    = 0.30    # IC too inconsistent
MONO_HARD    = -0.2    # quintile ordering meaningfully reversed
MONO_SOFT    = 0.3     # quintile staircase too shallow

# ---------------------------------------------------------------------------
# Tier 2 — Implementation
# Hard fail: unworkable.  Soft fail: tradeable with refinement.
# ---------------------------------------------------------------------------
SHARPE_HARD   = 0.0    # negative Sharpe
SHARPE_SOFT   = 0.50   # below viable threshold
TURNOVER_MAX  = 0.70   # constituent replacement rate (fraction 0–1)
DRAWDOWN_HARD = -0.40  # catastrophic drawdown
DRAWDOWN_SOFT = -0.25  # excessive but survivable drawdown


def check_tier1(metrics: BacktestMetrics) -> tuple[Verdict | None, str | None]:
    """Check Tier 1 (predictive power) before robustness is computed.

    Returns:
        ("failed", reason)  — hard fail, skip robustness entirely
        ("revise", reason)  — weak signal, still proceed to robustness
        (None, None)        — passed cleanly
    """
    hard = []
    if metrics["IC_mean"] <= IC_MEAN_HARD:
        hard.append(f"IC_mean {metrics['IC_mean']:.4f} ≤ {IC_MEAN_HARD} (no positive edge)")
    if metrics["ICIR"] <= ICIR_HARD:
        hard.append(f"ICIR {metrics['ICIR']:.4f} ≤ {ICIR_HARD} (net-negative IC)")
    if metrics["monotonicity"] <= MONO_HARD:
        hard.append(f"monotonicity {metrics['monotonicity']:.4f} ≤ {MONO_HARD} (quintile ordering reversed)")
    if hard:
        return "failed", "Tier 1 — no predictive power: " + "; ".join(hard)

    soft = []
    if metrics["IC_mean"] <= IC_MEAN_SOFT:
        soft.append(f"IC_mean {metrics['IC_mean']:.4f} ≤ {IC_MEAN_SOFT} (weak)")
    if metrics["ICIR"] <= ICIR_SOFT:
        soft.append(f"ICIR {metrics['ICIR']:.4f} ≤ {ICIR_SOFT} (inconsistent)")
    if metrics["monotonicity"] <= MONO_SOFT:
        soft.append(f"monotonicity {metrics['monotonicity']:.4f} ≤ {MONO_SOFT} (shallow staircase)")
    if soft:
        return "revise", "Tier 1 — weak predictive power: " + "; ".join(soft)

    return None, None


def decide(
    metrics: BacktestMetrics,
    robustness: RobustnessResult,
) -> tuple[Verdict, str | None]:
    """Check Tier 2 (implementation). Called only after Tier 1 has been checked.

    Returns "failed" / "revise" / "promising".
    Tier 3 diagnostics live in robustness and metrics — no verdict logic here.
    """
    hard = []
    if metrics["Sharpe"] <= SHARPE_HARD:
        hard.append(f"Sharpe {metrics['Sharpe']:.4f} ≤ {SHARPE_HARD} (negative)")
    if metrics["max_drawdown"] <= DRAWDOWN_HARD:
        hard.append(f"max_drawdown {metrics['max_drawdown']:.4f} ≤ {DRAWDOWN_HARD} (catastrophic)")
    if hard:
        return "failed", "Tier 2 — unworkable: " + "; ".join(hard)

    soft = []
    if metrics["Sharpe"] <= SHARPE_SOFT:
        soft.append(f"Sharpe {metrics['Sharpe']:.4f} ≤ {SHARPE_SOFT}")
    if metrics["turnover"] >= TURNOVER_MAX:
        soft.append(f"turnover {metrics['turnover']:.4f} ≥ {TURNOVER_MAX} (high churn)")
    if metrics["max_drawdown"] <= DRAWDOWN_SOFT:
        soft.append(f"max_drawdown {metrics['max_drawdown']:.4f} ≤ {DRAWDOWN_SOFT}")
    if soft:
        return "revise", "Tier 2 — implementation issues: " + "; ".join(soft)

    return "promising", None
