"""Decision logic — V4 continuous composite scoring with derived verdicts.

The V3 hard/soft threshold gates (check_tier1 / decide) are kept below for
reference and for the constants that other modules import, but the pipeline
now uses score_alpha().
"""

from __future__ import annotations

import numpy as np

from core.formula_validator import formula_complexity
from core.types import (
    AlphaScore, BacktestMetrics, DirectionStatus, RobustnessResult, SubScores,
    Verdict,
)

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


# ---------------------------------------------------------------------------
# V4 composite scoring
# ---------------------------------------------------------------------------

# Sub-score weights (sum to 1.0)
W_PERFORMANCE    = 0.65
W_IMPLEMENTATION = 0.20
W_SIMPLICITY     = 0.10
W_NOVELTY        = 0.05

# Verdict bands applied to the directional total
PROMISING_MIN = 65.0
REVISE_MIN    = 35.0

# Catastrophic hard gates — only truly fatal cases
FATAL_IC_ABS     = 0.005  # dead signal: no edge in either direction
FATAL_SHARPE_ABS = 0.10
FATAL_DRAWDOWN   = -0.40

# Structural (AST) similarity at or above this aborts before the backtest
# (enforced by the caller). Novelty uses the softer combined similarity.
HARD_DUPLICATE_THRESHOLD = 0.95

# Simplicity ramp anchors (formula_complexity units)
_COMPLEXITY_SIMPLE  = 4    # at or below → simplicity 1.0
_COMPLEXITY_GAMED   = 20   # at or above → simplicity 0.0


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0 at lo → 1 at hi, clipped."""
    if hi == lo:
        return 0.0 if x <= lo else 1.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _ramp2(x: float, lo: float, mid: float, hi: float) -> float:
    """Two-segment linear: 0 at lo, 0.5 at mid, 1 at hi, clipped.

    Anchored so every mid is an old soft threshold and every lo an old hard
    threshold — the score passes through 0.5/0.0 exactly where the V3 cliffs were.
    """
    if x <= mid:
        return 0.5 * _ramp(x, lo, mid)
    return 0.5 + 0.5 * _ramp(x, mid, hi)


def _performance_score(
    ic_mean: float, icir: float, sharpe: float, deflated_sharpe: float, mono: float,
) -> float:
    s_ic     = _ramp2(ic_mean, IC_MEAN_HARD, IC_MEAN_SOFT, 0.05)
    s_icir   = _ramp2(icir, ICIR_HARD, ICIR_SOFT, 1.0)
    s_sharpe = (0.7 * _ramp2(sharpe, SHARPE_HARD, SHARPE_SOFT, 1.5)
                + 0.3 * _ramp2(deflated_sharpe, SHARPE_HARD, SHARPE_SOFT, 1.5))
    s_mono   = _ramp2(mono, MONO_HARD, MONO_SOFT, 0.8)
    return 0.30 * s_ic + 0.25 * s_icir + 0.30 * s_sharpe + 0.15 * s_mono


def _turnover_score(turnover: float) -> float:
    if turnover <= 0.30:
        return 1.0
    if turnover <= TURNOVER_MAX:
        return 1.0 - 0.5 * _ramp(turnover, 0.30, TURNOVER_MAX)
    return 0.5 - 0.5 * _ramp(turnover, TURNOVER_MAX, 0.90)


def _implementation_score(turnover: float, max_drawdown: float) -> float:
    s_dd = _ramp2(max_drawdown, DRAWDOWN_HARD, DRAWDOWN_SOFT, -0.10)
    return 0.5 * _turnover_score(turnover) + 0.5 * s_dd


def _robustness_score(robustness: RobustnessResult | None) -> float:
    if not robustness:
        return 0.0
    s_sub     = float(np.clip(robustness.get("subperiod_stability", 0.0), 0.0, 1.0))
    s_placebo = float(np.clip(robustness.get("placebo_score", 0.0), 0.0, 1.0))
    regimes = robustness.get("market_regime_sharpe") or {}
    if regimes:
        s_regime = float(np.mean([_ramp2(v, -0.5, 0.0, 1.0) for v in regimes.values()]))
    else:
        s_regime = 0.5  # data unavailable (yfinance) — neutral, don't punish
    return (s_sub + s_placebo + s_regime) / 3.0


def _composite(sub: SubScores) -> float:
    # Robustness is reported in sub_scores as a diagnostic but excluded
    # from the composite.
    return 100.0 * (
        W_PERFORMANCE    * sub["performance"]
        + W_IMPLEMENTATION * sub["implementation"]
        + W_SIMPLICITY     * sub["simplicity"]
        + W_NOVELTY        * sub["novelty"]
    )


def classify_direction(metrics: BacktestMetrics) -> DirectionStatus:
    """Direction of the evidence relative to the stated hypothesis.

    Metrics-only so legacy records (saved before direction_status was stored)
    can be classified lazily.
    """
    ic, sharpe = metrics["IC_mean"], metrics["Sharpe"]
    if abs(ic) < FATAL_IC_ABS and abs(sharpe) < FATAL_SHARPE_ABS:
        return "uncertain"      # dead — no evidence either way
    if ic > 0 and sharpe > 0:
        return "supported"
    if ic < 0 and sharpe < 0:
        return "contradicted"
    return "uncertain"          # mixed signs


def score_alpha(
    metrics: BacktestMetrics,
    robustness: RobustnessResult | None,
    formula: str,
    similarity_score: float,
) -> AlphaScore:
    """Two composite scores in [0, 100] plus a direction flag.

    `total` uses the raw signed metrics — the hypothesis exactly as stated —
    and drives verdict bands and diagnostics. `predictive_magnitude` uses
    abs() metrics — "a signal exists here", direction-blind — and drives
    parent eligibility and bandit rewards. A contradicted direction is
    recorded in failure_reason and direction_status, never as its own verdict,
    and never worth a sign-flip re-run (the mirrored backtest carries no new
    information).
    """
    ic_mean  = metrics["IC_mean"]
    icir     = metrics["ICIR"]
    sharpe   = metrics["Sharpe"]
    deflated = metrics["deflated_sharpe"]
    mono     = metrics["monotonicity"]
    turnover = metrics["turnover"]
    max_dd   = metrics["max_drawdown"]

    s_robust  = _robustness_score(robustness)
    s_simple  = 1.0 - _ramp(formula_complexity(formula), _COMPLEXITY_SIMPLE, _COMPLEXITY_GAMED)
    s_novelty = float(np.clip(1.0 - similarity_score, 0.0, 1.0))

    sub_scores = SubScores(
        performance=_performance_score(ic_mean, icir, sharpe, deflated, mono),
        implementation=_implementation_score(turnover, max_dd),
        robustness=s_robust,
        simplicity=s_simple,
        novelty=s_novelty,
    )
    total = _composite(sub_scores)

    # Magnitude composite: performance on abs() metrics, all other sub-scores
    # shared. Drawdown stays as-stated — the mirrored drawdown path is not
    # reconstructable from stored metrics.
    magnitude_sub = SubScores(
        performance=_performance_score(
            abs(ic_mean), abs(icir), abs(sharpe), abs(deflated), abs(mono)
        ),
        implementation=sub_scores["implementation"],
        robustness=s_robust,
        simplicity=s_simple,
        novelty=s_novelty,
    )
    predictive_magnitude = _composite(magnitude_sub)

    direction_status = classify_direction(metrics)
    direction_note: str | None = None
    if direction_status == "contradicted":
        direction_note = (
            f"direction contradicted (IC_mean {ic_mean:.4f}, Sharpe {sharpe:.2f}; "
            f"magnitude score {predictive_magnitude:.1f} vs directional {total:.1f}) — "
            "the mirrored signal is already measured; do not re-test a sign flip"
        )

    # --- Fatal gates ----------------------------------------------------------
    fatal_reasons: list[str] = []
    if abs(ic_mean) < FATAL_IC_ABS and abs(sharpe) < FATAL_SHARPE_ABS:
        fatal_reasons.append(
            f"dead signal: |IC_mean| {abs(ic_mean):.4f} < {FATAL_IC_ABS} and "
            f"|Sharpe| {abs(sharpe):.4f} < {FATAL_SHARPE_ABS} (no edge in either direction)"
        )
    if max_dd <= FATAL_DRAWDOWN:
        fatal_reasons.append(
            f"max_drawdown {max_dd:.4f} ≤ {FATAL_DRAWDOWN}"
        )

    if fatal_reasons:
        failure_reason = "fatal — " + "; ".join(fatal_reasons)
        if direction_note:
            failure_reason += f"; {direction_note}"
        return AlphaScore(
            total=round(total, 2),
            predictive_magnitude=round(predictive_magnitude, 2),
            direction_status=direction_status,
            sub_scores=sub_scores,
            verdict="failed",
            failure_reason=failure_reason,
            fatal=True,
        )

    # --- Verdict bands on the directional total --------------------------------
    if total >= PROMISING_MIN:
        verdict: Verdict = "promising"
        failure_reason = None
    else:
        verdict = "revise" if total >= REVISE_MIN else "failed"
        # robustness is diagnostic-only — don't blame it for a low composite
        scored = {k: v for k, v in sub_scores.items() if k != "robustness"}
        weakest = min(scored, key=scored.__getitem__)
        detail = ""
        if weakest == "simplicity":
            detail = f", formula complexity {formula_complexity(formula)}"
        elif weakest == "novelty":
            detail = f", similarity {similarity_score:.2f}"
        failure_reason = (
            f"score {total:.1f} — weakest component: "
            f"{weakest} ({sub_scores[weakest]:.2f}{detail})"
        )
        if direction_note:
            failure_reason += f"; {direction_note}"

    return AlphaScore(
        total=round(total, 2),
        predictive_magnitude=round(predictive_magnitude, 2),
        direction_status=direction_status,
        sub_scores=sub_scores,
        verdict=verdict,
        failure_reason=failure_reason,
        fatal=False,
    )


# ---------------------------------------------------------------------------
# V3 tiered gates — DEPRECATED: no longer called by the pipeline.
# Kept for the threshold constants (imported elsewhere) and reference.
# ---------------------------------------------------------------------------

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
