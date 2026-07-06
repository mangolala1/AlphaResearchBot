"""Decision logic — V4 continuous composite scoring with derived verdicts.

The V3 hard/soft threshold gates (check_tier1 / decide) are kept below for
reference and for the constants that other modules import, but the pipeline
now uses score_alpha().
"""

from __future__ import annotations

import numpy as np

from core.formula_validator import formula_complexity
from core.types import (
    AlphaScore, BacktestMetrics, RobustnessResult, SubScores, Verdict,
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
W_PERFORMANCE    = 0.45
W_ROBUSTNESS     = 0.20
W_IMPLEMENTATION = 0.15
W_SIMPLICITY     = 0.10
W_NOVELTY        = 0.10

# Verdict bands applied to signal_strength
PROMISING_MIN = 65.0
REVISE_MIN    = 35.0

# Catastrophic hard gates — only truly fatal cases
FATAL_IC_ABS     = 0.005  # dead signal: no edge in either direction
FATAL_SHARPE_ABS = 0.10
FATAL_DRAWDOWN   = -0.40  # in the preferred direction

# Direction handling: inverting a wrong-direction hypothesis is not free
INVERSION_PENALTY = 5.0   # score points

# Similarity above this aborts before the backtest (enforced by the caller)
HARD_DUPLICATE_THRESHOLD = 0.90

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
    return 100.0 * (
        W_PERFORMANCE    * sub["performance"]
        + W_ROBUSTNESS     * sub["robustness"]
        + W_IMPLEMENTATION * sub["implementation"]
        + W_SIMPLICITY     * sub["simplicity"]
        + W_NOVELTY        * sub["novelty"]
    )


def score_alpha(
    metrics: BacktestMetrics,
    robustness: RobustnessResult | None,
    formula: str,
    similarity_score: float,
    portfolio_returns: list[float] | None = None,
) -> AlphaScore:
    """Direction-aware composite score in [0, 100].

    Separates "a predictive signal exists" from "the hypothesis direction is
    correct": `total` keeps the sign (hypothesis evaluation), while
    `signal_strength = max(total, inverted_total - INVERSION_PENALTY)` lets a
    real-but-inverted alpha survive into the parent pool with verdict
    "revise_invert" instead of "failed".

    `portfolio_returns` (per-period long-short returns from the backtest)
    enables exact inverted-direction drawdown/deflated-Sharpe; when None
    (e.g. lazy rescore of pre-V4 rows) only the directional score is computed.
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

    # --- Directional score: the hypothesis exactly as stated -----------------
    sub_scores = SubScores(
        performance=_performance_score(ic_mean, icir, sharpe, deflated, mono),
        implementation=_implementation_score(turnover, max_dd),
        robustness=s_robust,
        simplicity=s_simple,
        novelty=s_novelty,
    )
    total = _composite(sub_scores)

    # --- Inverted score: same signal traded with the opposite sign -----------
    inverted_total: float | None = None
    inv_max_dd = max_dd
    if portfolio_returns is not None and len(portfolio_returns) >= 4:
        from core.backtest import _deflated_sharpe, _max_drawdown  # local: avoid heavy import at module load

        inv_rets = -np.asarray(portfolio_returns, dtype=float)
        inv_sharpe = -sharpe  # mean negates, std unchanged
        inv_deflated = _deflated_sharpe(inv_rets, inv_sharpe)
        inv_max_dd = _max_drawdown(inv_rets)  # drawdown paths do NOT simply negate
        inv_sub = SubScores(
            performance=_performance_score(-ic_mean, -icir, inv_sharpe, inv_deflated, -mono),
            implementation=_implementation_score(turnover, inv_max_dd),
            robustness=s_robust,   # stability/placebo treated as direction-agnostic
            simplicity=s_simple,
            novelty=s_novelty,
        )
        inverted_total = _composite(inv_sub)

    if inverted_total is not None and inverted_total - INVERSION_PENALTY > total:
        signal_strength = inverted_total - INVERSION_PENALTY
        preferred_direction = -1
        preferred_max_dd = inv_max_dd
    else:
        signal_strength = total
        preferred_direction = 1
        preferred_max_dd = max_dd

    # --- Fatal gates ----------------------------------------------------------
    fatal_reasons: list[str] = []
    if abs(ic_mean) < FATAL_IC_ABS and abs(sharpe) < FATAL_SHARPE_ABS:
        fatal_reasons.append(
            f"dead signal: |IC_mean| {abs(ic_mean):.4f} < {FATAL_IC_ABS} and "
            f"|Sharpe| {abs(sharpe):.4f} < {FATAL_SHARPE_ABS} (no edge in either direction)"
        )
    if preferred_max_dd <= FATAL_DRAWDOWN:
        fatal_reasons.append(
            f"max_drawdown {preferred_max_dd:.4f} ≤ {FATAL_DRAWDOWN} in the preferred direction"
        )

    if fatal_reasons:
        signal_strength = min(signal_strength, 25.0)
        return AlphaScore(
            total=round(total, 2),
            signal_strength=round(signal_strength, 2),
            preferred_direction=preferred_direction,
            sub_scores=sub_scores,
            verdict="failed",
            failure_reason="fatal — " + "; ".join(fatal_reasons),
            fatal=True,
        )

    # --- Verdict bands on signal_strength --------------------------------------
    if signal_strength >= PROMISING_MIN:
        band: Verdict = "promising"
    elif signal_strength >= REVISE_MIN:
        band = "revise"
    else:
        band = "failed"

    if band != "failed" and preferred_direction == -1:
        verdict: Verdict = "revise_invert"
        failure_reason = (
            f"signal is real but direction is inverted "
            f"(directional {total:.1f}, inverted {inverted_total:.1f}) — "
            "hypothesis direction was wrong; flip the formula sign and restate the hypothesis"
        )
    elif band == "promising":
        verdict, failure_reason = "promising", None
    else:
        verdict = band
        weakest = min(sub_scores, key=sub_scores.__getitem__)
        detail = ""
        if weakest == "simplicity":
            detail = f", formula complexity {formula_complexity(formula)}"
        elif weakest == "novelty":
            detail = f", similarity {similarity_score:.2f}"
        failure_reason = (
            f"score {signal_strength:.1f} — weakest component: "
            f"{weakest} ({sub_scores[weakest]:.2f}{detail})"
        )

    return AlphaScore(
        total=round(total, 2),
        signal_strength=round(signal_strength, 2),
        preferred_direction=preferred_direction,
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
