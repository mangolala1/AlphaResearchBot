"""Single-experiment pipeline — callable from the CLI wrapper and the loop runner.

V4: verdicts come from the composite score (core.decision.score_alpha).
Similarity is a graded novelty penalty; only structural duplicates
(exact sign-canonical match, or AST similarity >= HARD_DUPLICATE_THRESHOLD)
abort before the backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from core.backtest import run_backtest
from core.decision import (
    FATAL_IC_ABS, FATAL_SHARPE_ABS, HARD_DUPLICATE_THRESHOLD, score_alpha,
)
from core.formula_validator import validate_alpha
from core.memory import ExperimentStore
from core.reflection import generate_reflection
from core.robustness import run_robustness
from core.similarity import check_similarity
from core.types import AlphaConfig, ExperimentRecord, SimilarityResult

if TYPE_CHECKING:
    from core.data_loader import DataLoader

_EMPTY_ROBUSTNESS = {
    "sector_stability": {}, "subperiod_stability": 0.0,
    "market_regime_sharpe": {}, "placebo_score": 0.0,
}


@dataclass
class ExperimentOutcome:
    status: Literal["completed", "duplicate", "validation_failed", "backtest_error"]
    record: ExperimentRecord | None
    similarity: SimilarityResult | None
    error: str | None


def run_single_experiment(
    alpha: AlphaConfig,
    store: ExperimentStore,
    loader: "DataLoader | None" = None,
    force: bool = False,
    verbose: bool = True,
) -> ExperimentOutcome:
    """Run the full validate → backtest → score → reflect → save pipeline."""

    def _log(msg: str = "") -> None:
        if verbose:
            print(msg)

    # ── Step 1: validate ─────────────────────────────────────────────────────
    _log("[ Step 1 ] Validating formula...")
    result = validate_alpha(alpha)
    for w in result.warnings:
        _log(f"  WARNING: {w}")
    if not result.valid:
        for e in result.errors:
            _log(f"  ERROR: {e}")
        return ExperimentOutcome(
            status="validation_failed", record=None, similarity=None,
            error="; ".join(result.errors),
        )
    _log("  Validation passed.\n")

    # ── Step 2: similarity — hard abort only at structural duplicates ────────
    _log("[ Step 2 ] Checking similarity against prior alphas...")
    sim = check_similarity(alpha, store, threshold=HARD_DUPLICATE_THRESHOLD)
    if not sim["is_unique"]:
        kind = "exact duplicate (sign-canonical)" if sim["is_exact_duplicate"] else "structural duplicate"
        _log(f"  {kind}: {sim['structural_similarity']:.0%} vs '{sim['most_similar_id']}'")
        if not force:
            _log("  Aborting (use force=True / --force to run anyway).")
            return ExperimentOutcome(
                status="duplicate", record=None, similarity=sim,
                error=(
                    "exact canonical duplicate" if sim["is_exact_duplicate"]
                    else f"structural similarity {sim['structural_similarity']:.2f} >= {HARD_DUPLICATE_THRESHOLD}"
                ),
            )
        _log("  force set, continuing.\n")
    else:
        _log(
            f"  Combined similarity {sim['similarity_score']:.0%} "
            f"(structural max {sim['structural_similarity']:.0%}; novelty flows into the score).\n"
        )

    # ── Step 3: backtest ──────────────────────────────────────────────────────
    _log("[ Step 3 ] Running backtest...")
    if loader is None:
        from core.data_loader import DataLoader as _DL
        loader = _DL()
    try:
        backtest_result = run_backtest(alpha, data_loader=loader)
    except Exception as exc:
        _log(f"  ERROR during backtest: {exc}")
        return ExperimentOutcome(
            status="backtest_error", record=None, similarity=sim, error=str(exc),
        )

    metrics = backtest_result["metrics"]
    _log(f"  IC_mean        : {metrics['IC_mean']:.4f}  (over {len(backtest_result['ic_series'])} periods)")
    _log(f"  ICIR           : {metrics['ICIR']:.4f}")
    _log(f"  Monotonicity   : {metrics['monotonicity']:.4f}")
    _log(f"  Sharpe         : {metrics['Sharpe']:.4f}")
    _log(f"  Deflated Sharpe: {metrics['deflated_sharpe']:.4f}")
    _log(f"  Q5-Q1 Return   : {metrics['Q5_Q1_return']:.4f}")
    _log(f"  Turnover       : {metrics['turnover']:.4f}  (constituent replacement rate)")
    _log(f"  Max Drawdown   : {metrics['max_drawdown']:.4f}")
    _log(f"  Noise Risk     : {metrics['noise_risk']}\n")

    # ── Step 4: robustness (skipped only for dead signals) ──────────────────
    dead_signal = (
        abs(metrics["IC_mean"]) < FATAL_IC_ABS
        and abs(metrics["Sharpe"]) < FATAL_SHARPE_ABS
    )
    if dead_signal:
        _log("[ Step 4 ] Dead signal (no edge in either direction) — skipping robustness.\n")
        robustness = dict(_EMPTY_ROBUSTNESS)
        robustness_for_score = None
    else:
        _log("[ Step 4 ] Running robustness checks...")
        robustness = run_robustness(alpha, backtest_result)
        sector_summary = "  ".join(
            f"{s}: {v:+.3f}" for s, v in sorted(
                robustness["sector_stability"].items(), key=lambda x: -abs(x[1])
            )[:5]
        )
        regime_summary = "  ".join(
            f"{r}: {v:+.3f}" for r, v in robustness["market_regime_sharpe"].items()
        )
        _log(f"  Sector IC          : {sector_summary or 'N/A'}")
        _log(f"  Subperiod Stability: {robustness['subperiod_stability']:.4f}")
        _log(f"  Regime Sharpes     : {regime_summary or 'N/A'}")
        _log(f"  Placebo Score      : {robustness['placebo_score']:.4f}\n")
        robustness_for_score = robustness

    # ── Step 5: composite score → verdict ────────────────────────────────────
    _log("[ Step 5 ] Composite scoring...")
    alpha_score = score_alpha(
        metrics,
        robustness_for_score,
        alpha.get("formula", ""),
        sim["similarity_score"],
    )
    sub = alpha_score.sub_scores
    _log(f"  Score (raw)          : {alpha_score.total:.1f} / 100")
    _log(f"  Predictive magnitude : {alpha_score.predictive_magnitude:.1f} / 100")
    _log(f"  Direction            : {alpha_score.direction_status}")
    _log(f"    performance   : {sub['performance']:.2f}")
    _log(f"    implementation: {sub['implementation']:.2f}")
    _log(f"    robustness    : {sub['robustness']:.2f}")
    _log(f"    simplicity    : {sub['simplicity']:.2f}")
    _log(f"    novelty       : {sub['novelty']:.2f}")
    _log(f"  Verdict: {alpha_score.verdict.upper()}")
    if alpha_score.failure_reason:
        _log(f"  Reason : {alpha_score.failure_reason}")
    _log()

    # ── Step 6: reflection ────────────────────────────────────────────────────
    _log("[ Step 6 ] Generating reflection...")
    reflection = generate_reflection(
        alpha, metrics, robustness, alpha_score.verdict,
        alpha_score.failure_reason, alpha_score=alpha_score,
    )
    _log()
    _log(reflection)
    _log()

    # ── Step 7: save ──────────────────────────────────────────────────────────
    _log("[ Step 7 ] Saving to database...")
    record = ExperimentRecord(
        alpha_id=alpha["alpha_id"],
        parent_id=alpha.get("parent_id"),
        batch_id=alpha.get("batch_id"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        hypothesis=alpha.get("hypothesis", ""),
        formula=alpha.get("formula", ""),
        features=alpha.get("features", []),
        mutation=alpha.get("mutation", ""),
        config=alpha,
        metrics=metrics,
        robustness=robustness,
        verdict=alpha_score.verdict,
        failure_reason=alpha_score.failure_reason,
        reflection=reflection,
        score=alpha_score.total,
        predictive_magnitude=alpha_score.predictive_magnitude,
        direction_status=alpha_score.direction_status,
        fatal=alpha_score.fatal,
        sub_scores=dict(alpha_score.sub_scores),
    )
    store.save_experiment(record)
    _log(f"  Saved experiment '{alpha['alpha_id']}'.\n")

    return ExperimentOutcome(status="completed", record=record, similarity=sim, error=None)
