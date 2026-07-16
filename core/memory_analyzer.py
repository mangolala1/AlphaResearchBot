"""Memory analyzer — classifies experiment failures and produces structured summaries."""

from __future__ import annotations

from core.decision import (
    FATAL_IC_ABS, FATAL_SHARPE_ABS, ICIR_SOFT, TURNOVER_MAX, classify_direction,
    score_alpha,
)
from core.types import DirectionStatus, ExperimentRecord, FailureCategory, MemorySummary
from core.formula_validator import AVAILABLE_RAW_COLUMNS

# direction_evidence_stable thresholds — a contradicted direction only counts
# as stable evidence when the negative IC is consistent, not one noisy draw.
STABLE_ICIR_MAX       = -0.10  # majority of period ICs negative
STABLE_SUBPERIOD_MIN  = 0.5    # same (negative) IC direction in both halves

# Most recent tried formulas surfaced to the planner prompt
_MAX_TRIED_FORMULAS = 30


def effective_score(record: ExperimentRecord) -> float:
    """Predictive-magnitude score for parent ranking / bandit rewards.

    Prefers the stored predictive_magnitude, then the stored directional
    score; pre-V4 rows are rescored lazily with neutral novelty 0.5
    (historical similarity context can't be reconstructed).
    """
    stored = record.get("predictive_magnitude")
    if stored is not None:
        return float(stored)
    stored = record.get("score")
    if stored is not None:
        return float(stored)

    metrics = record.get("metrics") or {}
    if not metrics:
        return 0.0
    try:
        result = score_alpha(
            metrics,
            record.get("robustness") or None,
            record.get("formula", ""),
            similarity_score=0.5,
        )
        return result.predictive_magnitude
    except Exception:
        return 0.0


def direction_status_of(record: ExperimentRecord) -> DirectionStatus:
    """Stored direction_status, else classified lazily from metrics (legacy rows)."""
    stored = record.get("direction_status")
    if stored in ("supported", "contradicted", "uncertain"):
        return stored
    metrics = record.get("metrics") or {}
    if not metrics:
        return "uncertain"
    return classify_direction(metrics)


def record_is_fatal(record: ExperimentRecord) -> bool:
    """Stored fatal flag, else the legacy failure_reason prefix."""
    stored = record.get("fatal")
    if stored is not None:
        return bool(stored)
    return str(record.get("failure_reason") or "").startswith("fatal")


def direction_evidence_stable(record: ExperimentRecord) -> bool:
    """Is a contradicted direction backed by consistent evidence?

    Requires IC and Sharpe both meaningfully negative, plus consistency:
    ICIR meaningfully negative (majority of period ICs negative) or high
    subperiod stability with negative IC (negative in both halves).
    """
    metrics = record.get("metrics") or {}
    if not metrics:
        return False
    ic = metrics.get("IC_mean", 0.0)
    sharpe = metrics.get("Sharpe", 0.0)
    both_negative = ic <= -FATAL_IC_ABS and sharpe <= -FATAL_SHARPE_ABS
    if not both_negative:
        return False
    if metrics.get("ICIR", 0.0) <= STABLE_ICIR_MAX:
        return True
    robustness = record.get("robustness") or {}
    return robustness.get("subperiod_stability", 0.0) >= STABLE_SUBPERIOD_MIN


def classify_failure(record: ExperimentRecord) -> str | None:
    """Return the failure category for an experiment, or None if it is promising.

    Priority: wrong_direction → high_turnover → negative_sharpe → weak_ic →
    high_noise → poor_robustness → too_complex → low_novelty.
    First match wins. Covers 'failed' and 'revise' verdicts.
    """
    if record.get("verdict") == "promising":
        return None

    metrics = record.get("metrics") or {}
    robustness = record.get("robustness") or {}
    sub_scores = record.get("sub_scores") or {}

    if direction_status_of(record) == "contradicted":
        return "wrong_direction"

    if metrics.get("turnover", 0.0) > TURNOVER_MAX:
        return "high_turnover"
    if metrics.get("Sharpe", 0.0) < 0:
        return "negative_sharpe"
    if metrics.get("ICIR", 0.0) < ICIR_SOFT:
        return "weak_ic"
    if metrics.get("noise_risk") == "high":
        return "high_noise"
    if (
        robustness.get("subperiod_stability", 1.0) < 0.3
        or robustness.get("placebo_score", 1.0) < 0.3
    ):
        return "poor_robustness"
    if sub_scores.get("simplicity", 1.0) < 0.3:
        return "too_complex"
    if sub_scores.get("novelty", 1.0) < 0.2:
        return "low_novelty"

    return None


def analyze_memory(store: "ExperimentStore") -> MemorySummary:  # noqa: F821
    """Aggregate all experiments into a structured MemorySummary."""
    from core.memory import ExperimentStore  # local import

    records = store.load_all()

    if not records:
        return MemorySummary(
            total_experiments=0,
            verdict_counts={},
            failure_category_counts={},
            best_experiments=[],
            tried_formulas=[],
            explored_features=[],
            unexplored_features=sorted(AVAILABLE_RAW_COLUMNS),
            trend_observations=["No experiments run yet."],
        )

    verdict_counts: dict[str, int] = {}
    failure_category_counts: dict[str, int] = {}
    explored: set[str] = set()

    for r in records:
        v = r.get("verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        explored.update(r.get("features") or [])
        cat = classify_failure(r)
        if cat:
            failure_category_counts[cat] = failure_category_counts.get(cat, 0) + 1

    promising = [r for r in records if r.get("verdict") == "promising"]
    promising.sort(key=effective_score, reverse=True)
    best_experiments = [
        {
            "alpha_id": r["alpha_id"],
            "formula": r.get("formula", ""),
            "Sharpe": (r.get("metrics") or {}).get("Sharpe", 0.0),
            "ICIR": (r.get("metrics") or {}).get("ICIR", 0.0),
            "score": round(effective_score(r), 1),
        }
        for r in promising[:3]
    ]

    # Every tried formula (all verdicts, most recent last) — the planner needs
    # to SEE what was tried to avoid re-proposing it; feature names alone are
    # not enough. Capped so the prompt stays bounded.
    tried_formulas = [
        {
            "alpha_id": r["alpha_id"],
            "formula": r.get("formula", ""),
            "verdict": r.get("verdict", ""),
        }
        for r in records[-_MAX_TRIED_FORMULAS:]
        if r.get("formula")
    ]

    unexplored = sorted(AVAILABLE_RAW_COLUMNS - explored)
    explored_list = sorted(explored)

    trend_observations = _build_trend_observations(
        records, verdict_counts, failure_category_counts, best_experiments, unexplored
    )

    return MemorySummary(
        total_experiments=len(records),
        verdict_counts=verdict_counts,
        failure_category_counts=failure_category_counts,
        best_experiments=best_experiments,
        tried_formulas=tried_formulas,
        explored_features=explored_list,
        unexplored_features=unexplored,
        trend_observations=trend_observations,
    )


def _build_trend_observations(
    records: list,
    verdict_counts: dict[str, int],
    failure_category_counts: dict[str, int],
    best_experiments: list[dict],
    unexplored: list[str],
) -> list[str]:
    observations: list[str] = []
    total = len(records)
    total_failures = sum(
        verdict_counts.get(v, 0) for v in ("failed", "revise")
    )

    if total_failures > 0 and failure_category_counts:
        dominant = max(failure_category_counts, key=failure_category_counts.get)  # type: ignore[arg-type]
        pct = int(100 * failure_category_counts[dominant] / total_failures)
        observations.append(
            f"Dominant failure mode: {dominant} ({failure_category_counts[dominant]} of {total_failures} failures, {pct}%)."
        )

    if best_experiments:
        best = best_experiments[0]
        observations.append(
            f"Best score so far: {best['score']:.1f} "
            f"({best['alpha_id']}, Sharpe {best['Sharpe']:.3f})."
        )
    else:
        observations.append("No promising experiments yet.")

    if unexplored:
        sample = ", ".join(unexplored[:5])
        suffix = f" (+{len(unexplored) - 5} more)" if len(unexplored) > 5 else ""
        observations.append(f"Unexplored signals: {sample}{suffix}.")

    return observations
