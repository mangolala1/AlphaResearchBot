"""Memory analyzer — classifies experiment failures and produces structured summaries."""

from __future__ import annotations

from core.decision import ICIR_SOFT, TURNOVER_MAX
from core.types import ExperimentRecord, FailureCategory, MemorySummary
from core.formula_validator import EVALUATOR_FEATURES


def classify_failure(record: ExperimentRecord) -> str | None:
    """Return the failure category for an experiment, or None if it is promising.

    Priority: high_turnover → negative_sharpe → weak_ic → high_noise → poor_robustness.
    First match wins. Covers both 'failed' and 'revise' verdicts.
    """
    if record.get("verdict") == "promising":
        return None

    metrics = record.get("metrics") or {}
    robustness = record.get("robustness") or {}

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
            explored_features=[],
            unexplored_features=sorted(EVALUATOR_FEATURES),
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
    promising.sort(key=lambda r: (r.get("metrics") or {}).get("Sharpe", float("-inf")), reverse=True)
    best_experiments = [
        {
            "alpha_id": r["alpha_id"],
            "formula": r.get("formula", ""),
            "Sharpe": (r.get("metrics") or {}).get("Sharpe", 0.0),
            "ICIR": (r.get("metrics") or {}).get("ICIR", 0.0),
        }
        for r in promising[:3]
    ]

    unexplored = sorted(EVALUATOR_FEATURES - explored)
    explored_list = sorted(explored)

    trend_observations = _build_trend_observations(
        records, verdict_counts, failure_category_counts, best_experiments, unexplored
    )

    return MemorySummary(
        total_experiments=len(records),
        verdict_counts=verdict_counts,
        failure_category_counts=failure_category_counts,
        best_experiments=best_experiments,
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
            f"Best Sharpe so far: {best['Sharpe']:.3f} ({best['alpha_id']})."
        )
    else:
        observations.append("No promising experiments yet.")

    if unexplored:
        sample = ", ".join(unexplored[:5])
        suffix = f" (+{len(unexplored) - 5} more)" if len(unexplored) > 5 else ""
        observations.append(f"Unexplored signals: {sample}{suffix}.")

    return observations
