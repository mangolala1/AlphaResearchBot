"""Structural similarity check — compares a new alpha against all prior experiments.

Uses Jaccard similarity on feature sets and formula token sets.
No LLM required: formulas are short and structured enough for token comparison.
"""

from __future__ import annotations

import re

from core.types import AlphaConfig, SimilarityResult


def check_similarity(
    new_alpha: AlphaConfig,
    store: "ExperimentStore",  # noqa: F821 — avoid circular import
    threshold: float = 0.8,
) -> SimilarityResult:
    """Compare new_alpha against all experiments in store.

    Returns SimilarityResult with is_unique=False if any prior alpha scores >= threshold.
    """
    from core.memory import ExperimentStore  # local import to avoid circular dep

    records = store.load_all()
    if not records:
        return SimilarityResult(
            is_unique=True,
            most_similar_id=None,
            similarity_score=0.0,
            reason="No prior experiments to compare against.",
        )

    best_score = 0.0
    best_id: str | None = None

    new_features = set(new_alpha.get("features") or [])
    new_tokens = _formula_tokens(new_alpha.get("formula") or "")

    for record in records:
        score = _score(new_alpha, new_features, new_tokens, record)
        if score > best_score:
            best_score = score
            best_id = record["alpha_id"]

    is_unique = best_score < threshold

    if is_unique:
        reason = (
            f"Most similar prior alpha is '{best_id}' at {best_score:.0%} — below the "
            f"{threshold:.0%} threshold. Alpha is sufficiently different."
        ) if best_id else "No prior experiments found."
    else:
        reason = (
            f"Alpha is {best_score:.0%} similar to '{best_id}' based on overlapping "
            f"features and formula tokens. Mutate further or use --force to override."
        )

    return SimilarityResult(
        is_unique=is_unique,
        most_similar_id=best_id,
        similarity_score=round(best_score, 4),
        reason=reason,
    )


def _score(
    new_alpha: AlphaConfig,
    new_features: set[str],
    new_tokens: set[str],
    record: dict,
) -> float:
    existing_features = set(record.get("features") or [])
    existing_tokens = _formula_tokens(record.get("formula") or "")

    feature_jaccard = _jaccard(new_features, existing_features)
    token_jaccard = _jaccard(new_tokens, existing_tokens)
    combined = 0.5 * feature_jaccard + 0.5 * token_jaccard

    # Config bonus: same universe + rebalance + neutralization
    existing_config = record.get("config") or {}
    config_match = (
        new_alpha.get("universe") == existing_config.get("universe")
        and new_alpha.get("rebalance") == existing_config.get("rebalance")
        and new_alpha.get("neutralization") == existing_config.get("neutralization")
    )
    if config_match:
        combined = min(1.0, combined + 0.1)

    return combined


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _formula_tokens(formula: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
