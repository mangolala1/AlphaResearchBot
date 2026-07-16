"""Tests for AST-based structural similarity (core.similarity).

Sign blindness is intentional: -f, -1*f, f*-1 are exact duplicates of f.
Structure matters: operand order, repetition, and window constants are seen.
Config match contributes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.decision import HARD_DUPLICATE_THRESHOLD
from core.similarity import (
    canonical_formula, check_similarity, formula_columns, structural_similarity,
)


class FakeStore:
    def __init__(self, records):
        self._records = records

    def load_all(self):
        return self._records


def _record(alpha_id, formula, features=None, config=None):
    return dict(
        alpha_id=alpha_id, formula=formula, features=features or [],
        config=config or {"universe": "sp500", "rebalance": "monthly",
                          "neutralization": "sector"},
    )


def _alpha(formula, features=None):
    return dict(alpha_id="new", formula=formula, features=features or [],
                universe="sp500", rebalance="monthly", neutralization="sector")


BASE = "CFO_LTM / ADJUSTED_PRICE"


# ── Sign blindness (intentional) ──────────────────────────────────────────────

@pytest.mark.parametrize("flipped", [
    f"-({BASE})",
    f"-1 * ({BASE})",
    f"({BASE}) * -1",
    f"-(-(-({BASE})))",
    BASE,
])
def test_sign_flips_are_exact_duplicates(flipped):
    assert canonical_formula(flipped) == canonical_formula(BASE)
    store = FakeStore([_record("orig", BASE)])
    sim = check_similarity(_alpha(flipped), store, threshold=HARD_DUPLICATE_THRESHOLD)
    assert sim["is_exact_duplicate"]
    assert not sim["is_unique"]
    assert sim["structural_similarity"] == 1.0


def test_inner_sign_is_not_stripped():
    # Only the OUTER sign is canonicalized away — an inner negation changes structure
    assert canonical_formula("rank(-CFO_LTM) + rank(DA_LTM)") != \
           canonical_formula("rank(CFO_LTM) + rank(DA_LTM)")


# ── Structure the old token sets could not see ────────────────────────────────

def test_operand_order_matters():
    a = "rank(CFO_LTM / ADJUSTED_PRICE)"
    b = "rank(ADJUSTED_PRICE / CFO_LTM)"
    s = structural_similarity(a, b)
    assert s < HARD_DUPLICATE_THRESHOLD  # old token sets scored these identical
    store = FakeStore([_record("orig", a)])
    assert check_similarity(_alpha(b), store)["is_unique"]


def test_repetition_matters():
    a = "ts_mean(CFO_LTM, 20)"
    b = "ts_mean(ts_mean(CFO_LTM, 20), 20)"
    assert canonical_formula(a) != canonical_formula(b)
    assert structural_similarity(a, b) < HARD_DUPLICATE_THRESHOLD


def test_window_constants_matter():
    assert structural_similarity("ts_mean(CFO_LTM, 20)", "ts_mean(CFO_LTM, 60)") \
        < HARD_DUPLICATE_THRESHOLD


def test_commutative_reorder_is_duplicate():
    a = "rank(CFO_LTM) + rank(DA_LTM)"
    b = "rank(DA_LTM) + rank(CFO_LTM)"
    assert structural_similarity(a, b) == 1.0
    store = FakeStore([_record("orig", a)])
    assert not check_similarity(_alpha(b), store)["is_unique"]


def test_identical_formula_is_exact_duplicate():
    store = FakeStore([_record("orig", "rank(REVENUE_LTM / COGS_LTM)")])
    sim = check_similarity(_alpha("rank(REVENUE_LTM / COGS_LTM)"), store)
    assert sim["is_exact_duplicate"]


# ── Config no longer inflates similarity ──────────────────────────────────────

def test_matching_config_adds_nothing():
    # Different signals under the identical standard config must stay unique
    store = FakeStore([_record("orig", "rank(CFO_LTM / REVENUE_LTM)",
                               features=["CFO_LTM", "REVENUE_LTM"])])
    sim = check_similarity(
        _alpha("rank(NET_INCOME_LTM / ADJUSTED_PRICE)",
               features=["NET_INCOME_LTM", "ADJUSTED_PRICE"]),
        store,
    )
    assert sim["is_unique"]
    assert sim["structural_similarity"] < HARD_DUPLICATE_THRESHOLD


# ── Duplicate abort vs novelty are separate measures ─────────────────────────

def test_related_but_different_runs_with_reduced_novelty():
    """A mutation wrapping the parent formula must run, with graded similarity."""
    parent = "-1 * CFO_LTM / ADJUSTED_PRICE"
    child = "-1 * ts_zscore(CFO_LTM / ADJUSTED_PRICE, 12)"  # aborted at 0.933 pre-refactor
    store = FakeStore([_record("p", parent, features=["CFO_LTM", "ADJUSTED_PRICE"])])
    sim = check_similarity(_alpha(child, features=["CFO_LTM", "ADJUSTED_PRICE"]), store)
    assert sim["is_unique"]                    # runs
    assert 0.2 < sim["similarity_score"] < 1.0  # but novelty is penalized


def test_parse_failure_falls_back_to_tokens():
    store = FakeStore([_record("orig", "rank(CFO_LTM / ADJUSTED_PRICE")])  # unbalanced
    sim = check_similarity(_alpha("rank(CFO_LTM / ADJUSTED_PRICE"), store)
    assert not sim["is_unique"]  # identical token sets → 1.0 via fallback


# ── Column extraction (deterministic, not LLM metadata) ───────────────────────

def test_formula_columns_extraction():
    assert formula_columns("rank(CFO_LTM / REVENUE_LTM) * ts_mean(DA_LTM, 20)") == \
        {"CFO_LTM", "REVENUE_LTM", "DA_LTM"}
    assert formula_columns("rank(NOT_A_COLUMN)") == set()


def test_declared_feature_mismatch_warns():
    from core.formula_validator import validate_alpha
    result = validate_alpha(dict(
        alpha_id="x", formula="rank(CFO_LTM)", features=["CFO_LTM", "REVENUE_LTM"],
        universe="sp500", start_date="2021-01-01", end_date="2026-06-01",
    ))
    assert result.valid
    assert any("REVENUE_LTM" in w for w in result.warnings)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
