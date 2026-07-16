"""Tests for the mutator's generation-time duplicate pre-check."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.mutator as mutator_mod
from core.mutator import _ensure_not_duplicate, generate_mutation


class FakeStore:
    def __init__(self, records):
        self._records = records

    def load_all(self):
        return self._records

    def load_by_id(self, alpha_id):
        return next((r for r in self._records if r["alpha_id"] == alpha_id), None)


def _parent(alpha_id="p", formula="rank(CFO_LTM / REVENUE_LTM)", turnover=0.2, sharpe=0.5):
    return dict(
        alpha_id=alpha_id, formula=formula, verdict="revise",
        features=["CFO_LTM", "REVENUE_LTM"],
        metrics=dict(turnover=turnover, Sharpe=sharpe, ICIR=0.3),
        robustness={}, sub_scores={}, failure_reason=None,
        hypothesis="h", score=50.0, predictive_magnitude=50.0,
        config={"formula": formula, "rebalance": "monthly", "universe": "sp500"},
    )


def _llm_unavailable(*args, **kwargs):
    raise RuntimeError("no api")


def test_fallback_duplicate_raises(monkeypatch):
    # turnover > 0.7 → rule-based mutation keeps the formula (rebalance-only
    # change), which duplicates the stored parent → must fail at generation
    parent = _parent(turnover=0.8)
    store = FakeStore([parent])
    monkeypatch.setattr(mutator_mod, "_llm_mutation", _llm_unavailable)
    with pytest.raises(ValueError, match="structural duplicate"):
        generate_mutation("p", store)


def test_fallback_unique_mutation_passes(monkeypatch):
    # sharpe < 0.3, no quality columns → rule-based adds a quality overlay,
    # structurally different from the parent → returned normally
    parent = _parent(formula="rank(ADJUSTED_VOLUME)", sharpe=0.1)
    parent["features"] = ["ADJUSTED_VOLUME"]
    parent["config"]["formula"] = "rank(ADJUSTED_VOLUME)"
    store = FakeStore([parent])
    monkeypatch.setattr(mutator_mod, "_llm_mutation", _llm_unavailable)
    child = generate_mutation("p", store)
    assert child["parent_id"] == "p"
    assert child["formula"] != parent["formula"]


def test_ensure_not_duplicate_flags_sign_flip():
    store = FakeStore([_parent()])
    with pytest.raises(ValueError, match="structural duplicate"):
        _ensure_not_duplicate(
            {"alpha_id": "c", "formula": "-1 * (rank(CFO_LTM / REVENUE_LTM))",
             "features": ["CFO_LTM", "REVENUE_LTM"]},
            store,
        )


def test_ensure_not_duplicate_passes_novel():
    store = FakeStore([_parent()])
    _ensure_not_duplicate(
        {"alpha_id": "c", "formula": "rank(DA_LTM / COGS_LTM)",
         "features": ["DA_LTM", "COGS_LTM"]},
        store,
    )  # no raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
