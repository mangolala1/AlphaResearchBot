"""Tests for planner duplicate avoidance and unknown-column validation.

Covers the 7.16 fixes: tried formulas in the planner prompt, explore-path
avoid-list, generation-time duplicate pre-check, and unknown columns as
validation errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.planner as planner_mod
from core.formula_validator import validate_alpha
from core.memory_analyzer import analyze_memory
from core.planner import _build_plan_prompt, plan_next_research


class FakeStore:
    def __init__(self, records):
        self._records = records

    def load_all(self):
        return self._records


def _record(alpha_id, formula, verdict="revise", features=None):
    return dict(
        alpha_id=alpha_id, formula=formula, verdict=verdict,
        features=features or [], metrics={}, robustness={}, sub_scores={},
        failure_reason=None, score=50.0, predictive_magnitude=50.0,
    )


def _suggestion(direction, formula, features):
    return dict(direction=direction, hypothesis="h", formula=formula,
                features=features, parent_id=None, rationale="r")


# ── Fix 4: unknown columns are validation errors, not warnings ────────────────

def test_unknown_column_is_validation_error():
    result = validate_alpha(dict(
        alpha_id="x", formula="OPERATING_INCOME_LTM / ASSETS_LTM", features=[],
        universe="sp500", start_date="2021-01-01", end_date="2026-06-01",
    ))
    assert not result.valid
    assert any("ASSETS_LTM" in e for e in result.errors)


def test_known_columns_still_validate():
    result = validate_alpha(dict(
        alpha_id="x", formula="rank(CFO_LTM / REVENUE_LTM)", features=[],
        universe="sp500", start_date="2021-01-01", end_date="2026-06-01",
    ))
    assert result.valid


# ── Fix 1: tried formulas (all verdicts) reach the planner prompt ─────────────

def test_prompt_shows_tried_formulas_and_avoid_list():
    store = FakeStore([
        _record("a1", "CFO_LTM / REVENUE_LTM", verdict="failed"),
        _record("a2", "GROSS_PROFIT_LTM / REVENUE_LTM", verdict="revise"),
    ])
    summary = analyze_memory(store)
    assert [t["formula"] for t in summary["tried_formulas"]] == [
        "CFO_LTM / REVENUE_LTM", "GROSS_PROFIT_LTM / REVENUE_LTM",
    ]

    prompt = _build_plan_prompt(summary, 3, avoid_formulas=["rank(DA_LTM)"])
    assert "[failed] CFO_LTM / REVENUE_LTM" in prompt          # non-promising visible
    assert "[revise] GROSS_PROFIT_LTM / REVENUE_LTM" in prompt
    assert "REJECTED as duplicates" in prompt                  # avoid block present
    assert "rank(DA_LTM)" in prompt
    assert "do NOT re-propose any already-tested formula" in prompt


# ── Fix 3: generation-time duplicate pre-check ────────────────────────────────

def test_plan_drops_duplicate_suggestions(monkeypatch):
    store = FakeStore([_record("orig", "CFO_LTM / REVENUE_LTM",
                               features=["CFO_LTM", "REVENUE_LTM"])])

    def fake_llm(summary, n, avoid_formulas=None):
        return [
            _suggestion("dup exact", "CFO_LTM / REVENUE_LTM",
                        ["CFO_LTM", "REVENUE_LTM"]),
            _suggestion("dup signflip", "-1 * (CFO_LTM / REVENUE_LTM)",
                        ["CFO_LTM", "REVENUE_LTM"]),
            _suggestion("novel", "rank(ADJUSTED_VOLUME) * rank(DA_LTM / COGS_LTM)",
                        ["ADJUSTED_VOLUME", "DA_LTM", "COGS_LTM"]),
        ]

    monkeypatch.setattr(planner_mod, "_llm_plan", fake_llm)
    got = plan_next_research(store, n=1)
    assert got
    assert got[0]["direction"] == "novel"


def test_fallback_suggestions_are_also_duplicate_checked(monkeypatch):
    # Store already contains the first rule-based fallback formula
    tried = "rank(ADJUSTED_PRICE / (REVENUE_LTM / SHARES_DILUTED)) * -1"
    store = FakeStore([_record("v", tried)])

    def fake_llm(summary, n, avoid_formulas=None):
        raise RuntimeError("no api")

    monkeypatch.setattr(planner_mod, "_llm_plan", fake_llm)
    got = plan_next_research(store, n=2)
    formulas = [s["formula"] for s in got]
    assert len(got) == 2
    assert tried not in formulas


# ── Fix 2: avoid_formulas reaches the LLM call ────────────────────────────────

def test_avoid_formulas_passed_through(monkeypatch):
    store = FakeStore([])
    seen = {}

    def fake_llm(summary, n, avoid_formulas=None):
        seen["avoid"] = avoid_formulas
        return [_suggestion("s", "rank(CFO_LTM)", ["CFO_LTM"])]

    monkeypatch.setattr(planner_mod, "_llm_plan", fake_llm)
    plan_next_research(store, n=1, avoid_formulas=["rank(DA_LTM)"])
    assert seen["avoid"] == ["rank(DA_LTM)"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
