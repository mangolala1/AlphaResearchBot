"""Tests for v3.5: memory_analyzer, failure taxonomy, planner prompt, graph enrichment."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest
from unittest.mock import MagicMock, patch

from core.types import ExperimentRecord, BacktestMetrics, RobustnessResult
from core.memory_analyzer import classify_failure, analyze_memory
from core.formula_validator import EVALUATOR_FEATURES, ALLOWED_FUNCTION_NAMES


def _make_record(
    alpha_id: str = "test_alpha",
    verdict: str = "failed",
    turnover: float = 350.0,
    sharpe: float = -0.1,
    icir: float = -0.05,
    noise_risk: str = "low",
    sector_stability: float = 0.8,
    subperiod_stability: float = 0.8,
    placebo_score: float = 0.8,
    features: list[str] | None = None,
    parent_id: str | None = None,
    mutation: str = "",
) -> ExperimentRecord:
    metrics = BacktestMetrics(
        IC_mean=0.01,
        ICIR=icir,
        Sharpe=sharpe,
        turnover=turnover,
        max_drawdown=-0.3,
        deflated_sharpe=-0.5,
        noise_risk=noise_risk,
    )
    robustness = RobustnessResult(
        sector_stability=sector_stability,
        subperiod_stability=subperiod_stability,
        market_regime_sharpe=0.5,
        placebo_score=placebo_score,
    )
    return ExperimentRecord(
        alpha_id=alpha_id,
        parent_id=parent_id,
        timestamp="2026-06-22T00:00:00",
        hypothesis="test hypothesis",
        formula="rank(MOM12_1)",
        features=features or ["MOM12_1"],
        mutation=mutation,
        config={},
        metrics=metrics,
        robustness=robustness,
        verdict=verdict,
        failure_reason="some reason",
        reflection="",
    )


class TestClassifyFailure(unittest.TestCase):

    def test_promising_returns_none(self):
        r = _make_record(verdict="promising", turnover=100, sharpe=0.8, icir=0.5)
        self.assertIsNone(classify_failure(r))

    def test_high_turnover_wins_over_negative_sharpe(self):
        r = _make_record(verdict="failed", turnover=400, sharpe=-0.5, icir=-0.3)
        self.assertEqual(classify_failure(r), "high_turnover")

    def test_negative_sharpe(self):
        r = _make_record(verdict="failed", turnover=100, sharpe=-0.1, icir=-0.05)
        self.assertEqual(classify_failure(r), "negative_sharpe")

    def test_weak_ic(self):
        r = _make_record(verdict="failed", turnover=100, sharpe=0.2, icir=0.1)
        self.assertEqual(classify_failure(r), "weak_ic")

    def test_high_noise(self):
        r = _make_record(verdict="inconclusive", turnover=100, sharpe=0.6, icir=0.4, noise_risk="high")
        self.assertEqual(classify_failure(r), "high_noise")

    def test_poor_robustness_sector(self):
        r = _make_record(verdict="failed", turnover=100, sharpe=0.6, icir=0.4, sector_stability=0.2)
        self.assertEqual(classify_failure(r), "poor_robustness")

    def test_poor_robustness_subperiod(self):
        r = _make_record(verdict="failed", turnover=100, sharpe=0.6, icir=0.4, subperiod_stability=0.1)
        self.assertEqual(classify_failure(r), "poor_robustness")

    def test_poor_robustness_placebo(self):
        r = _make_record(verdict="failed", turnover=100, sharpe=0.6, icir=0.4, placebo_score=0.2)
        self.assertEqual(classify_failure(r), "poor_robustness")

    def test_inconclusive_still_classified(self):
        r = _make_record(verdict="inconclusive", turnover=400)
        self.assertEqual(classify_failure(r), "high_turnover")


class TestAnalyzeMemory(unittest.TestCase):

    def _make_store(self, records):
        store = MagicMock()
        store.load_all.return_value = records
        return store

    def test_empty_store(self):
        summary = analyze_memory(self._make_store([]))
        self.assertEqual(summary["total_experiments"], 0)
        self.assertEqual(summary["verdict_counts"], {})
        self.assertEqual(summary["failure_category_counts"], {})
        self.assertEqual(summary["best_experiments"], [])
        self.assertEqual(summary["explored_features"], [])
        self.assertEqual(set(summary["unexplored_features"]), EVALUATOR_FEATURES)
        self.assertEqual(summary["trend_observations"], ["No experiments run yet."])

    def test_verdict_counts(self):
        records = [
            _make_record("a1", verdict="failed", turnover=400),
            _make_record("a2", verdict="failed", turnover=400),
            _make_record("a3", verdict="inconclusive", turnover=100, sharpe=0.6, icir=0.4, noise_risk="high"),
        ]
        summary = analyze_memory(self._make_store(records))
        self.assertEqual(summary["verdict_counts"]["failed"], 2)
        self.assertEqual(summary["verdict_counts"]["inconclusive"], 1)
        self.assertEqual(summary["total_experiments"], 3)

    def test_failure_category_counts(self):
        records = [
            _make_record("a1", verdict="failed", turnover=400),
            _make_record("a2", verdict="failed", turnover=400),
            _make_record("a3", verdict="failed", turnover=100, sharpe=-0.2, icir=-0.1),
        ]
        summary = analyze_memory(self._make_store(records))
        self.assertEqual(summary["failure_category_counts"]["high_turnover"], 2)
        self.assertEqual(summary["failure_category_counts"]["negative_sharpe"], 1)

    def test_explored_unexplored_features(self):
        records = [
            _make_record("a1", features=["MOM12_1", "EBITDA_MARGIN"]),
        ]
        summary = analyze_memory(self._make_store(records))
        self.assertIn("MOM12_1", summary["explored_features"])
        self.assertIn("EBITDA_MARGIN", summary["explored_features"])
        self.assertNotIn("MOM12_1", summary["unexplored_features"])
        self.assertNotIn("EBITDA_MARGIN", summary["unexplored_features"])

    def test_best_experiments_sorted_by_sharpe(self):
        records = [
            _make_record("a1", verdict="promising", turnover=50, sharpe=0.6, icir=0.4),
            _make_record("a2", verdict="promising", turnover=50, sharpe=0.9, icir=0.5),
            _make_record("a3", verdict="promising", turnover=50, sharpe=0.7, icir=0.45),
        ]
        summary = analyze_memory(self._make_store(records))
        self.assertEqual(len(summary["best_experiments"]), 3)
        self.assertEqual(summary["best_experiments"][0]["alpha_id"], "a2")

    def test_trend_observations_content(self):
        records = [_make_record("a1", verdict="failed", turnover=400)]
        summary = analyze_memory(self._make_store(records))
        obs_text = " ".join(summary["trend_observations"])
        self.assertIn("high_turnover", obs_text)


class TestValidatorConstants(unittest.TestCase):

    def test_evaluator_features_count(self):
        self.assertEqual(len(EVALUATOR_FEATURES), 14)

    def test_evaluator_features_excludes_internal(self):
        for f in ("SECTOR", "INDUSTRY", "TICKER", "EPS_NTM", "SALES_NTM"):
            self.assertNotIn(f, EVALUATOR_FEATURES)

    def test_safe_operators_excludes_unimplemented(self):
        from core.mutator import _SAFE_OPERATORS
        for op in ("delta", "ts_mean", "ts_std"):
            self.assertNotIn(op, _SAFE_OPERATORS)
        for op in ("rank", "zscore", "log", "abs", "sign"):
            self.assertIn(op, _SAFE_OPERATORS)


class TestGraphEnrichment(unittest.TestCase):

    def test_node_has_failure_category(self):
        from core.graph import ResearchGraph
        g = ResearchGraph()
        r = _make_record("test_node", verdict="failed", turnover=400, mutation="added quality overlay")
        g.add_experiment(r)
        attrs = g.get_graph().nodes["test_node"]
        self.assertEqual(attrs["failure_category"], "high_turnover")
        self.assertEqual(attrs["mutation_reason"], "added quality overlay")

    def test_promising_node_has_na_category(self):
        from core.graph import ResearchGraph
        g = ResearchGraph()
        r = _make_record("test_promising", verdict="promising", turnover=50, sharpe=0.8, icir=0.5)
        g.add_experiment(r)
        attrs = g.get_graph().nodes["test_promising"]
        self.assertEqual(attrs["failure_category"], "N/A")


class TestVisualizationTooltip(unittest.TestCase):

    def test_tooltip_includes_new_fields(self):
        from core.visualization import _build_tooltip
        attrs = {
            "alpha_id": "alpha_001",
            "hypothesis": "test",
            "formula": "rank(MOM12_1)",
            "Sharpe": 0.5,
            "ICIR": 0.3,
            "deflated_sharpe": 0.2,
            "verdict": "failed",
            "failure_reason": "High turnover",
            "failure_category": "high_turnover",
            "mutation_reason": "switched to quarterly rebalance",
            "reflection": "some reflection",
        }
        tooltip = _build_tooltip(attrs)
        self.assertIn("Failure Category", tooltip)
        self.assertIn("high_turnover", tooltip)
        self.assertIn("Mutation Reason", tooltip)
        self.assertIn("switched to quarterly rebalance", tooltip)


if __name__ == "__main__":
    unittest.main(verbosity=2)
