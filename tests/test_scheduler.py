"""Unit tests for the fractional-Beta Thompson scheduler (core.scheduler)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.memory import ExperimentStore
from core.scheduler import COLD_START_MIN, EXPLORE_ARM, ThompsonScheduler


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(db_path=str(tmp_path / "test.db"))


def _fake_record(alpha_id, verdict="revise", strength=50.0, parent_id=None):
    return dict(
        alpha_id=alpha_id, parent_id=parent_id, batch_id="b1",
        timestamp="2026-07-05T00:00:00", hypothesis="h", formula="rank(REVENUE_LTM)",
        features=[], mutation="", config={},
        metrics=dict(IC_mean=0.02, ICIR=0.3, Sharpe=0.5, deflated_sharpe=0.4,
                     monotonicity=0.3, turnover=0.4, max_drawdown=-0.2,
                     Q5_Q1_return=0.05, noise_risk="low"),
        robustness=dict(sector_stability={}, subperiod_stability=0.5,
                        market_regime_sharpe={}, placebo_score=0.5),
        verdict=verdict, failure_reason=None, reflection="",
        score=strength, signal_strength=strength, preferred_direction=1,
        sub_scores=None,
    )


def test_cold_start_forces_explore(store):
    sched = ThompsonScheduler(store)
    for _ in range(10):
        action, parent = sched.select_action()
        assert action == "explore"
        assert parent is None


def test_fractional_update_is_exact(store):
    sched = ThompsonScheduler(store)
    a0, b0 = sched._params(EXPLORE_ARM)
    sched.update(EXPLORE_ARM, 0.7)
    state = store.load_bandit_state()[EXPLORE_ARM]
    assert state["alpha"] == pytest.approx(a0 + 0.7)  # no binarization
    assert state["beta"] == pytest.approx(b0 + 0.3)
    assert state["pulls"] == 1


def test_update_clips_reward(store):
    sched = ThompsonScheduler(store)
    a0, b0 = sched._params(EXPLORE_ARM)
    sched.update(EXPLORE_ARM, 1.7)
    state = store.load_bandit_state()[EXPLORE_ARM]
    assert state["alpha"] == pytest.approx(a0 + 1.0)
    assert state["beta"] == pytest.approx(b0)


def test_eligibility_excludes_failed_and_low_scores(store):
    store.save_experiment(_fake_record("good", "promising", 80.0))
    store.save_experiment(_fake_record("inverted", "revise_invert", 60.0))
    store.save_experiment(_fake_record("weak", "revise", 20.0))     # below floor
    store.save_experiment(_fake_record("dead", "failed", 70.0))     # failed verdict
    sched = ThompsonScheduler(store)
    arms = sched.eligible_parent_arms()
    assert arms == ["mutate:good", "mutate:inverted"]


def test_convergence_to_better_arm(store):
    random.seed(42)
    # Enough records to leave cold start, with two eligible parents
    store.save_experiment(_fake_record("arm_a", "promising", 80.0))
    store.save_experiment(_fake_record("arm_b", "promising", 79.0))
    store.save_experiment(_fake_record("filler", "failed", 10.0))
    sched = ThompsonScheduler(store)

    # Feed asymmetric rewards: arm_a gets 0.9, arm_b gets 0.1, explore 0.1
    for _ in range(30):
        sched.update("mutate:arm_a", 0.9)
        sched.update("mutate:arm_b", 0.1)
        sched.update(EXPLORE_ARM, 0.1)

    picks = {"mutate:arm_a": 0, "mutate:arm_b": 0, EXPLORE_ARM: 0}
    for _ in range(200):
        action, parent = sched.select_action()
        key = EXPLORE_ARM if action == "explore" else f"mutate:{parent}"
        picks[key] += 1
    assert picks["mutate:arm_a"] > 150  # converged to the good arm


def test_state_persists_across_instances(store):
    sched = ThompsonScheduler(store)
    sched.update(EXPLORE_ARM, 0.6)
    sched2 = ThompsonScheduler(store)
    a, b = sched2._params(EXPLORE_ARM)
    assert a == pytest.approx(2.6)  # EXPLORE_PRIOR (2,1) + 0.6
    assert b == pytest.approx(1.4)


def test_reward_for_failed_outcome_is_zero(store):
    from core.experiment import ExperimentOutcome
    sched = ThompsonScheduler(store)
    outcome = ExperimentOutcome(status="duplicate", record=None, similarity=None, error="dup")
    assert sched.reward_for("explore", None, outcome) == 0.0


def test_mutation_reward_symmetric_around_parent(store):
    from core.experiment import ExperimentOutcome
    sched = ThompsonScheduler(store)
    parent = _fake_record("p", "promising", 50.0)

    child_same = _fake_record("c1", "revise", 50.0, parent_id="p")
    outcome = ExperimentOutcome(status="completed", record=child_same, similarity=None, error=None)
    assert sched.reward_for("mutate", parent, outcome) == pytest.approx(0.5)

    child_up = _fake_record("c2", "promising", 75.0, parent_id="p")
    outcome = ExperimentOutcome(status="completed", record=child_up, similarity=None, error=None)
    assert sched.reward_for("mutate", parent, outcome) == pytest.approx(1.0)

    child_down = _fake_record("c3", "failed", 25.0, parent_id="p")
    outcome = ExperimentOutcome(status="completed", record=child_down, similarity=None, error=None)
    assert sched.reward_for("mutate", parent, outcome) == pytest.approx(0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
