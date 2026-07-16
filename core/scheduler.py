"""Thompson-sampling bandit scheduler — decides mutate-vs-explore each iteration.

Fractional-Beta Thompson sampling for continuous rewards in [0, 1]:
a reward r updates the posterior as `alpha += r; beta += (1 - r)` — treating
r as "r successes + (1-r) failures", the standard extension of Beta-Bernoulli
TS to bounded continuous rewards. No binarization anywhere.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal

from core.memory_analyzer import (
    direction_evidence_stable, direction_status_of, effective_score,
    record_is_fatal,
)
from core.types import ExperimentRecord

if TYPE_CHECKING:
    from core.experiment import ExperimentOutcome
    from core.memory import ExperimentStore

EXPLORE_ARM = "__explore__"
MAX_PARENT_ARMS = 5          # top-K eligible parents by effective_score
PARENT_SCORE_FLOOR = 35.0    # parents must be out of the failed band
CONTRADICTED_PARENT_FLOOR = 50.0  # stricter bar for contradicted-direction parents
COLD_START_MIN = 3           # force explore until store has >= this many experiments
REWARD_HALF_RANGE = 25.0     # +/- this many score points maps reward to 1.0 / 0.0
EXPLORE_PRIOR = (2.0, 1.0)   # optimistic — bootstraps exploration
BASELINE_DEFAULT = 50.0      # explore baseline when the store is empty

_ELIGIBLE_VERDICTS = {"promising", "revise"}


def _is_eligible_parent(record: ExperimentRecord) -> bool:
    """Two ways into the parent pool.

    Normal path: a promising/revise verdict above the standard floor.
    Contradicted path: the stated direction was wrong but the signal itself is
    strong (predictive_magnitude above a stricter floor), non-fatal, and the
    negative direction is consistent — a research branch worth mutating
    structurally, not a verdict.
    """
    score = effective_score(record)
    if record.get("verdict") in _ELIGIBLE_VERDICTS and score >= PARENT_SCORE_FLOOR:
        return True
    return (
        direction_status_of(record) == "contradicted"
        and not record_is_fatal(record)
        and score >= CONTRADICTED_PARENT_FLOOR
        and direction_evidence_stable(record)
    )


def _parent_prior(parent_score: float) -> tuple[float, float]:
    """New parent arms start with a prior whose mean scales with parent quality."""
    return (1.0 + parent_score / 100.0, 1.0)


class ThompsonScheduler:
    """Beta-posterior bandit over one explore arm + top-K mutable parents."""

    def __init__(self, store: "ExperimentStore") -> None:
        self._store = store
        self._state: dict[str, dict] = store.load_bandit_state()

    # ── Arms ──────────────────────────────────────────────────────────────────

    def eligible_parent_arms(self) -> list[str]:
        """Arm ids ("mutate:<alpha_id>") for the top-K eligible parents."""
        records = self._store.load_all()
        eligible = [r for r in records if _is_eligible_parent(r)]
        eligible.sort(key=effective_score, reverse=True)
        return [f"mutate:{r['alpha_id']}" for r in eligible[:MAX_PARENT_ARMS]]

    def _params(self, arm_id: str) -> tuple[float, float]:
        """Posterior (alpha, beta) for an arm, falling back to its prior."""
        if arm_id in self._state:
            return self._state[arm_id]["alpha"], self._state[arm_id]["beta"]
        if arm_id == EXPLORE_ARM:
            return EXPLORE_PRIOR
        parent_id = arm_id.removeprefix("mutate:")
        record = self._store.load_by_id(parent_id)
        parent_score = effective_score(record) if record else PARENT_SCORE_FLOOR
        return _parent_prior(parent_score)

    # ── Selection ─────────────────────────────────────────────────────────────

    def select_action(self) -> tuple[Literal["explore", "mutate"], str | None]:
        """Sample every arm's posterior and pick the argmax.

        Returns ("explore", None) or ("mutate", parent_alpha_id).
        """
        if len(self._store.load_all()) < COLD_START_MIN:
            return "explore", None

        arms = [EXPLORE_ARM] + self.eligible_parent_arms()
        samples = {
            arm: random.betavariate(*self._params(arm)) for arm in arms
        }
        best = max(samples, key=samples.__getitem__)
        if best == EXPLORE_ARM:
            return "explore", None
        return "mutate", best.removeprefix("mutate:")

    # ── Rewards ───────────────────────────────────────────────────────────────

    def reward_for(
        self,
        action: Literal["explore", "mutate"],
        parent_record: ExperimentRecord | None,
        outcome: "ExperimentOutcome",
    ) -> float:
        """Continuous reward in [0, 1] based on composite-score improvement."""
        if outcome.status != "completed" or outcome.record is None:
            return 0.0  # wasted pull — duplicates / validation / backtest errors

        child_score = effective_score(outcome.record)
        if action == "mutate" and parent_record is not None:
            baseline = effective_score(parent_record)
        else:
            records = self._store.load_all()
            prior = [r for r in records if r["alpha_id"] != outcome.record["alpha_id"]]
            baseline = (
                sum(effective_score(r) for r in prior) / len(prior)
                if prior else BASELINE_DEFAULT
            )
        r = 0.5 + (child_score - baseline) / (2 * REWARD_HALF_RANGE)
        return min(1.0, max(0.0, r))

    # ── Posterior update ──────────────────────────────────────────────────────

    def update(self, arm_id: str, reward: float) -> None:
        """Fractional Beta update: alpha += r, beta += (1 - r). Persists immediately."""
        r = min(1.0, max(0.0, reward))
        a, b = self._params(arm_id)
        state = self._state.setdefault(arm_id, {"alpha": a, "beta": b, "pulls": 0})
        state["alpha"] = a + r
        state["beta"] = b + (1.0 - r)
        state["pulls"] += 1
        self._store.upsert_bandit_arm(
            arm_id, state["alpha"], state["beta"], state["pulls"]
        )

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary_lines(self) -> list[str]:
        """Human-readable per-arm posterior summary for loop output."""
        lines = []
        for arm_id, s in sorted(self._state.items()):
            mean = s["alpha"] / (s["alpha"] + s["beta"])
            lines.append(
                f"  {arm_id:<40s} α={s['alpha']:.2f} β={s['beta']:.2f} "
                f"pulls={s['pulls']} mean={mean:.3f}"
            )
        return lines or ["  (no pulls yet)"]
