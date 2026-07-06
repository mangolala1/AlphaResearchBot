"""Shared type definitions for AlphaResearchBot V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict


# "revise_invert": the signal is real but the hypothesis direction was wrong —
# treated as revise-equivalent downstream (parent-pool eligible), but semantically
# distinct so the system learns the economic intuition was inverted, not weak.
Verdict = Literal["promising", "revise", "revise_invert", "failed"]


class AlphaConfig(TypedDict, total=False):
    alpha_id: str
    parent_id: str | None
    batch_id: str | None
    hypothesis: str
    formula: str       # execution formula — uses raw column names as panel DataFrames
    features: list[str]
    mutation: str
    universe: str
    start_date: str
    end_date: str
    neutralization: str
    rebalance: str
    transaction_cost_bps: int
    holding_period_days: int


class BacktestMetrics(TypedDict):
    IC_mean: float
    ICIR: float
    Sharpe: float
    Q5_Q1_return: float
    turnover: float
    monotonicity: float
    max_drawdown: float
    deflated_sharpe: float
    noise_risk: Literal["low", "medium", "high"]


class RobustnessResult(TypedDict):
    sector_stability: dict[str, float]
    subperiod_stability: float
    market_regime_sharpe: dict[str, float]
    placebo_score: float


class SubScores(TypedDict):
    """Composite-score components, each in [0, 1]."""
    performance: float
    implementation: float
    robustness: float
    simplicity: float
    novelty: float


@dataclass
class AlphaScore:
    """Result of score_alpha() — direction-aware composite score.

    `total` evaluates the hypothesis AS STATED (sign preserved) so the system
    learns whether the economic intuition was correct. `signal_strength` is
    what parent-pool survival and bandit rewards use: the better of the two
    directions, minus an inversion penalty.
    """
    total: float                 # 0-100, directional score (hypothesis as stated)
    signal_strength: float       # 0-100, max(total, inverted_total - INVERSION_PENALTY)
    preferred_direction: int     # +1 (as stated) or -1 (inverted works better)
    sub_scores: SubScores        # directional sub-scores
    verdict: Verdict             # derived from bands on signal_strength
    failure_reason: str | None
    fatal: bool                  # a catastrophic hard gate fired


class ExperimentRecord(TypedDict):
    alpha_id: str
    parent_id: str | None
    batch_id: str | None
    timestamp: str
    hypothesis: str
    formula: str
    features: list[str]
    mutation: str
    config: AlphaConfig
    metrics: BacktestMetrics
    robustness: RobustnessResult
    verdict: Verdict
    failure_reason: str | None
    reflection: str
    # V4 composite-score fields (None on records saved before V4)
    score: NotRequired[float | None]                # directional total, 0-100
    signal_strength: NotRequired[float | None]      # max-direction score, 0-100
    preferred_direction: NotRequired[int | None]    # +1 or -1
    sub_scores: NotRequired[dict | None]            # SubScores as plain dict


class BacktestResult(TypedDict):
    metrics: BacktestMetrics
    ic_series: list[float]
    portfolio_returns: list[float]
    dates: list[str]
    sector_ic: dict[str, list[float]]
    forward_returns: list[float]
    signal_values: list[list[float]]


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SimilarityResult(TypedDict):
    is_unique: bool
    most_similar_id: str | None
    similarity_score: float
    reason: str


class ResearchSuggestion(TypedDict):
    direction: str
    hypothesis: str
    formula: str       # execution formula — raw column DataFrames, eval'd at runtime
    features: list[str]
    parent_id: str | None
    rationale: str


FailureCategory = Literal[
    "high_turnover", "weak_ic", "negative_sharpe", "high_noise", "poor_robustness",
    "too_complex", "low_novelty", "wrong_direction",
]


class MemorySummary(TypedDict):
    total_experiments: int
    verdict_counts: dict[str, int]
    failure_category_counts: dict[str, int]
    best_experiments: list[dict]
    explored_features: list[str]
    unexplored_features: list[str]
    trend_observations: list[str]
