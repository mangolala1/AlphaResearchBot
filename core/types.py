"""Shared type definitions for AlphaResearchBot V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict


# Direction is not a separate research outcome: a real-but-inverted signal is
# just "failed" (or "revise"), with the contradiction recorded in failure_reason.
Verdict = Literal["promising", "revise", "failed"]

# Direction of the evidence relative to the stated hypothesis:
# "supported"    — IC and Sharpe both positive with real magnitude
# "contradicted" — IC and Sharpe both negative with real magnitude
# "uncertain"    — dead signal or mixed signs
DirectionStatus = Literal["supported", "contradicted", "uncertain"]


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
    """Result of score_alpha() — two composites plus a direction flag.

    `total` evaluates the hypothesis AS STATED (raw signed metrics) — verdicts
    and diagnostics use it. `predictive_magnitude` scores the same alpha on
    abs() metrics — "a signal exists here", direction-blind — and drives parent
    eligibility and bandit rewards. `direction_status` says whether the
    evidence supported or contradicted the stated direction.
    """
    total: float                 # 0-100, composite on raw signed metrics
    predictive_magnitude: float  # 0-100, composite on abs() metrics
    direction_status: DirectionStatus
    sub_scores: SubScores        # directional sub-scores
    verdict: Verdict             # derived from bands on total
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
    predictive_magnitude: NotRequired[float | None] # abs-metric composite, 0-100
    direction_status: NotRequired[str | None]       # DirectionStatus
    fatal: NotRequired[bool | None]                 # a catastrophic hard gate fired
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
    is_unique: bool                # False → duplicate abort (exact or structural >= threshold)
    is_exact_duplicate: bool       # canonical match after sign stripping
    most_similar_id: str | None
    similarity_score: float        # combined research similarity → novelty = 1 - this
    structural_similarity: float   # max sign-blind AST overlap (drives the abort)
    feature_similarity: float      # declared-feature Jaccard vs the most similar alpha
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
    tried_formulas: list[dict]        # recent {alpha_id, formula, verdict}, all verdicts
    explored_features: list[str]
    unexplored_features: list[str]
    trend_observations: list[str]
