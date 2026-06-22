"""Shared type definitions for AlphaResearchBot V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict


Verdict = Literal["promising", "failed", "inconclusive"]


class AlphaConfig(TypedDict, total=False):
    alpha_id: str
    parent_id: str | None
    hypothesis: str
    formula: str
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
    turnover: float
    max_drawdown: float
    deflated_sharpe: float
    noise_risk: Literal["low", "medium", "high"]


class RobustnessResult(TypedDict):
    sector_stability: float
    subperiod_stability: float
    market_regime_sharpe: float
    placebo_score: float


class ExperimentRecord(TypedDict):
    alpha_id: str
    parent_id: str | None
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
    formula: str
    features: list[str]
    parent_id: str | None
    rationale: str
