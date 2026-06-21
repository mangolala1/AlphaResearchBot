"""Mock robustness checks — deterministic via alpha hash."""

from __future__ import annotations

import hashlib
import random

from core.types import AlphaConfig, BacktestMetrics, RobustnessResult


def run_robustness(alpha: AlphaConfig, metrics: BacktestMetrics) -> RobustnessResult:
    """Return deterministic mock robustness scores in [0, 1]."""
    rng = _make_rng(alpha)

    sector_stability = round(rng.uniform(0.0, 1.0), 4)
    subperiod_stability = round(rng.uniform(0.0, 1.0), 4)
    market_regime_sharpe = round(rng.uniform(0.0, 1.0), 4)

    # placebo_score near 0 means the alpha performs poorly on shuffled data — good sign
    placebo_score = round(rng.uniform(0.0, 1.0), 4)

    return RobustnessResult(
        sector_stability=sector_stability,
        subperiod_stability=subperiod_stability,
        market_regime_sharpe=market_regime_sharpe,
        placebo_score=placebo_score,
    )


def _make_rng(alpha: AlphaConfig) -> random.Random:
    key = "robustness|" + "|".join([
        alpha.get("formula", ""),
        alpha.get("universe", ""),
        alpha.get("neutralization", ""),
    ])
    digest = hashlib.sha256(key.encode()).hexdigest()
    seed = int(digest, 16) % (2**32)
    return random.Random(seed)
