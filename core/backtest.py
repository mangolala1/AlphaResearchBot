"""Mock deterministic backtest engine."""

from __future__ import annotations

import hashlib
import random

from core.types import AlphaConfig, BacktestMetrics


def run_backtest(alpha: AlphaConfig) -> BacktestMetrics:
    """Return deterministic mock backtest metrics seeded from the alpha config."""
    rng = _make_rng(alpha)

    sharpe = round(rng.uniform(-0.3, 2.2), 4)
    ic_mean = round(rng.uniform(-0.04, 0.12), 4)
    icir = round(rng.uniform(-0.3, 1.8), 4)
    turnover = round(rng.uniform(20.0, 450.0), 2)
    max_drawdown = round(rng.uniform(-0.45, -0.05), 4)

    deflation_ratio = rng.uniform(0.4, 1.0)
    deflated_sharpe = round(sharpe * deflation_ratio, 4)

    if sharpe <= 0:
        noise_risk: str = "high"
    else:
        ratio = deflated_sharpe / sharpe
        if ratio < 0.5:
            noise_risk = "high"
        elif ratio < 0.75:
            noise_risk = "medium"
        else:
            noise_risk = "low"

    return BacktestMetrics(
        IC_mean=ic_mean,
        ICIR=icir,
        Sharpe=sharpe,
        turnover=turnover,
        max_drawdown=max_drawdown,
        deflated_sharpe=deflated_sharpe,
        noise_risk=noise_risk,  # type: ignore[arg-type]
    )


def _make_rng(alpha: AlphaConfig) -> random.Random:
    """Seed a Random instance from formula + universe + neutralization."""
    key = "|".join([
        alpha.get("formula", ""),
        alpha.get("universe", ""),
        alpha.get("neutralization", ""),
    ])
    digest = hashlib.sha256(key.encode()).hexdigest()
    seed = int(digest, 16) % (2**32)
    return random.Random(seed)
