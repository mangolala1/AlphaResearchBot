"""Real robustness checks computed from BacktestResult data."""

from __future__ import annotations

import random

import numpy as np
from scipy import stats

from core.types import AlphaConfig, BacktestResult, RobustnessResult


def run_robustness(
    alpha: AlphaConfig,
    backtest_result: BacktestResult,
) -> RobustnessResult:
    """Compute real robustness scores from backtest intermediate data."""
    ic_series = backtest_result["ic_series"]
    portfolio_returns = backtest_result["portfolio_returns"]
    sector_ic = backtest_result["sector_ic"]
    forward_returns = backtest_result["forward_returns"]

    sector_stability = _sector_stability(ic_series, sector_ic)
    subperiod_stability = _subperiod_stability(ic_series)
    market_regime_sharpe = _market_regime_sharpe(alpha, portfolio_returns, backtest_result["dates"])
    placebo_score = _placebo_score(ic_series, forward_returns, backtest_result["signal_values"])

    return RobustnessResult(
        sector_stability=round(sector_stability, 4),
        subperiod_stability=round(subperiod_stability, 4),
        market_regime_sharpe=round(market_regime_sharpe, 4),
        placebo_score=round(placebo_score, 4),
    )


def _sector_stability(
    ic_series: list[float], sector_ic: dict[str, list[float]]
) -> float:
    """How consistent is the IC across sectors? Higher = more stable."""
    if not sector_ic or len(sector_ic) < 2:
        return 0.5  # can't compute with fewer than 2 sectors

    overall_ic_mean = abs(np.mean(ic_series)) + 1e-9
    per_sector_means = [np.mean(v) for v in sector_ic.values() if len(v) >= 3]

    if len(per_sector_means) < 2:
        return 0.5

    dispersion = np.std(per_sector_means, ddof=1)
    score = 1.0 - dispersion / overall_ic_mean
    return float(np.clip(score, 0.0, 1.0))


def _subperiod_stability(ic_series: list[float]) -> float:
    """How consistent is IC in the first vs second half of the period?"""
    if len(ic_series) < 4:
        return 0.5

    mid = len(ic_series) // 2
    ic_first = np.mean(ic_series[:mid])
    ic_second = np.mean(ic_series[mid:])
    denom = abs(ic_first) + abs(ic_second) + 1e-9
    score = 1.0 - abs(ic_first - ic_second) / denom
    return float(np.clip(score, 0.0, 1.0))


def _market_regime_sharpe(
    alpha: AlphaConfig,
    portfolio_returns: list[float],
    dates: list[str],
) -> float:
    """Sharpe in bull vs bear regimes. Score = bull_sharpe / (|bull| + |bear|)."""
    if len(portfolio_returns) < 6:
        return 0.5

    try:
        import yfinance as yf
        spy = yf.download("SPY", start=alpha.get("start_date"), end=alpha.get("end_date"),
                          progress=False, auto_adjust=True)["Close"].squeeze()
        spy_monthly = spy.resample("BME").last().pct_change().dropna()
    except Exception:
        # If yfinance fails, fall back to subperiod approach
        return _subperiod_stability(portfolio_returns)

    # Align SPY dates with our backtest dates
    import pandas as pd
    spy_monthly.index = spy_monthly.index.tz_localize(None)
    port_series = pd.Series(portfolio_returns, index=pd.to_datetime(dates[:len(portfolio_returns)]))

    common_dates = port_series.index.intersection(spy_monthly.index)
    if len(common_dates) < 6:
        return _subperiod_stability(portfolio_returns)

    port_aligned = port_series.loc[common_dates]
    spy_aligned = spy_monthly.loc[common_dates]

    bull_mask = spy_aligned > 0
    bull_rets = port_aligned[bull_mask]
    bear_rets = port_aligned[~bull_mask]

    def _sharpe(rets):
        if len(rets) < 3:
            return 0.0
        return float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-9) * np.sqrt(12))

    bull_sharpe = _sharpe(bull_rets)
    bear_sharpe = _sharpe(bear_rets)
    denom = abs(bull_sharpe) + abs(bear_sharpe) + 1e-9
    score = bull_sharpe / denom
    return float(np.clip(score, 0.0, 1.0))


def _placebo_score(
    ic_series: list[float],
    forward_returns: list[float],
    signal_values: list[list[float]],
) -> float:
    """1 - (mean placebo IC / |real IC mean|). Near 1 = signal not just noise."""
    if not forward_returns or not signal_values:
        return 0.5

    real_ic_mean = abs(np.mean(ic_series)) + 1e-9
    fwd = list(forward_returns)
    placebo_ics: list[float] = []

    rng = random.Random(42)
    for _ in range(50):
        shuffled = fwd[:]
        rng.shuffle(shuffled)
        # Compute IC between first signal_values batch and shuffled returns
        if signal_values and len(shuffled) >= len(signal_values[0]):
            sig_flat = signal_values[0]
            ret_chunk = shuffled[: len(sig_flat)]
            if len(sig_flat) >= 5:
                ic_val, _ = stats.spearmanr(sig_flat, ret_chunk)
                if not np.isnan(ic_val):
                    placebo_ics.append(abs(float(ic_val)))

    if not placebo_ics:
        return 0.5

    mean_placebo = np.mean(placebo_ics)
    score = 1.0 - mean_placebo / real_ic_mean
    return float(np.clip(score, 0.0, 1.0))
