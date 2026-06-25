"""Real robustness checks computed from BacktestResult data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.types import AlphaConfig, BacktestResult, RobustnessResult

# Module-level cache: populated once on first call to _load_regime_data()
_REGIME_CACHE: dict = {}


def _load_regime_data() -> bool:
    """Fetch SPY and VIX from 2020-01-01 to today, compute regime booleans, and cache.
    Returns True on success, False if data could not be fetched."""
    if _REGIME_CACHE:
        return True

    try:
        import yfinance as yf
        spy_raw = yf.download("SPY", start="2020-01-01", progress=False, auto_adjust=True)["Close"].squeeze()
        vix_raw = yf.download("^VIX", start="2020-01-01", progress=False, auto_adjust=True)["Close"].squeeze()
    except Exception:
        return False

    def _strip_tz(s: pd.Series) -> pd.Series:
        if s.index.tz is not None:
            s.index = s.index.tz_convert(None)
        return s

    spy_monthly = _strip_tz(spy_raw.resample("BME").last())
    vix_monthly = _strip_tz(vix_raw.resample("BME").last())
    spy_trend = spy_monthly.pct_change(12)

    vix_25 = vix_monthly.quantile(0.25)
    vix_75 = vix_monthly.quantile(0.75)

    _REGIME_CACHE["bull"]        = spy_trend > 0
    _REGIME_CACHE["high_vol"]    = vix_monthly > vix_75
    _REGIME_CACHE["low_vol"]     = vix_monthly < vix_25
    _REGIME_CACHE["neutral_vol"] = (vix_monthly >= vix_25) & (vix_monthly <= vix_75)
    return True


def run_robustness(
    alpha: AlphaConfig,
    backtest_result: BacktestResult,
) -> RobustnessResult:
    """Compute real robustness scores from backtest intermediate data."""
    ic_series = backtest_result["ic_series"]
    portfolio_returns = backtest_result["portfolio_returns"]
    sector_ic = backtest_result["sector_ic"]
    forward_returns = backtest_result["forward_returns"]

    sector_stability = _sector_stability(sector_ic)
    subperiod_stability = _subperiod_stability(ic_series)
    market_regime_sharpe = _market_regime_sharpe(portfolio_returns, backtest_result["dates"])
    placebo_score = _placebo_score(ic_series, forward_returns, backtest_result["signal_values"])

    return RobustnessResult(
        sector_stability=sector_stability,
        subperiod_stability=round(subperiod_stability, 4),
        market_regime_sharpe=market_regime_sharpe,
        placebo_score=round(placebo_score, 4),
    )


def _sector_stability(sector_ic: dict[str, list[float]]) -> dict[str, float]:
    """Return mean IC per sector."""
    return {sector: round(float(np.mean(ics)), 4) for sector, ics in sector_ic.items() if ics}


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
    portfolio_returns: list[float],
    dates: list[str],
) -> dict[str, float]:
    """Return annualised Sharpe for each of five regimes: bull, bear, high_vol,
    low_vol, neutral_vol. Regime data is fetched once from yfinance and cached."""
    if not _load_regime_data():
        return {}

    port_series = pd.Series(
        portfolio_returns,
        index=pd.to_datetime(dates[: len(portfolio_returns)]),
    )

    common = port_series.index.intersection(_REGIME_CACHE["bull"].dropna().index)
    if not len(common):
        return {}

    port        = port_series.loc[common]
    bull        = _REGIME_CACHE["bull"].loc[common]
    high_vol    = _REGIME_CACHE["high_vol"].loc[common]
    low_vol     = _REGIME_CACHE["low_vol"].loc[common]
    neutral_vol = _REGIME_CACHE["neutral_vol"].loc[common]

    regimes = {
        "bull":        port[bull],
        "bear":        port[~bull],
        "high_vol":    port[high_vol],
        "low_vol":     port[low_vol],
        "neutral_vol": port[neutral_vol],
    }

    def _sharpe(rets: pd.Series) -> float | None:
        if len(rets) < 3:
            return None
        return round(float(np.mean(rets) / (np.std(rets, ddof=1) + 1e-9) * np.sqrt(12)), 4)

    return {name: s for name, rets in regimes.items() if (s := _sharpe(rets)) is not None}


def _placebo_score(
    ic_series: list[float],
    forward_returns: list[float],
    signal_values: list[list[float]],
) -> float:
    """For each month, shuffle that month's forward returns, compute IC against
    that month's signal, collect placebo ICs, then average.
    Score = 1 - mean_placebo_IC / real_IC_mean (higher = real IC is well above chance)."""
    if not forward_returns or not signal_values:
        return 0.5

    real_ic_mean = abs(np.mean(ic_series)) + 1e-9
    rng = np.random.default_rng(42)
    placebo_ics: list[float] = []

    # Reconstruct per-period returns: signal_values[i] has the stocks for period i,
    # and forward_returns is a flat concatenation in the same order.
    offset = 0
    for sig in signal_values:
        n = len(sig)
        if offset + n > len(forward_returns):
            break
        period_rets = np.array(forward_returns[offset : offset + n])
        offset += n

        if n < 5:
            continue

        shuffled = rng.permutation(period_rets)
        with np.errstate(invalid="ignore"):
            ic_val, _ = stats.spearmanr(sig, shuffled)
        if not np.isnan(ic_val):
            placebo_ics.append(abs(float(ic_val)))

    if not placebo_ics:
        return 0.5

    score = 1.0 - np.mean(placebo_ics) / real_ic_mean
    return float(np.clip(score, 0.0, 1.0))
