"""Backtest engine — loads data via DataLoader and simulates a long-short portfolio."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from core.features import compute_features
from core.formula_eval import evaluate_formula
from core.types import AlphaConfig, BacktestMetrics, BacktestResult

if TYPE_CHECKING:
    from core.data_loader import DataLoader


def run_backtest(
    alpha: AlphaConfig,
    data_loader: "DataLoader | None" = None,
) -> BacktestResult:
    """Run a real backtest and return metrics plus rich intermediate data for robustness."""
    if data_loader is None:
        from core.data_loader import DataLoader as _DL
        data_loader = _DL()

    start_date = alpha["start_date"]
    end_date = alpha["end_date"]

    # Load raw data (cached after first run)
    prices_df, fundamentals_df, universe_df = data_loader.load(start_date, end_date)

    # Compute derived features
    feature_panel = compute_features(
        prices_df, fundamentals_df, universe_df, alpha.get("features", [])
    )

    # Generate monthly rebalancing dates
    rebal_dates = _monthly_dates(start_date, end_date, feature_panel)

    # Build price pivot for forward return calculation
    price_pivot = prices_df.copy()
    price_pivot["DATE"] = pd.to_datetime(price_pivot["DATE"])
    price_pivot = price_pivot.pivot_table(
        index="DATE", columns="FACTSET_ID", values="ADJUSTED_PRICE"
    )

    ic_series: list[float] = []
    portfolio_returns: list[float] = []
    dates_out: list[str] = []
    sector_ic: dict[str, list[float]] = {}
    all_forward_returns: list[float] = []
    all_signal_values: list[list[float]] = []

    holding_days = int(alpha.get("holding_period_days", 20))
    prev_weights: pd.Series | None = None

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]

        # Cross-section of features at this date
        try:
            cs_features = feature_panel.loc[date]
        except KeyError:
            continue

        if cs_features.empty:
            continue

        # Build feature dict for formula evaluator
        feature_cols = [c for c in cs_features.columns if c != "SECTOR"]
        cross_section = {col: cs_features[col].dropna() for col in feature_cols}

        # Evaluate formula → signal
        try:
            signal = evaluate_formula(alpha["formula"], cross_section)
        except Exception:
            continue

        if len(signal) < 20:
            continue

        # Forward returns: price at next_date vs price at date
        stocks = signal.index
        try:
            fwd_prices_now = price_pivot.loc[date, stocks]
            fwd_prices_next = price_pivot.loc[next_date, stocks]
        except KeyError:
            continue

        fwd_ret = (fwd_prices_next / fwd_prices_now - 1).dropna()
        common = signal.index.intersection(fwd_ret.index)
        if len(common) < 10:
            continue

        signal_aligned = signal.loc[common]
        fwd_ret_aligned = fwd_ret.loc[common]

        # IC: Spearman rank correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic_val, _ = stats.spearmanr(signal_aligned.values, fwd_ret_aligned.values)

        if np.isnan(ic_val):
            continue

        ic_series.append(float(ic_val))
        all_forward_returns.extend(fwd_ret_aligned.tolist())
        all_signal_values.append(signal_aligned.tolist())
        dates_out.append(str(date.date()) if hasattr(date, "date") else str(date))

        # Sector-level IC
        if "SECTOR" in cs_features.columns:
            sector_col = cs_features["SECTOR"].reindex(common)
            for sector, grp in sector_col.groupby(sector_col):
                if len(grp) < 5:
                    continue
                grp_signal = signal_aligned.loc[grp.index]
                grp_fwd = fwd_ret_aligned.loc[grp.index]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sec_ic, _ = stats.spearmanr(grp_signal.values, grp_fwd.values)
                if not np.isnan(sec_ic):
                    sector_ic.setdefault(sector, []).append(float(sec_ic))

        # Portfolio simulation: long top quintile, short bottom quintile
        q_low = signal_aligned.quantile(0.2)
        q_high = signal_aligned.quantile(0.8)
        long_stocks = signal_aligned[signal_aligned >= q_high].index
        short_stocks = signal_aligned[signal_aligned <= q_low].index

        long_ret = fwd_ret_aligned.loc[long_stocks].mean() if len(long_stocks) > 0 else 0.0
        short_ret = fwd_ret_aligned.loc[short_stocks].mean() if len(short_stocks) > 0 else 0.0
        port_ret = float(long_ret - short_ret)
        portfolio_returns.append(port_ret)

    if len(ic_series) < 3:
        raise RuntimeError(
            "Backtest produced fewer than 3 valid periods — "
            "check that the date range and features align with available data."
        )

    metrics = _compute_metrics(ic_series, portfolio_returns)

    return BacktestResult(
        metrics=metrics,
        ic_series=ic_series,
        portfolio_returns=portfolio_returns,
        dates=dates_out,
        sector_ic=sector_ic,
        forward_returns=all_forward_returns,
        signal_values=all_signal_values,
    )


def _monthly_dates(
    start_date: str, end_date: str, feature_panel: pd.DataFrame
) -> list:
    """Return month-end dates available in the feature panel index."""
    panel_dates = feature_panel.index.get_level_values("DATE").unique().sort_values()
    monthly = pd.date_range(start=start_date, end=end_date, freq="BME")
    result = []
    for m in monthly:
        # Snap to nearest available date
        available = panel_dates[panel_dates <= m]
        if len(available) > 0:
            result.append(available[-1])
    return sorted(set(result))


def _compute_metrics(
    ic_series: list[float],
    portfolio_returns: list[float],
) -> BacktestMetrics:
    ics = np.array(ic_series)
    rets = np.array(portfolio_returns)

    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1)) + 1e-9
    icir = ic_mean / ic_std

    ret_mean = float(np.mean(rets))
    ret_std = float(np.std(rets, ddof=1)) + 1e-9
    sharpe = ret_mean / ret_std * np.sqrt(12)  # annualised (monthly periods)

    max_drawdown = _max_drawdown(rets)

    # Deflated Sharpe: 25th percentile of rolling 12-period Sharpe windows
    deflated_sharpe = _deflated_sharpe(rets, sharpe)

    if sharpe <= 0:
        noise_risk = "high"
    else:
        ratio = deflated_sharpe / (sharpe + 1e-9)
        if ratio < 0.5:
            noise_risk = "high"
        elif ratio < 0.75:
            noise_risk = "medium"
        else:
            noise_risk = "low"

    # Turnover estimate: approximated from IC volatility (not from actual weight changes
    # since we don't track individual weights across periods in this simplified engine)
    turnover = float(np.std(ics, ddof=1) * 200 * 12)  # rough bps/yr proxy

    return BacktestMetrics(
        IC_mean=round(ic_mean, 4),
        ICIR=round(icir, 4),
        Sharpe=round(float(sharpe), 4),
        turnover=round(turnover, 2),
        max_drawdown=round(max_drawdown, 4),
        deflated_sharpe=round(deflated_sharpe, 4),
        noise_risk=noise_risk,  # type: ignore[arg-type]
    )


def _max_drawdown(returns: np.ndarray) -> float:
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))


def _deflated_sharpe(returns: np.ndarray, sharpe: float) -> float:
    """Conservative Sharpe estimate: 25th percentile of rolling 12-period windows."""
    if len(returns) < 13:
        return sharpe * 0.7
    window = 12
    rolling_sharpes = []
    for i in range(len(returns) - window + 1):
        window_rets = returns[i : i + window]
        w_mean = np.mean(window_rets)
        w_std = np.std(window_rets, ddof=1) + 1e-9
        rolling_sharpes.append(w_mean / w_std * np.sqrt(12))
    return float(np.percentile(rolling_sharpes, 25))
