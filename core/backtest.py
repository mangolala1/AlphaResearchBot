"""Backtest engine — loads data via DataLoader and simulates a long-short portfolio."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from core.data_process import _DATA_DIR, process as process_data, processed_path
from core.signal_calculation import compute_signal
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
    no_cache = getattr(data_loader, "_no_cache", False)

    prices_df, income_ttm_df, cashflow_ttm_df, universe_df = data_loader.load(start_date, end_date)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Process and cache income statement
    income_path = processed_path("income", start_date, end_date)
    if not no_cache and income_path.exists():
        # print(f"[Backtest] Loading processed income from {income_path}")
        processed_income = pd.read_parquet(income_path)
    else:
        processed_income = process_data(income_ttm_df)
        processed_income.to_parquet(income_path)
        print(f"[Backtest] Saved processed income to {income_path}")

    # Process and cache cash flow statement
    cashflow_path = processed_path("cashflow", start_date, end_date)
    if not no_cache and cashflow_path.exists():
        # print(f"[Backtest] Loading processed cashflow from {cashflow_path}")
        processed_cashflow = pd.read_parquet(cashflow_path)
    else:
        processed_cashflow = process_data(cashflow_ttm_df)
        processed_cashflow.to_parquet(cashflow_path)
        print(f"[Backtest] Saved processed cashflow to {cashflow_path}")

    # Join income + cashflow on (DATE, TICKER) index; drop any accidental duplicate columns
    processed_df = processed_income.join(processed_cashflow, how="outer", rsuffix="_dup")
    processed_df = processed_df.drop(
        columns=[c for c in processed_df.columns if c.endswith("_dup")]
    )

    # Add SECTOR from universe
    if "SECTOR" in universe_df.columns:
        sector_map = universe_df.set_index("TICKER")["SECTOR"]
        processed_df["SECTOR"] = processed_df.index.get_level_values("TICKER").map(sector_map)

    # Join raw price columns back — not standardised, used in momentum/ratio formulas.
    # Then restrict the panel to actual trading dates (dates present in the price data).
    # The fundamentals panel uses pd.bdate_range which includes US market holidays; prices
    # don't. Dropping non-trading rows here means rolling operators like ts_std(X, 60)
    # always see a clean window with no NaN gaps from holidays.
    prices = prices_df.copy()
    prices["DATE"] = pd.to_datetime(prices["DATE"])
    price_mi = prices.set_index(["DATE", "TICKER"])[["ADJUSTED_PRICE", "ADJUSTED_VOLUME"]]
    processed_df = processed_df.join(price_mi)
    trading_dates = price_mi.index.get_level_values("DATE").unique()
    processed_df = processed_df[
        processed_df.index.get_level_values("DATE").isin(trading_dates)
    ]

    # Compute full signal panel over the entire date range at once
    signal_series = compute_signal(processed_df, alpha["formula"])

    # Generate monthly rebalancing dates
    rebal_dates = _monthly_dates(start_date, end_date, processed_df)

    # Build price pivot for forward return calculation
    price_pivot = prices_df.copy()
    price_pivot["DATE"] = pd.to_datetime(price_pivot["DATE"])
    price_pivot = price_pivot.pivot_table(
        index="DATE", columns="TICKER", values="ADJUSTED_PRICE"
    )

    ic_series: list[float] = []
    portfolio_returns: list[float] = []
    dates_out: list[str] = []
    sector_ic: dict[str, list[float]] = {}
    all_forward_returns: list[float] = []
    all_signal_values: list[list[float]] = []
    quintile_rets_list: list[list[float]] = []
    top_quintile_sets: list[frozenset] = []
    bottom_quintile_sets: list[frozenset] = []
    degraded_bin_periods: int = 0

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]

        # Slice the pre-computed signal at this rebalancing date
        try:
            signal = signal_series.loc[date]
        except KeyError:
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
        try:
            sector_col = processed_df.loc[date]["SECTOR"].reindex(common)
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
        except KeyError:
            pass

        # Portfolio simulation: quintiles (Q1=bottom, Q5=top)
        # Deduplicate bin edges first — duplicate quantiles collapse bins and
        # cause a label-count mismatch when the signal has low cardinality.
        cuts = signal_aligned.quantile([0.2, 0.4, 0.6, 0.8])
        bins = sorted(set([-np.inf, float(cuts[0.2]), float(cuts[0.4]),
                           float(cuts[0.6]), float(cuts[0.8]), np.inf]))
        n_bins = len(bins) - 1
        if n_bins < 5:
            degraded_bin_periods += 1
        bin_labels = list(range(1, n_bins + 1))
        labels = pd.cut(signal_aligned, bins=bins, labels=bin_labels)
        q_rets = [
            float(fwd_ret_aligned[labels == q].mean()) if (labels == q).sum() > 0 else 0.0
            for q in bin_labels
        ]
        # Pad to 5 elements so downstream _compute_metrics always sees the same shape
        q_rets_5 = (q_rets + [0.0] * 5)[:5]
        quintile_rets_list.append(q_rets_5)
        top_quintile_sets.append(frozenset(signal_aligned[labels == bin_labels[-1]].index.tolist()))
        bottom_quintile_sets.append(frozenset(signal_aligned[labels == bin_labels[0]].index.tolist()))

        port_ret = q_rets[-1] - q_rets[0]  # top bin - bottom bin
        portfolio_returns.append(port_ret)

    total_periods = len(ic_series)
    if degraded_bin_periods > 0:
        pct = 100 * degraded_bin_periods / total_periods if total_periods else 0
        print(
            f"  [backtest] WARNING: signal collapsed to <5 quintile bins in "
            f"{degraded_bin_periods}/{total_periods} periods ({pct:.0f}%). "
            "This indicates weak cross-sectional resolution: many stocks share "
            "identical signal values."
        )

    if len(ic_series) < 3:
        raise RuntimeError(
            "Backtest produced fewer than 3 valid periods — "
            "check that the date range and features align with available data."
        )

    metrics = _compute_metrics(
        ic_series, portfolio_returns, quintile_rets_list,
        top_quintile_sets, bottom_quintile_sets,
    )

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
    start_date: str, end_date: str, panel: pd.DataFrame
) -> list:
    """Return month-end dates available in the panel index."""
    panel_dates = panel.index.get_level_values("DATE").unique().sort_values()
    monthly = pd.date_range(start=start_date, end=end_date, freq="BME")
    result = []
    for m in monthly:
        available = panel_dates[panel_dates <= m]
        if len(available) > 0:
            result.append(available[-1])
    return sorted(set(result))


def _compute_metrics(
    ic_series: list[float],
    portfolio_returns: list[float],
    quintile_rets_list: list[list[float]],
    top_quintile_sets: list[frozenset],
    bottom_quintile_sets: list[frozenset],
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
    deflated_sharpe = _deflated_sharpe(rets, sharpe)
    long_turnover = _compute_turnover(top_quintile_sets)
    short_turnover = _compute_turnover(bottom_quintile_sets)
    turnover = (long_turnover + short_turnover) / 2
    monotonicity = _compute_monotonicity(quintile_rets_list)

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

    return BacktestMetrics(
        IC_mean=round(ic_mean, 4),
        ICIR=round(icir, 4),
        Sharpe=round(float(sharpe), 4),
        turnover=round(turnover, 4),
        monotonicity=round(monotonicity, 4),
        max_drawdown=round(max_drawdown, 4),
        deflated_sharpe=round(deflated_sharpe, 4),
        noise_risk=noise_risk,  # type: ignore[arg-type]
    )


def _compute_turnover(quintile_sets: list[frozenset]) -> float:
    if len(quintile_sets) < 2:
        return 0.0
    turnovers = []
    for prev, curr in zip(quintile_sets[:-1], quintile_sets[1:]):
        if len(prev) == 0:
            continue
        overlap = len(prev & curr)
        turnovers.append(1 - overlap / len(prev))
    return float(np.mean(turnovers))


def _compute_monotonicity(quintile_rets_list: list[list[float]]) -> float:
    if not quintile_rets_list:
        return 0.0
    quintile_ranks = [1, 2, 3, 4, 5]
    rhos = []
    for q_rets in quintile_rets_list:
        if len(q_rets) != 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = stats.spearmanr(quintile_ranks, q_rets)
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else 0.0


def _max_drawdown(returns: np.ndarray) -> float:
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))


def _deflated_sharpe(returns: np.ndarray, sharpe: float) -> float:
    """Sharpe ratio haircut using López de Prado (2013) SR standard error."""
    n = len(returns)
    if n < 4:
        return 0.0

    sigma = returns.std(ddof=1) + 1e-9
    normalised = (returns - returns.mean()) / sigma

    skew = float(np.mean(normalised ** 3))
    kurt = float(np.mean(normalised ** 4))

    sr_var = (1 + 0.5 * sharpe ** 2 - skew * sharpe + ((kurt - 3) / 4) * sharpe ** 2) / (n - 1)
    sr_std = float(np.sqrt(max(sr_var, 0.0)))

    return round(float(sharpe - sr_std), 4)


if __name__ == "__main__":
    import json
    import pprint

    with open("experiments/sample_alpha_001.json") as f:
        config: AlphaConfig = json.load(f)

    result = run_backtest(config)
    pprint.pprint(result["metrics"])
