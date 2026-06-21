# AlphaResearchBot V2 — Implementation Plan

## Context

V1 delivered a complete research loop with deterministic mock backtest and robustness checks. V2 replaces those two mock modules with real implementations backed by Snowflake data, and adds a feature engineering layer to bridge raw Snowflake columns (EBITDA_LTM, ADJUSTED_PRICE, etc.) to derived alpha features (EBITDA_MARGIN, MOM12_1, etc.).

**What changes:** `core/backtest.py`, `core/robustness.py`, `core/types.py` (extended), plus three new modules: `core/data_loader.py`, `core/features.py`, `core/formula_eval.py`.

**What stays unchanged:** `core/validator.py`, `core/decision.py`, `core/reflection.py`, `core/memory.py`, `core/graph.py`, `core/visualization.py`, `scripts/export_graph.py`. The SQLite schema and all downstream storage/visualization are untouched.

**`scripts/run_experiment.py`** needs a small update to pass `BacktestResult` (not just `BacktestMetrics`) to robustness.

---

## New Directory Structure (additions only)

```
AlphaResearchBot/
├── core/
│   ├── data_loader.py      # NEW — thin wrapper around SnowflakeDataRetriever + parquet cache
│   ├── features.py         # NEW — derive EBITDA_MARGIN, MOM12_1, VOL_20D, etc. from raw DataFrames
│   └── formula_eval.py     # NEW — safe formula evaluator (restricted eval with pandas ops)
├── cache/                  # NEW — auto-created; parquet cache for Snowflake query results
└── requirements.txt        # UPDATED — add scipy, numpy, pyarrow
```

---

## Updated Type Definitions — `core/types.py`

Add `BacktestResult` to pass rich output from backtest → robustness (avoids re-querying data):

```python
class BacktestResult(TypedDict):
    metrics: BacktestMetrics          # same TypedDict as V1 (unchanged)
    ic_series: list[float]            # IC per rebalancing period
    portfolio_returns: list[float]    # portfolio return per period
    dates: list[str]                  # rebalancing dates (ISO strings)
    sector_ic: dict[str, list[float]] # {sector_name: [ic_per_period]}
    forward_returns: list[float]      # raw forward returns per period (for placebo)
    signal_values: list[list[float]]  # signal cross-sections per period (for placebo)
```

`BacktestMetrics` and all other types are **unchanged** — DB schema is unaffected.

---

## New Module: `core/data_loader.py`

Thin wrapper around `Data/data_retrieval.SnowflakeDataRetriever` with **parquet caching** so Snowflake is only queried once per date range.

```python
class DataLoader:
    def __init__(self, cache_dir: str = "cache") -> None

    def load(
        self,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (prices_df, fundamentals_df, universe_df).
        Fetches from Snowflake on first call; reads parquet cache on subsequent calls.
        """
```

Cache key: `sha256(start_date + end_date + table_name)` → filename like `cache/prices_<hash>.parquet`.

Cache strategy: if the parquet file exists, read it. Otherwise call `SnowflakeDataRetriever`, write result to parquet, then return. No TTL in V2 — delete `cache/` manually to refresh.

**Uses from `Data/data_retrieval.py`:**
- `SnowflakeDataRetriever.get_prices_data(start_date, end_date)`
- `SnowflakeDataRetriever.get_fundamentals_data(start_date, end_date)`
- `SnowflakeDataRetriever.get_universe_data()`

**Uses from `Data/config.py`:** `SNOWFLAKE_CONFIG` dict for connection.

---

## New Module: `core/features.py`

Computes derived features from raw Snowflake DataFrames using pandas. Returns a wide panel DataFrame indexed by `(DATE, FACTSET_ID)`.

```python
def compute_features(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute requested derived features. Returns MultiIndex (DATE, FACTSET_ID) DataFrame."""
```

**Feature definitions** (all computed per stock per date):

| Feature | Computation |
|---|---|
| `EBITDA_MARGIN` | `EBITDA_LTM / SALES_LTM` (winsorized at 1%/99%) |
| `MOM12_1` | 12-month price return, excluding most recent month: `price[t-21] / price[t-252] - 1` |
| `MOM6_1` | 6-month price return, excluding most recent month: `price[t-21] / price[t-126] - 1` |
| `SALES_GROWTH` | YoY sales growth: `(SALES_LTM[t] / SALES_LTM[t-252]) - 1` |
| `EPS_GROWTH` | YoY EPS growth: `(EPS_LTM[t] / EPS_LTM[t-252]) - 1` |
| `PRICE_TO_SALES` | `ADJUSTED_PRICE / (SALES_LTM / 1e6)` (scaled) |
| `VOL_20D` | 20-day rolling std of daily log returns |
| Raw columns | `EPS_LTM`, `EBITDA_LTM`, `ADJUSTED_PRICE`, `ADJUSTED_VOLUME`, etc. — passed through directly |

Raw Snowflake columns (`EPS_LTM`, `ADJUSTED_PRICE`, etc.) are passed through directly when requested.

Missing / inf values: filled with NaN (stocks with NaN signal are excluded from that period's cross-section).

---

## New Module: `core/formula_eval.py`

Safely evaluates a formula string against a cross-sectional Series per stock.

```python
def evaluate_formula(
    formula: str,
    cross_section: dict[str, pd.Series],
) -> pd.Series:
    """Evaluate formula string using a restricted namespace of feature Series.
    Returns a signal Series indexed by FACTSET_ID.
    """
```

**Approach:** `eval()` with an explicit allowlist namespace — no builtins exposed.

Namespace includes:
- Each feature name → the corresponding `pd.Series` (already in the cross-section dict)
- `rank(s)` → `s.rank(pct=True)`
- `zscore(s)` → `(s - s.mean()) / s.std()`
- `log(s)` → `np.log(s.clip(lower=1e-9))`
- `abs(s)` → `s.abs()`
- `sign(s)` → `np.sign(s)`
- `delta(s, n=1)` → requires time-series context; raises `NotImplementedError` with message "delta() requires time-series context — use ts_mean or ts_std instead"
- `ts_mean(s, n)` → requires time-series context; same error
- `ts_std(s, n)` → requires time-series context; same error

> Note: `delta`, `ts_mean`, `ts_std` are time-series operators that require panel context. In V2 these are supported at the feature computation stage (pre-computed features), not inside the formula evaluator. The validator will warn if these are used directly in the formula string.

---

## Replaced Module: `core/backtest.py`

The mock is fully replaced. Same public interface signature, extended return type.

```python
def run_backtest(
    alpha: AlphaConfig,
    data_loader: DataLoader | None = None,
) -> BacktestResult:
```

If `data_loader` is None, a default `DataLoader()` is instantiated.

**Pipeline:**

1. **Load data** — `DataLoader.load(start_date, end_date)` → prices, fundamentals, universe
2. **Compute features** — `compute_features(prices, fundamentals, universe, alpha["features"])` → panel DataFrame
3. **Get rebalancing dates** — generate monthly dates within `[start_date, end_date]` aligned to month-end
4. **For each rebalancing date:**
   a. Extract cross-section of features at that date
   b. Evaluate formula → raw signal Series (FACTSET_ID)
   c. Compute forward returns: price return over `holding_period_days` (shift prices forward)
   d. Drop stocks with NaN signal or NaN forward return
   e. Compute IC: Spearman rank correlation (`scipy.stats.spearmanr`) between signal and forward return
   f. Record IC and sector membership for each stock
5. **Portfolio simulation:**
   - Long top-quintile stocks, short bottom-quintile stocks, equal-weighted within each leg
   - Compute period-by-period portfolio return (long leg avg return − short leg avg return)
   - Compute turnover: mean absolute change in portfolio weights between periods × 10000 (bps)
6. **Aggregate metrics:**
   - `IC_mean`: mean of IC series
   - `ICIR`: `IC_mean / std(IC series)`
   - `Sharpe`: `mean(portfolio_returns) / std(portfolio_returns) * sqrt(12)` (annualized, monthly)
   - `max_drawdown`: maximum peak-to-trough of cumulative portfolio returns
   - `deflated_sharpe`: Sharpe computed on a rolling 12-period window; take the 25th percentile of rolling Sharpes as a conservative estimate
   - `noise_risk`: derived same rule as V1 (`deflated_sharpe / Sharpe` ratio)

Returns `BacktestResult` with all of the above plus `ic_series`, `portfolio_returns`, `dates`, `sector_ic`, `forward_returns`, `signal_values`.

---

## Replaced Module: `core/robustness.py`

Real computation from `BacktestResult`. Updated signature:

```python
def run_robustness(
    alpha: AlphaConfig,
    backtest_result: BacktestResult,
) -> RobustnessResult:
```

**Real computations:**

| Check | Method |
|---|---|
| `sector_stability` | For each sector: compute mean IC over all periods where that sector had ≥5 stocks. `sector_stability = 1 - std(per_sector_mean_IC) / (abs(overall_IC_mean) + 1e-9)`, clipped to [0, 1]. Higher = more stable across sectors. |
| `subperiod_stability` | Split periods into first half and second half. `IC_first_half`, `IC_second_half`. Score = `1 - abs(IC_first - IC_second) / (abs(IC_first) + abs(IC_second) + 1e-9)`, clipped to [0, 1]. |
| `market_regime_sharpe` | Compute SPY monthly returns (use yfinance, already in data_retrieval.py). Bull = SPY return > 0, Bear = SPY return ≤ 0. Compute alpha Sharpe in each regime. Score = `bull_sharpe / (abs(bull_sharpe) + abs(bear_sharpe) + 1e-9)`, clipped to [0, 1]. |
| `placebo_score` | Shuffle `forward_returns` randomly 50 times, compute IC each time. `placebo_score = 1 - (mean_placebo_IC / (abs(IC_mean) + 1e-9))`, clipped to [0, 1]. Near 1 = placebo fails → good signal. Near 0 = placebo succeeds → likely noise. |

---

## Updated: `scripts/run_experiment.py`

One change: pass `BacktestResult` (not just `BacktestMetrics`) to `run_robustness`, and extract `metrics` from the result:

```python
backtest_result = run_backtest(alpha)          # returns BacktestResult
metrics = backtest_result["metrics"]           # extract BacktestMetrics for display
robustness = run_robustness(alpha, backtest_result)   # updated signature
```

Everything else (validation, decision, reflection, memory, graph) is unchanged.

Add a `--no-cache` flag to force fresh Snowflake fetch:
```
python scripts/run_experiment.py --config experiments/sample_alpha_001.json --no-cache
```

---

## Updated: `requirements.txt`

```
networkx>=3.0
pyvis>=0.3.2
python-dotenv>=1.0.0
scipy>=1.11
numpy>=1.24
pyarrow>=14.0
yfinance>=0.2
```

(`pandas` and `snowflake-connector-python` are already in `.venv`.)

---

## Implementation Sequence

| Step | File | Notes |
|---|---|---|
| 1 | `requirements.txt` | Add scipy, numpy, pyarrow, yfinance |
| 2 | `core/types.py` | Add `BacktestResult` TypedDict |
| 3 | `core/data_loader.py` | Wraps `SnowflakeDataRetriever`; parquet cache |
| 4 | `core/features.py` | All derived feature computations |
| 5 | `core/formula_eval.py` | Restricted eval with pandas ops |
| 6 | `core/backtest.py` | Full replacement — calls loader → features → eval → simulate |
| 7 | `core/robustness.py` | Full replacement — real sector/subperiod/regime/placebo |
| 8 | `scripts/run_experiment.py` | Update to use `BacktestResult`; add `--no-cache` |

---

## Verification

```bash
# Ensure Snowflake credentials are in .env
cat .env

# Install new deps
pip install -r requirements.txt

# First run — fetches from Snowflake, writes parquet cache (will take ~1-2 min)
python scripts/run_experiment.py --config experiments/sample_alpha_001.json

# Second run — reads from cache (should complete in seconds)
python scripts/run_experiment.py --config experiments/sample_alpha_001.json

# Force fresh fetch
python scripts/run_experiment.py --config experiments/sample_alpha_001.json --no-cache

# Rebuild graph after new run
python scripts/export_graph.py
# Open reports/research_graph.html in browser

# Sanity checks on output:
# - IC_mean should be in [-0.1, 0.2] for a reasonable alpha
# - ICIR should be in [-1, 3]
# - Sharpe should be in [-1, 3]
# - sector_stability, subperiod_stability ∈ [0, 1]
# - placebo_score near 1 means signal is not just noise
```
