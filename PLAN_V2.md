# AlphaResearchBot V2 — Implementation Plan

## Context

V1 proved the architecture end-to-end with everything mocked: `backtest.py` returned
deterministic hash-seeded metrics, `robustness.py` returned hash-seeded scores, and no real
market data was touched. The research loop (validate → backtest → robustness → decide →
reflect → persist → visualize) worked, but produced no meaningful alpha signals.

V2 makes the research engine real:

- Replace mock backtest with a real cross-sectional IC backtest (Spearman correlation of
  signal vs. forward returns, monthly rebalancing, long-short portfolio simulation).
- Replace mock robustness with real checks (sector IC stability, subperiod consistency,
  bull/bear regime Sharpe, placebo test with shuffled returns).
- Add a feature engineering layer that derives alpha signals from raw price and fundamental
  data (momentum, margins, growth, volatility).
- Add a real data layer backed by **free, open-source sources** — no credentials, no cloud
  warehouse. Scope is US stocks only (S&P 500).

Everything from V1 that is not related to data or backtest is carried forward unchanged.

---

## What V2 Adds vs V1

| Module | V1 | V2 |
|---|---|---|
| `core/backtest.py` | Mock hash-based metrics | Real: Spearman IC, long-short simulation |
| `core/robustness.py` | Mock hash-based scores | Real: sector/subperiod/regime/placebo checks |
| `core/features.py` | Does not exist | NEW: derives EBITDA_MARGIN, MOM12_1, VOL_20D, etc. from raw DataFrames |
| `core/formula_eval.py` | Does not exist | NEW: safe `eval()` of formula string against cross-sectional feature Series |
| `core/data_loader.py` | Does not exist | NEW: fetches prices + fundamentals + universe; caches as parquet |
| `core/types.py` | `AlphaConfig`, `BacktestMetrics`, `RobustnessResult` | + `BacktestResult` TypedDict |
| `scripts/run_experiment.py` | Calls `run_backtest` → `BacktestMetrics` | Updated: passes full `BacktestResult` to robustness; `--no-cache` flag |

**Unchanged from V1:** `core/validator.py`, `core/decision.py`, `core/reflection.py`,
`core/memory.py`, `core/graph.py`, `core/visualization.py`, `scripts/export_graph.py`,
SQLite schema.

---

## Implementation Status

Most of V2 is already written. The only remaining piece is `core/data_loader.py`.

| Module | Status |
|---|---|
| `core/types.py` | Done — `BacktestResult` added |
| `core/backtest.py` | Done — real IC backtest |
| `core/robustness.py` | Done — real checks |
| `core/features.py` | Done — feature engine |
| `core/formula_eval.py` | Done — safe evaluator |
| `core/data_loader.py` | **Stub only — needs implementation** |
| `requirements.txt` | Needs `simfin>=0.9.0`, `pandas-datareader>=0.10` added |
| `core/features.py` cleanup | NTM column references should be removed (see §Cleanup) |

---

## Data Sources

Two sources. No API keys required beyond SimFin's free tier.

### yfinance — prices and universe

**What it provides:**
- Daily OHLCV with automatic split and dividend adjustment (`auto_adjust=True`)
- S&P 500 constituent list is fetched separately from Wikipedia (see §Universe)

**Why:** Already a dependency (`yfinance>=0.2`), handles all 503 S&P 500 tickers in a
single batch call, data goes back to the 1990s, zero configuration.

**Covers:** `ADJUSTED_PRICE`, `ADJUSTED_VOLUME`, plus serves as the source for SPY
benchmark returns used in `robustness.py`'s market regime check (already implemented there).

---

### SimFin — fundamentals (with fallback)

**What it provides:**
- Standardized US income statements and cash flow statements from SEC filings
- TTM (trailing twelve months) variant: pre-aggregated rolling four-quarter sums
- Point-in-time `Publish Date`: the date the data became publicly available, which is
  the anchor for forward-filling to avoid look-ahead bias

**Why:** Single `pip install simfin`, bulk CSV download (~50 MB, once), no per-query
rate limits after initial download. Covers Revenue, COGS, Operating Income, EPS, and D&A
(needed to compute EBITDA). Free tier key is the literal string `"free"`.

**Covers:** `SALES_LTM`, `COGS_LTM`, `EBITDA_LTM`, `EPS_LTM`

**Known failure modes on the free tier:**
- Rate limit hit on first bulk download (HTTP 429)
- SimFin servers temporarily unavailable
- Column names change between SimFin package versions
- Partial download leaves a corrupt local CSV

**Fallback behaviour:** If SimFin fails for any reason, `DataLoader` logs a warning and
returns an empty `fundamentals_df`. The pipeline continues with price-only features
(`MOM12_1`, `MOM6_1`, `VOL_20D`, `LIQUIDITY`). Fundamental features (`EBITDA_MARGIN`,
`SALES_GROWTH`, `EPS_GROWTH`, `PRICE_TO_SALES`) simply produce NaN for that run and are
skipped by the backtest engine. See §SimFin Fallback Design for implementation details.

**Setup:**
```python
import simfin as sf
sf.set_api_key("free")          # literal string; no account required for US data
sf.set_data_dir("~/.simfin")   # local CSV cache; downloaded once, ~50 MB total
```

---

### FRED — not needed for V2

`robustness.py` already fetches SPY via yfinance for the market regime check. FRED
(risk-free rate, VIX) is available via `pandas-datareader` and useful for V3 macro
features, but is not required for V2.

---

## Universe

**S&P 500 — ~503 US stocks.**

```python
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(url)[0]
# Columns: Symbol, Security, GICS Sector, GICS Sub-Industry, ...
```

This gives ticker symbols and GICS sector labels directly, no API call needed.
All stocks are US-listed, so `COUNTRY = "US"` for every row.

**Ticker normalization:** Wikipedia uses `.` as separator (e.g., `BRK.B`); yfinance
requires `-` (e.g., `BRK-B`). Apply `ticker.replace(".", "-")` before passing to yfinance.
SimFin uses the original `.` form or its own SimFin ID — handle via its `Ticker` column.

---

## Data Schema Contract

`DataLoader.load()` returns three DataFrames. Column names must match exactly —
`features.py` and `backtest.py` depend on them without modification.

### `prices_df`

| Column | Type | Source |
|---|---|---|
| `FACTSET_ID` | str | Ticker symbol, e.g. `"AAPL"` |
| `DATE` | datetime64 | Trading date |
| `ADJUSTED_PRICE` | float64 | yfinance `Close` with `auto_adjust=True` |
| `ADJUSTED_VOLUME` | float64 | yfinance `Volume` |

### `fundamentals_df`

| Column | Type | Source | Computation |
|---|---|---|---|
| `FACTSET_ID` | str | SimFin `Ticker` | |
| `DATE` | datetime64 | Derived | Forward-filled from SimFin `Publish Date` to every trading day |
| `SALES_LTM` | float64 | SimFin income TTM `Revenue` | Direct column |
| `COGS_LTM` | float64 | SimFin income TTM `Cost of Revenue` | Direct column |
| `EPS_LTM` | float64 | SimFin income TTM `EPS Diluted` | Direct column |
| `EBITDA_LTM` | float64 | Derived | `Operating Income (Loss)` + `Depreciation & Amortization` (from cash flow TTM) |

**NTM columns (`SALES_NTM`, `EPS_NTM`, `EBITDA_NTM`, `COGS_NTM`) are dropped.** Free
sources do not provide analyst consensus estimates. See §Cleanup for the minor code changes
this requires.

### `universe_df`

| Column | Type | Source |
|---|---|---|
| `FACTSET_ID` | str | Wikipedia `Symbol` (normalized) |
| `SECTOR` | str | Wikipedia `GICS Sector` |
| `COUNTRY` | str | Always `"US"` |

---

## `DataLoader` Implementation Spec (`core/data_loader.py`)

### Public interface (unchanged stub → full implementation)

```python
class DataLoader:
    def __init__(self, cache_dir: str = "cache", no_cache: bool = False) -> None

    def load(
        self,
        start_date: str,   # "YYYY-MM-DD"
        end_date: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (prices_df, fundamentals_df, universe_df).
        Reads parquet cache if present; fetches and writes cache on first call.
        """
```

Cache key: `sha256(f"{table}|{start_date}|{end_date}")[:16]` — already implemented in
the stub (`_cache_path` method). No changes needed to cache logic.

### Fetch order and caching

```
load(start_date, end_date)
  │
  ├─ _fetch_universe()              → universe_df, cache: universe_<hash>.parquet
  │    download tickers list (no start/end needed — static table)
  │
  ├─ _fetch_prices(tickers, start, end) → prices_df, cache: prices_<hash>.parquet
  │    yfinance batch download → reshape → rename
  │
  └─ _fetch_fundamentals(tickers, start, end) → fundamentals_df, cache: fundamentals_<hash>.parquet
       SimFin income TTM + cashflow TTM → merge → compute EBITDA → forward-fill → rename
```

### `_fetch_universe()`

```python
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df = pd.read_html(url)[0][["Symbol", "GICS Sector"]]
df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
df = df.rename(columns={"Symbol": "FACTSET_ID", "GICS Sector": "SECTOR"})
df["COUNTRY"] = "US"
```

### `_fetch_prices(tickers, start_date, end_date)`

```python
raw = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    group_by="ticker",
    threads=True,
    progress=False,
)
# raw.columns is MultiIndex: (field, ticker)
# Extract Close and Volume, stack to long form
close = raw["Close"].stack(future_stack=True).reset_index()
close.columns = ["DATE", "FACTSET_ID", "ADJUSTED_PRICE"]
volume = raw["Volume"].stack(future_stack=True).reset_index()
volume.columns = ["DATE", "FACTSET_ID", "ADJUSTED_VOLUME"]
prices_df = close.merge(volume, on=["DATE", "FACTSET_ID"]).dropna()
```

### `_fetch_fundamentals(tickers, start_date, end_date)`

```python
import simfin as sf
sf.set_api_key(os.getenv("SIMFIN_API_KEY", "free"))
sf.set_data_dir(Path.home() / ".simfin")

# Load TTM income and cash flow statements (downloads once, cached locally)
income = sf.load_income(variant="ttm", market="us")
cashflow = sf.load_cashflow(variant="ttm", market="us")

# SimFin returns a MultiIndex DataFrame with (Ticker, Report Date) or similar
# Reset index to get flat columns including "Ticker" and "Publish Date"
income = income.reset_index()
cashflow = cashflow.reset_index()

# Filter to S&P 500 tickers (SimFin uses "." form, universe uses "-" form)
sp500_simfin = [t.replace("-", ".") for t in tickers]
income = income[income["Ticker"].isin(sp500_simfin)]
cashflow = cashflow[cashflow["Ticker"].isin(sp500_simfin)]

# Join income + cashflow on (Ticker, Fiscal Year, Fiscal Period)
merged = income.merge(
    cashflow[["Ticker", "Fiscal Year", "Fiscal Period", "Depreciation & Amortization"]],
    on=["Ticker", "Fiscal Year", "Fiscal Period"],
    how="left",
)

# Derive EBITDA = Operating Income + D&A
merged["EBITDA_LTM"] = (
    merged["Operating Income (Loss)"].fillna(0)
    + merged["Depreciation & Amortization"].fillna(0)
)

# Rename to schema contract
merged = merged.rename(columns={
    "Ticker":         "FACTSET_ID",
    "Publish Date":   "DATE",
    "Revenue":        "SALES_LTM",
    "Cost of Revenue":"COGS_LTM",
    "EPS Diluted":    "EPS_LTM",
})

# Normalize ticker format back to yfinance style
merged["FACTSET_ID"] = merged["FACTSET_ID"].str.replace(".", "-", regex=False)

# Keep only needed columns and date range
cols = ["FACTSET_ID", "DATE", "SALES_LTM", "COGS_LTM", "EPS_LTM", "EBITDA_LTM"]
fund = merged[cols].dropna(subset=["DATE"])
fund["DATE"] = pd.to_datetime(fund["DATE"])
fund = fund[(fund["DATE"] >= start_date) & (fund["DATE"] <= end_date)]

# Forward-fill from each Publish Date to every trading day
trading_days = pd.bdate_range(start=start_date, end=end_date)
fund = (
    fund.set_index(["FACTSET_ID", "DATE"])
    .groupby(level="FACTSET_ID")
    .apply(lambda g: g.droplevel(0).reindex(trading_days).ffill())
    .reset_index()
    .rename(columns={"level_1": "DATE"})
)
```

---

## SimFin Fallback Design

SimFin's free tier can fail silently (corrupt CSV, rate limit, column rename between
package versions). A single SimFin failure must not crash the entire pipeline. The
fallback is: log a warning and return an empty `fundamentals_df`; the run continues
with whichever features can still be computed from prices alone.

### Layer 1 — `DataLoader._fetch_fundamentals` wraps everything in try/except

```python
def _fetch_fundamentals(self, tickers, start_date, end_date) -> pd.DataFrame:
    _EMPTY = pd.DataFrame(columns=["FACTSET_ID", "DATE",
                                    "SALES_LTM", "COGS_LTM", "EPS_LTM", "EBITDA_LTM"])
    try:
        # ... full SimFin fetch logic above ...
        return fund
    except Exception as exc:
        print(
            f"\n  [DataLoader] WARNING: SimFin fetch failed — {exc}\n"
            f"  Continuing with empty fundamentals. Price-only features "
            f"(MOM12_1, MOM6_1, VOL_20D, LIQUIDITY) will still work.\n"
            f"  Fundamental features (EBITDA_MARGIN, SALES_GROWTH, EPS_GROWTH, "
            f"PRICE_TO_SALES) will be skipped for this run.\n"
        )
        return _EMPTY
```

The empty DataFrame is still written to the parquet cache so subsequent runs don't
re-attempt SimFin until the cache is cleared. The cache file will contain zero rows,
which `features.py` handles correctly (all fund_pivots will be empty).

### Layer 2 — `features.py` already skips unavailable features

`_compute_single_feature` already returns `None` when a required pivot is missing.
`compute_features` collects only the non-None panels. If fundamentals are empty,
all fundamental-based features (`EBITDA_MARGIN`, `SALES_GROWTH`, etc.) return `None`
and are silently excluded from the panel — no error thrown at this layer.

The one edge case: if `feature_panels` is completely empty (every requested feature
failed), `compute_features` raises `ValueError("No features could be computed")`.
This only happens when a formula uses **exclusively** fundamental features and SimFin
is down. The fix is to ensure the formula or feature list always includes at least
one price-only feature (see §Price-Only Safe Features below).

### Layer 3 — `backtest.py` receives a reduced feature panel and continues

If the feature panel contains only price-based columns, the backtest loop still runs
normally. IC is computed on whatever signal the formula produces from the available
features. Fundamental features referenced in the formula but absent from the cross-section
cause `evaluate_formula` to raise `NameError` → `ValueError`, which `run_backtest`
catches per-period with `except Exception: continue`. Periods where the formula fails
are skipped; if at least 3 periods produce valid IC values the backtest completes.

---

## Price-Only Safe Features

These features require only `prices_df` (yfinance) and always work even when SimFin
is unavailable. Formulae that reference at least one of these will produce valid
IC estimates in fallback mode.

| Feature | Formula | Already in `features.py`? |
|---|---|---|
| `MOM12_1` | `price[t-21] / price[t-252] - 1` | Yes |
| `MOM6_1` | `price[t-21] / price[t-126] - 1` | Yes |
| `VOL_20D` | 20-day rolling std of log returns | Yes |
| `LIQUIDITY` | 20-day avg of `price × volume` (dollar volume) | **No — add in V2** |

### Add `LIQUIDITY` to `core/features.py`

```python
if name == "LIQUIDITY":
    dollar_vol = price_pivot * volume_pivot
    return dollar_vol.rolling(20).mean()
```

Add `"LIQUIDITY"` to `ALLOWED_FEATURES` in `core/validator.py`.

**Recommended default formula for new experiments:** Include at least `MOM12_1` or
`LIQUIDITY` alongside any fundamental feature. Example:

```
rank(EBITDA_MARGIN) + 0.5 * rank(MOM12_1)   ← safe: MOM12_1 works in fallback
rank(SALES_GROWTH) + rank(EPS_GROWTH)         ← risky: fails entirely if SimFin is down
```

---

## Minor Cleanup Required

### `core/features.py`

Remove NTM columns from `_RAW_FUNDAMENTAL_COLS` since they will never be present in
`fundamentals_df`:

```python
# Before
_RAW_FUNDAMENTAL_COLS = {
    "EPS_LTM", "EPS_NTM", "SALES_LTM", "SALES_NTM",
    "EBITDA_LTM", "EBITDA_NTM", "COGS_LTM", "COGS_NTM",
}

# After
_RAW_FUNDAMENTAL_COLS = {
    "EPS_LTM", "SALES_LTM", "EBITDA_LTM", "COGS_LTM",
}
```

Also update the module docstring to remove the Snowflake reference on line 1.

### `core/validator.py`

Remove NTM features from `ALLOWED_FEATURES` and `FUTURE_LOOKING_FIELDS`, or keep them
in `FUTURE_LOOKING_FIELDS` with an added warning that they are unavailable. The simplest
approach: keep NTM in `FUTURE_LOOKING_FIELDS` so that any formula using them gets a
validation warning ("feature not available in V2 data source"), which prevents silent
failure at runtime.

---

## Updated `requirements.txt`

```
networkx>=3.0
pyvis>=0.3.2
python-dotenv>=1.0.0
scipy>=1.11
numpy>=1.24
pyarrow>=14.0
yfinance>=0.2.40
simfin>=0.9.0
pandas-datareader>=0.10
```

(`pandas-datareader` is included now for FRED access in V3; adds no cost to V2.)

---

## Implementation Sequence

| Step | File | Action |
|---|---|---|
| 1 | `requirements.txt` | Add `simfin>=0.9.0`, `pandas-datareader>=0.10` |
| 2 | `core/data_loader.py` | Implement `_fetch_universe`, `_fetch_prices`, `_fetch_fundamentals` with try/except fallback |
| 3 | `core/features.py` | Remove NTM columns; add `LIQUIDITY`; fix docstring |
| 4 | `core/validator.py` | Add `LIQUIDITY` to `ALLOWED_FEATURES`; add NTM unavailability warning |
| 5 | `cache/` | Delete stale parquet files from any previous runs |

Steps 3–4 are small edits. Step 2 is the primary implementation work.

---

## Verification

```bash
# Install updated dependencies
pip install -r requirements.txt

# First run — fetches from Wikipedia + yfinance + SimFin, writes parquet to cache/
# Expect ~2-5 minutes on first run (SimFin bulk download + yfinance batch)
python scripts/run_experiment.py --config experiments/sample_alpha_001.json

# Second run — reads from parquet cache, expect <15 seconds
python scripts/run_experiment.py --config experiments/sample_alpha_001.json

# Force fresh fetch
rm -rf cache/
python scripts/run_experiment.py --config experiments/sample_alpha_001.json

# Export and open research graph
python scripts/export_graph.py
# Open reports/research_graph.html in browser
```

Expected output sanity ranges:
- `IC_mean` ∈ [-0.10, 0.20] for a reasonable alpha
- `ICIR` ∈ [-1, 3]
- `Sharpe` ∈ [-1, 3]
- `sector_stability`, `subperiod_stability`, `placebo_score` ∈ [0, 1]
- First run: SimFin download progress, then yfinance batch, then backtest output

**Testing the SimFin fallback path:**
```bash
# Temporarily rename ~/.simfin to simulate SimFin unavailability
mv ~/.simfin ~/.simfin_bak
rm -rf cache/

python scripts/run_experiment.py --config experiments/sample_alpha_001.json
# Expected: WARNING about SimFin failure, then backtest completes on price-only features

# Restore
mv ~/.simfin_bak ~/.simfin
```
The backtest must still produce valid output (not crash) when using a formula that
includes at least one price-only feature such as `MOM12_1` or `LIQUIDITY`.
