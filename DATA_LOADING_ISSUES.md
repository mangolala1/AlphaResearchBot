# Data Loading Issues — Root Causes and Fixes

> Version: June 2026  
> Context: The `run_experiment.py` pipeline force-quits 4+ times during the data loading stage (Step 2: backtest). This document identifies why and proposes concrete fixes and alternative data paths.

---

## 1. What Happens During a Run

```
run_experiment.py
  └── run_backtest()           ← backtest.py
        └── DataLoader.load()  ← data_loader.py
              ├── get_universe_data()           → ~820 KB parquet (cached, fine)
              ├── get_prices_data(factset_ids)  ← FORCE QUIT HERE
              └── get_fundamentals_data(...)    ← or here
        └── compute_features()   ← features.py (secondary kill point)
```

The process is killed by macOS's OOM killer (no crash log, just terminates) because RAM spikes beyond what is available.

---

## 2. Root Causes

### 2.1 `cursor.fetchall()` pulls every row into RAM at once

**File:** `Data/data_retrieval.py` — `execute_query()` (line 68)

```python
results = cursor.fetchall()  # entire result set materialised in Python memory
df = pd.DataFrame(results, columns=...)
```

For prices over 2015–2025 for thousands of FactSet IDs, this can be hundreds of millions of rows. `fetchall()` blocks until everything is received and holds the full result in memory twice (once as a list of tuples, once as a DataFrame) before the list can be garbage-collected.

**Fix:** Replace with `fetchmany(batch_size)` and build the DataFrame incrementally, or use `cursor.fetch_pandas_all()` (Snowflake connector native method) which is more memory-efficient.

---

### 2.2 `DataLoader.load()` ignores `MAX_STOCKS`

**File:** `core/data_loader.py` — `load()` (line 39)

```python
factset_ids = universe["FACTSET_ID"].dropna().unique().tolist()  # all stocks — could be 5000+
prices = retriever.get_prices_data(start_date, end_date, factset_ids=factset_ids)
```

The `MAX_STOCKS` cap in `Data/config.py` is only applied in `get_all_data()` (an older path not called by the current pipeline). `DataLoader.load()` calls `get_prices_data` with the full universe and no limit. For a 10-year date range and 5000+ stocks, the prices query alone can return 10–50 million rows.

**Fix:** Apply `MAX_STOCKS` (or a user-configured `max_stocks` parameter) inside `DataLoader.load()` before the prices/fundamentals fetch.

---

### 2.3 Chunk accumulation doesn't actually save memory

**File:** `Data/data_retrieval.py` — `get_prices_data()` and `get_fundamentals_data()` (lines 313–349, 162–196)

```python
chunks = [factset_ids[i:i+1000] for i in range(0, len(factset_ids), 1000)]
all_results = []
for chunk in chunks:
    chunk_result = self.execute_query(query)
    all_results.append(chunk_result)  # accumulates all chunks in memory
return pd.concat(all_results, ignore_index=True)  # full dataset in RAM
```

Chunking the Snowflake query only reduces individual query size; `all_results` still holds the full dataset in memory by the time `pd.concat` runs. With 50 chunks × 1 million rows each, this is 50 million rows in RAM simultaneously.

**Fix:** Write each chunk to parquet incrementally and read back at the end, or stream directly into a single parquet file using `pyarrow`.

---

### 2.4 Multiple full-universe pivot tables in `compute_features`

**File:** `core/features.py` — `compute_features()` (lines 36–48)

```python
price_pivot  = prices_df.pivot_table(index="DATE", columns="FACTSET_ID", values="ADJUSTED_PRICE")
volume_pivot = prices_df.pivot_table(index="DATE", columns="FACTSET_ID", values="ADJUSTED_VOLUME")
# Plus up to 8 more fundamentals pivots, one per column
```

For N=5000 stocks × T=2500 trading days (10 years), each pivot is a 5000×2500 float64 DataFrame = ~100 MB. With 10 pivots that's ~1 GB from pivots alone, before any feature computation. All pivots coexist in memory simultaneously.

**Fix:** Compute features column-by-column and immediately drop each intermediate pivot after use, or use a lazy/chunked computation via `polars` or `dask`.

---

### 2.5 No keepalive / connection timeout during long fetches

**File:** `Data/data_retrieval.py` — `connect()` (line 32)

```python
self.conn = snowflake.connector.connect(..., login_timeout=120)
```

`login_timeout=120` sets the connection establishment timeout but not the query execution timeout. Queries fetching millions of rows can run for 10–30 minutes, during which the Snowflake warehouse may suspend, returning partial results or errors that silently corrupt the DataFrame.

**Fix:** Add `network_timeout` and `socket_timeout` parameters; implement query result caching at the Snowflake layer (result cache is free and automatic within 24 hours for identical queries).

---

### 2.6 `get_prices_data_by_date_range` — no filter at all

**File:** `Data/data_retrieval.py` — `get_prices_data_by_date_range()` (line 421) and `get_fundamentals_data_by_date_range()` (line 384)

These two functions fetch the **entire global universe** for every date in the range with no stock filtering. They are called by `load_raw_data_for_model()` (line 462). If this path is ever triggered it will certainly OOM.

**Fix:** These functions should be removed or gated; all fetch paths should go through the filtered `get_prices_data(factset_ids=...)` variant.

---

## 3. Immediate Fixes (Minimal Code Changes)

### Fix A — Apply stock limit in `DataLoader.load()`

In `core/data_loader.py`, after retrieving the universe, sample stocks before fetching prices:

```python
factset_ids = universe["FACTSET_ID"].dropna().unique().tolist()
max_stocks = 500  # or pull from config
if len(factset_ids) > max_stocks:
    import random; random.seed(42)
    factset_ids = random.sample(factset_ids, max_stocks)
    print(f"  [DataLoader] Sampled {max_stocks} stocks from universe of {len(factset_ids)}")
```

### Fix B — Use Snowflake's native pandas fetch

Replace `execute_query` to use `fetch_pandas_all()` which avoids the Python list intermediate:

```python
cursor.execute(query)
df = cursor.fetch_pandas_all()  # uses Arrow under the hood, ~3× less RAM
cursor.close()
return df
```

Requires `snowflake-connector-python[pandas]` (already likely available).

### Fix C — Write chunks to parquet during fetch

In the chunked loops in `get_prices_data` and `get_fundamentals_data`, write each chunk to a temp parquet file, then read them all back with `pd.read_parquet`:

```python
import tempfile, os
tmp_files = []
for chunk in chunks:
    df_chunk = self.execute_query(query_for_chunk)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df_chunk.to_parquet(f.name, index=False)
        tmp_files.append(f.name)

result = pd.read_parquet(tmp_files)  # reads lazily from disk
for f in tmp_files: os.unlink(f)
return result
```

### Fix D — Free memory between pivots in `compute_features`

In `core/features.py`, instead of pre-building all pivots upfront, compute and drop each one inline:

```python
feature_panels: dict[str, pd.DataFrame] = {}
for feat in feature_names:
    panel = _compute_single_feature(feat, prices_df, fundamentals_df, universe_meta)
    if panel is not None:
        feature_panels[feat] = panel
    # No global price_pivot or volume_pivot kept in memory
```

And use `dtype=np.float32` when building pivots to halve memory vs. float64.

---

## 4. Alternative Data Sources (No Snowflake Required)

The project's research goal — testing alpha factor hypotheses via backtest — does not require the specific FactSet Snowflake tables. The following free or low-cost sources cover the same data needs.

### 4.1 yfinance (already imported)

**What it provides:** OHLCV + adjusted prices, market cap, basic fundamentals (P/E, EPS TTM, revenue), sector/industry metadata.  
**Coverage:** US stocks (S&P 500, Russell 1000/3000), daily data, 20+ years of history.  
**Limits:** Rate-limited; batch-fetch up to ~500 tickers cleanly; no intraday.

```python
import yfinance as yf

tickers = ["AAPL", "MSFT", ...]  # S&P 500 list
data = yf.download(tickers, start="2015-01-01", end="2025-01-01",
                   auto_adjust=True, group_by="ticker")
# prices: data["AAPL"]["Close"], etc.
```

**Best for:** Quick iteration on US-only price-based alphas (momentum, volatility, volume factors).

---

### 4.2 OpenBB Platform

**What it provides:** Unified API over many providers (Yahoo, FMP, Polygon, SEC EDGAR, etc.).  
**Coverage:** Prices, fundamentals (quarterly/annual from SEC), macro data, ETF flows.  
**Limits:** Free for most basic endpoints; some providers need API keys.

```bash
pip install openbb
```

```python
from openbb import obb
prices = obb.equity.price.historical("AAPL", start_date="2015-01-01")
fundamentals = obb.equity.fundamental.income("AAPL", period="annual")
```

**Best for:** Replacing both price and fundamentals Snowflake tables with a single library call, while keeping the same DataFrame interface.

---

### 4.3 SimFin (free academic tier)

**What it provides:** Standardised income statements, balance sheets, cash flows for 3000+ US companies; quarterly and annual.  
**Coverage:** 1995–present; maps to ticker or SimFin ID.  
**Limits:** Daily/bulk download is free with API key; rate limit 2000 calls/day.

```bash
pip install simfin
```

```python
import simfin as sf
sf.set_api_key("YOUR_FREE_KEY")
sf.set_data_dir("~/simfin_data")
income_df = sf.load_income(variant="annual", market="us")
```

**Best for:** Replacing `EDS_FACTORS_FUNDAMENTALS_NTM_LTM` (SALES, EBITDA, EPS) with standardised SEC-derived data.

---

### 4.4 Polygon.io (free tier)

**What it provides:** Daily OHLCV + adjusted prices, ticker metadata, corporate actions (splits/dividends).  
**Coverage:** All US equities; 2+ years free, unlimited via paid tier.  
**Limits:** Free tier = 5 API calls/min; bulk flat-file access on paid tier.

**Best for:** Split-adjusted prices matching `SPLIT_ADJUSTED_PRICES_EDS_FACTORS`.

---

### 4.5 FRED (via `pandas_datareader`)

**What it provides:** Macro factors — risk-free rates, VIX, exchange rates (vs USD), S&P 500 index.  
**Coverage:** Daily to monthly, decades of history; completely free.

```python
import pandas_datareader as pdr
fx = pdr.DataReader("DEXUSEU", "fred", start="2015-01-01")  # USD/EUR exchange rate
```

**Best for:** Replacing `EXCHANGERATES` and getting benchmark (S&P 500) returns.

---

## 5. Recommended Approach for Next Version

| Goal | Approach |
|------|----------|
| Fast local iteration | yfinance for S&P 500 prices + SimFin for fundamentals; cache as parquet on first run |
| Eliminate OOM kills | Apply `MAX_STOCKS=500` cap in `DataLoader.load()`; use `fetch_pandas_all()` in Snowflake path |
| Keep Snowflake as source | Fix chunking to write temp parquet per chunk; use Arrow fetch |
| Scale to full global universe | Run on a machine with ≥32 GB RAM or process data in rolling windows and cache monthly feature panels |

A minimal yfinance-backed `DataLoader` that matches the existing interface (`prices_df`, `fundamentals_df`, `universe_df`) would let the rest of the pipeline (`backtest.py`, `features.py`) run unchanged while eliminating the Snowflake dependency for development and testing.

---

## 6. Files Modified / To Modify

| File | Issue | Priority |
|------|-------|----------|
| `core/data_loader.py` | Missing `MAX_STOCKS` cap before fetching prices | High |
| `Data/data_retrieval.py` | `fetchall()` OOM; no Arrow fetch; unsafe unfiltered `_by_date_range` methods | High |
| `core/features.py` | Simultaneous full-universe pivots; should be built one at a time | Medium |
| `Data/config.py` | `MAX_STOCKS=500` is set but not wired to the live fetch path | Medium |

## 7. Next Steps
1. Identify what free data sources are out there, and what data columns and years of history as well as granularity data am i able to retrieve as a free user? 
- Databento, yfinance, OpenBB, SimFin, Polygon.io, FRED, etc.
2. Refer to the MCP established in financial services released by Claude, how does it work?
- MCP is merely a protocol that lets Claude call tools.
- Two ways to use MCP to retrieve data: 
- (1) Online every time: process data in memory and return result, can be slower and rate-limited, but no storage needed.
- (2) Download + cache: check cache, if missing, download data once, store as parquet, and read from disk on subsequent runs. Faster after first run, but requires storage and cache management.