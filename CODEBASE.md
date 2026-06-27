# AlphaResearchBot — Codebase Reference

A quick-reference guide to every Python file in `core/` and `scripts/`.

---

## How the pieces fit together

```
scripts/run_experiment.py
  │
  ├── core/formula_validator.py   validate config, raw_formula, build eval namespace
  ├── core/similarity.py          deduplicate against prior runs
  ├── core/data_loader.py         fetch prices / income TTM / cashflow TTM (cached separately in cache/)
  │     └── core/backtest.py      process each statement separately → data/; join → signal
  │           ├── core/data_process.py   rename + winsorise + standardise + ffill → (DATE, TICKER)
  │           └── core/signal_calculation.py   eval raw_formula → signal Series
  │                 └── core/formula_validator.py   build_panel_namespace()
  ├── core/decision.py            tiered pass / revise / fail rules
  ├── core/robustness.py          sector, subperiod, regime, placebo checks
  ├── core/reflection.py          LLM explanation + next-step suggestion
  └── core/memory.py              persist ExperimentRecord to SQLite
        └── core/memory_analyzer.py    aggregate summaries & failure tags

scripts/mutate_alpha.py
  └── core/mutator.py             LLM → child AlphaConfig (rule-based fallback)

scripts/plan_next.py
  └── core/planner.py             LLM → new research directions (rule-based fallback)
        └── core/memory_analyzer.py

scripts/export_graph.py
  ├── core/graph.py               build NetworkX DAG from experiment store
  └── core/visualization.py       render interactive HTML with vis-network.js
```

---

## `core/` — Library modules

### `types.py`
Defines all shared data contracts as `TypedDict`s and dataclasses. Nothing is computed here — this file is imported by almost every other module.

| Type | Purpose |
|------|---------|
| `AlphaConfig` | Input spec for one experiment. Key fields: `formula` (display label, e.g. `rank(MOM12_1)`), `raw_formula` (executed expression using raw column names, e.g. `rank(ADJUSTED_PRICE.shift(21) / ADJUSTED_PRICE.shift(252) - 1)`), `universe`, `start_date`, `end_date` |
| `BacktestMetrics` | Output metrics: IC_mean, ICIR, Sharpe, turnover, monotonicity, max_drawdown, deflated_sharpe, noise_risk |
| `BacktestResult` | Full backtest output including IC series, portfolio returns, sector IC, and per-period signal values |
| `RobustnessResult` | Sector stability, subperiod stability, market-regime Sharpes, placebo score |
| `ExperimentRecord` | Everything persisted to SQLite: config + metrics + robustness + verdict + reflection |
| `ValidationResult` | Dataclass holding `valid` bool, `errors` list, `warnings` list |
| `SimilarityResult` | Jaccard similarity score against the most-similar prior alpha |
| `ResearchSuggestion` | One planner suggestion: direction, hypothesis, formula, features, rationale |
| `MemorySummary` | Aggregated view of all experiments: counts, best runs, unexplored features, trend observations |
| `Verdict` | Literal `"promising" | "revise" | "failed"` |
| `FailureCategory` | Literal for dominant failure mode: `high_turnover`, `weak_ic`, `negative_sharpe`, `high_noise`, `poor_robustness` |

---

### `data_loader.py`
Fetches raw market data and caches it locally as parquet files so remote sources are only hit once per date range.

- **Prices** — `yfinance`: daily adjusted close and volume for all S&P 500 tickers.
- **Income TTM** — `SimFin` free tier, trailing-twelve-month income statement. Cached independently as `cache/income_ttm_*.parquet`.
- **Cashflow TTM** — `SimFin` free tier, trailing-twelve-month cash flow statement. Cached independently as `cache/cashflow_ttm_*.parquet`.
- **Universe** — scraped from Wikipedia's S&P 500 list; includes ticker and GICS sector.

Income and cashflow are **never merged** here — each statement is fetched, cleaned, and cached separately. The merge happens downstream in `backtest.py` after both have been processed.

Each SimFin statement goes through the same `_fetch_simfin_TTM` → `_prefetch_and_ffill` pipeline:
1. Load from SimFin, filter to S&P 500 tickers
2. Rename only `Ticker → TICKER` and `Publish Date → DATE` (all further renaming is done in `data_process.py`)
3. Drop SimFin metadata columns (`Fiscal Year`, `Fiscal Period`, `Currency`, `Report Date`, `Restated Date`, `SimFinId`)
4. Drop sparse columns (>1000 NaN rows)
5. Pre-fetch 1 year before `start_date` to seed the forward-fill, forward-fill to business-day frequency, trim back to `start_date`

Cache key is a SHA-256 hash of `(table, start_date, end_date)`. Delete `cache/` to force a fresh fetch.

`load()` returns a **4-tuple**: `(prices_df, income_ttm_df, cashflow_ttm_df, universe_df)`.

| DataFrame | Key columns |
|-----------|------------|
| `prices_df` | `TICKER`, `DATE`, `ADJUSTED_PRICE`, `ADJUSTED_VOLUME` |
| `income_ttm_df` | `TICKER`, `DATE`, original SimFin income column names (renamed in `data_process.py`) |
| `cashflow_ttm_df` | `TICKER`, `DATE`, original SimFin cashflow column names (renamed in `data_process.py`) |
| `universe_df` | `TICKER`, `SECTOR`, `COUNTRY` |

---

### `data_process.py`
Cleans, winsorises, standardises, and forward-fills a single SimFin statement into a `(DATE, TICKER)` MultiIndex DataFrame. Called separately for income and cashflow — statements are never merged here.

**`processed_path(statement, start_date, end_date) → Path`** — returns `data/processed_{statement}_{hash}.parquet`. Canonical cache location for processed output; shared by `backtest.py` and the `__main__` block.

**`process(df, value_cols, winsorise, standardise, ffill_daily)`** — main entry point:
1. Apply `_COLUMN_RENAMES` — maps original SimFin names (e.g. `"Revenue"`) to standardised identifiers (e.g. `REVENUE_LTM`)
2. Sanitise any remaining non-standard column names (spaces/special chars → underscores, uppercased) via `_sanitize_col()`
3. Normalise to `(DATE, TICKER)` MultiIndex
4. Drop dates where every stock is NaN for a column
5. Cross-sectional winsorisation — clip at 1%/99% **across stocks per date**
6. Cross-sectional z-score standardisation — subtract cross-stock mean, divide by cross-stock std per date
7. Drop `(date, ticker)` rows with any remaining NaN in value columns
8. Forward-fill to business-day frequency via `_ffill_to_daily()`

`_COLUMN_RENAMES` covers all income and cashflow SimFin column names. `_NON_VALUE_COLS = {"SECTOR", "COUNTRY"}` are carried through unchanged. `_PRICE_COLS = {"ADJUSTED_PRICE", "ADJUSTED_VOLUME"}` are excluded from winsorisation/standardisation.

`_DATA_DIR = Path("data")` — module-level constant imported by `backtest.py` so processed parquets always go to the same location.

Run `python core/data_process.py` to manually process both statements from cache, save to `data/`, and print shape, columns, date range, NaN counts, and a sample.

---

### `formula_validator.py`
Single owner of all formula-related logic: config validation, operator definitions, namespace construction, and formula evaluation. Every other module that needs to evaluate or validate a formula imports from here.

**Constants:**

| Name | Purpose |
|------|---------|
| `AVAILABLE_RAW_COLUMNS` | frozenset of 8 raw column names valid in `raw_formula`: `ADJUSTED_PRICE`, `ADJUSTED_VOLUME`, `SALES_LTM`, `COGS_LTM`, `NET_INCOME_LTM`, `OPER_INCOME_LTM`, `DA_LTM`, `SHARES_DILUTED` |
| `ALLOWED_FUNCTION_NAMES` | `rank`, `zscore`, `log`, `abs`, `sign`, `delta`, `ts_mean`, `ts_std` |

`ALLOWED_FEATURES`, `EVALUATOR_FEATURES`, and `FUTURE_LOOKING_FIELDS` have been removed — the formula system is fully raw-column-based.

**`validate_alpha(alpha)`** — validates an `AlphaConfig`, returns `ValidationResult`:
1. Required fields: `alpha_id`, `raw_formula`, `universe`, `start_date`, `end_date`
2. `raw_formula` — checks parenthesis balance, verifies at least one `AVAILABLE_RAW_COLUMNS` column is referenced, warns on unknown ALL_CAPS identifiers
3. `formula` is optional (display label only, never validated or evaluated)
4. Universe and date ordering

**`build_panel_namespace(processed_df)`** — builds the eval namespace for `raw_formula`. Each column of `processed_df` is pivoted to a full `DATE × TICKER` wide DataFrame. The namespace also includes:
- `float` and `nan` — explicitly whitelisted so `float('nan')` works inside the sandboxed eval
- Cross-sectional operators: `rank(X)`, `zscore(X)`, `log(X)`, `abs(X)`, `sign(X)`
- `np` for numpy math
- `delta`, `ts_mean`, `ts_std` raise `NotImplementedError` directing users to pandas methods

---

### `signal_calculation.py`
Execution layer for alpha signal generation. Wires `formula_validator.build_panel_namespace` and `data_process._winsorise/_standardise` together to produce the signal Series consumed by `backtest.py`.

**`compute_signal(processed_df, raw_formula, post_winsorise, post_standardise)`**:
1. Call `build_panel_namespace(processed_df)` → namespace of `DATE × TICKER` DataFrames
2. `eval(raw_formula, namespace)` → `DATE × TICKER` signal DataFrame
3. Apply cross-sectional winsorise + standardise to the result
4. Stack to `(DATE, TICKER)` MultiIndex `pd.Series` named `"signal"`, drop NaNs

Each `AlphaConfig` carries two formula strings:
- `raw_formula` — executed here; uses raw column names and pandas operations (e.g. `rank(ADJUSTED_PRICE.shift(21) / ADJUSTED_PRICE.shift(252) - 1) + 0.5 * rank(OPER_INCOME_LTM / SALES_LTM)`)
- `formula` — display only; human-readable shorthand shown in the UI (e.g. `rank(MOM12_1) + 0.5 * rank(EBITDA_MARGIN)`); never evaluated here

---

### `backtest.py`
Core simulation engine. Orchestrates data processing, signal computation, and the monthly-rebalanced long-short simulation.

**Data pipeline (run once per `(start_date, end_date)`):**
1. Unpack 4-tuple from `data_loader.load()` → `prices_df`, `income_ttm_df`, `cashflow_ttm_df`, `universe_df`
2. Process income statement independently via `data_process.process()` → cache to `data/processed_income_*.parquet`
3. Process cashflow statement independently via `data_process.process()` → cache to `data/processed_cashflow_*.parquet`
4. Outer-join processed income + cashflow on `(DATE, TICKER)` index
5. Add `SECTOR` from universe and raw `ADJUSTED_PRICE` / `ADJUSTED_VOLUME` from prices

**Signal and backtest pipeline per rebalancing period:**
1. Evaluate `raw_formula` via `signal_calculation.py` over the full date range at once → signal Series
2. At each month-end rebalancing date, slice signal and compute Spearman IC against next-month forward returns
3. Split into quintiles; simulate Q5 − Q1 long-short portfolio return
4. Compute per-sector IC (for robustness)

Cache paths come from `data_process.processed_path(statement, start_date, end_date)`. `--no-cache` skips the processed parquet cache and rebuilds from the raw `cache/` files.

**Metrics computed:**
- `IC_mean` and `ICIR` (IC / IC std)
- `Sharpe` (annualised, monthly periods)
- `monotonicity` (avg Spearman rho between quintile rank and quintile return)
- `turnover` (avg constituent replacement rate in top/bottom quintiles)
- `max_drawdown`
- `deflated_sharpe` (López de Prado SR adjusted for skewness and kurtosis)
- `noise_risk` (`low` / `medium` / `high` based on deflated/raw Sharpe ratio)

Raises `RuntimeError` if fewer than 3 valid periods are produced.

---

### `decision.py`
Rule-based, two-tier verdict system. All thresholds are defined as module-level constants (easy to adjust) and are mirrored in the HTML visualisation.

**Tier 1 — Predictive Power** (checked first; hard fail skips robustness entirely):
- Hard fail: `IC_mean ≤ 0`, `ICIR ≤ 0`, or `monotonicity ≤ -0.2`
- Soft fail (→ `revise`): `IC_mean ≤ 0.02`, `ICIR ≤ 0.30`, or `monotonicity ≤ 0.30`

**Tier 2 — Implementation** (checked only if Tier 1 passed or soft-failed):
- Hard fail: `Sharpe ≤ 0` or `max_drawdown ≤ -0.40`
- Soft fail (→ `revise`): `Sharpe ≤ 0.50`, `turnover ≥ 0.70`, or `max_drawdown ≤ -0.25`
- Otherwise → `promising`

A Tier 1 soft-fail overrides a Tier 2 `promising` verdict, keeping the result as `revise`.

---

### `robustness.py`
Computes four robustness diagnostics from the intermediate data returned by `backtest.py`. All checks use real data — no simulation.

| Diagnostic | What it measures |
|-----------|-----------------|
| `sector_stability` | Mean IC per GICS sector; reveals if the signal is sector-concentrated |
| `subperiod_stability` | IC consistency between the first and second halves of the period; detects regime-specific alphas |
| `market_regime_sharpe` | Annualised Sharpe split across five regimes (bull/bear, high/low/neutral VIX) using SPY and VIX from yfinance |
| `placebo_score` | Shuffles forward returns within each period and computes IC against the real signal; score near 1.0 means real IC is well above chance |

SPY and VIX data are fetched once and cached in a module-level dict for the process lifetime.

---

### `reflection.py`
Produces a structured written analysis of an experiment after the verdict is decided. Tries the DeepSeek LLM first; falls back to rule-based templates if the API key is missing or the call fails.

**Output format** (both LLM and fallback):
```
Observation: <what the numbers show>
Failure Reason: <from decision.py>
Possible Explanation: <why the alpha performed this way>
Next Mutation: <one concrete change to try>
```

LLM output is prefixed with a disclaimer: `[DISCLAIMER: LLM-generated hypothesis, not validated evidence]`. The rule-based fallback inspects feature types (momentum, quality, growth) and metric values to generate templated observations.

---

### `mutator.py`
Generates a child `AlphaConfig` by mutating a parent experiment. Reads the parent's metrics, robustness results, and prior reflection, then asks DeepSeek to propose one targeted change.

**LLM path:** Sends parent details (formula, features, verdict, failure reason, metrics, reflection) to `deepseek-chat` and parses a JSON `AlphaConfig`. The child is validated before being returned; invalid output raises an exception that triggers the fallback.

**Rule-based fallback** (priority order):
1. High turnover → switch to quarterly rebalance
2. Weak Sharpe + no quality → add `EBITDA_MARGIN` overlay
3. Low ICIR → add `VOL_20D` damper `(1 - rank(VOL_20D))`
4. Has momentum but no quality → add `EBITDA_MARGIN`
5. Default → add `LIQUIDITY` screen

The child gets a new `alpha_id` formatted as `{parent_id}_mut_{timestamp}` and has `parent_id` set so the graph edge is created correctly.

---

### `planner.py`
Surveys all past experiments (via `memory_analyzer.py`) and proposes N new alpha research directions. Prioritises signals that haven't been tried yet.

**LLM path:** Sends a `MemorySummary` (verdict counts, failure modes, best experiments, explored/unexplored features) to DeepSeek and parses a JSON array of `ResearchSuggestion`s. Each suggestion is validated before being accepted.

**Rule-based fallback:** Five pre-defined suggestions covering value, quality+value, earnings growth, low volatility, and liquidity+momentum. Suggestions whose features haven't been explored are preferred over those that overlap with prior experiments.

---

### `memory.py`
SQLite persistence layer. All experiment records are stored in a single `experiments` table. JSON fields (`features`, `config`, `metrics`, `robustness`) are serialised with `json.dumps` and deserialised on read.

Key methods on `ExperimentStore`:
- `save_experiment(record)` — INSERT OR REPLACE by `alpha_id`
- `load_all()` — returns all records ordered by `timestamp`
- `load_by_id(alpha_id)` — returns one record or `None`

Database path defaults to `db/experiments.db` and is created automatically if missing.

---

### `memory_analyzer.py`
Aggregates `ExperimentStore` records into a `MemorySummary` and classifies failure modes.

`classify_failure(record)` assigns one `FailureCategory` per experiment (highest-priority match wins):
`high_turnover` → `negative_sharpe` → `weak_ic` → `high_noise` → `poor_robustness`

`analyze_memory(store)` returns:
- Verdict and failure-category counts
- Top-3 promising experiments by Sharpe
- Which features have been explored vs. unexplored
- Trend observations (dominant failure mode, best Sharpe, unexplored signals)

---

### `similarity.py`
Prevents re-running near-duplicate alphas. Compares a new alpha against every prior experiment using a combined Jaccard similarity score:

```
score = 0.5 × feature_jaccard + 0.5 × formula_token_jaccard
```

A +0.1 bonus is added if `universe`, `rebalance`, and `neutralization` all match. If the best score exceeds the threshold (default `0.80`), the alpha is flagged as a duplicate and `run_experiment.py` exits unless `--force` is passed.

---

### `graph.py`
Builds a `networkx.DiGraph` where each node is an alpha experiment and each directed edge is a parent → child mutation relationship. Node attributes include all metrics and metadata so the visualisation can render them without querying the database again.

`ResearchGraph.build_from_store(store)` loads all records in one pass and calls `memory_analyzer.classify_failure` to tag each node with its failure category.

---

### `visualization.py`
Exports the research graph as a fully self-contained, interactive HTML file using the [vis-network.js](https://visjs.github.io/vis-network/) library (loaded from CDN).

Features of the exported HTML:
- **Summary bar** — total experiments, promising / revise / failed counts, best Sharpe, best IC
- **Hierarchical graph** — nodes colour-coded by verdict (green = promising, amber = revise, red = failed); failed nodes have dashed borders
- **Retractable detail panel** — clicking a node slides in a 360 px right panel with hypothesis, formula, all metrics (with warning indicators against the tier thresholds), failure reason, mutation, and reflection; clicking empty canvas or the ✕ button closes it and re-fits the graph
- **Light mode, Inter font** — readable at a glance without a dark background

Node and edge data are serialised as JSON and embedded directly in the HTML so the file works offline.

---

## `scripts/` — CLI entry points

### `run_experiment.py`
**Main end-to-end pipeline.** Accepts a JSON config file and runs the full experiment in seven labelled steps:

```
Step 1   Validate formula and config       (formula_validator.py)
Step 1.5 Similarity check                  (similarity.py)
Step 2   Load + process data               (data_loader.py → data_process.py)
Step 3   Compute signal                    (signal_calculation.py)
Step 4   Backtest                          (backtest.py)
Step 5   Tier 1 + Tier 2 decision          (decision.py)
Step 6   Robustness checks                 (robustness.py)
Step 7   Generate LLM reflection           (reflection.py)
Step 8   Save ExperimentRecord to SQLite   (memory.py)
```

**Usage:**
```bash
python scripts/run_experiment.py --config experiments/sample_alpha_001.json
python scripts/run_experiment.py --config experiments/my_alpha.json --no-cache --force
```

Flags: `--no-cache` bypasses the parquet cache; `--force` skips the similarity gate.

---

### `mutate_alpha.py`
Looks up a parent experiment by `alpha_id`, calls `core/mutator.py` to generate a child config, saves it as `experiments/{child_id}.json`, and optionally runs it immediately via a subprocess call to `run_experiment.py`.

**Usage:**
```bash
python scripts/mutate_alpha.py --parent alpha_001
python scripts/mutate_alpha.py --parent alpha_001 --run
```

---

### `plan_next.py`
Reads all experiments from the database, summarises what has been tried, and uses `core/planner.py` to suggest N new alpha directions. Optionally saves each suggestion as a ready-to-run JSON config in `experiments/`.

**Usage:**
```bash
python scripts/plan_next.py
python scripts/plan_next.py --n 5
python scripts/plan_next.py --n 3 --save
```

After saving, each file can be run directly with `run_experiment.py`.

---

### `export_graph.py`
Loads all experiments from SQLite, builds the `ResearchGraph`, and calls `core/visualization.py` to write a self-contained HTML file.

**Usage:**
```bash
python scripts/export_graph.py
python scripts/export_graph.py --db db/experiments.db --output reports/research_graph.html
```

Default output: `reports/research_graph.html`.
