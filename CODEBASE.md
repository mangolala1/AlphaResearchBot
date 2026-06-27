# AlphaResearchBot — Codebase Reference

A quick-reference guide to every Python file in `core/` and `scripts/`.

---

## How the pieces fit together

```
scripts/run_experiment.py
  │
  ├── core/validator.py          validate formula & config
  ├── core/similarity.py         deduplicate against prior runs
  ├── core/data_loader.py        fetch prices + fundamentals (cached)
  │     └── core/features.py     compute derived signals
  │           └── core/formula_eval.py   evaluate formula string
  ├── core/backtest.py           IC, Sharpe, turnover, drawdown
  ├── core/decision.py           tiered pass / revise / fail rules
  ├── core/robustness.py         sector, subperiod, regime, placebo checks
  ├── core/reflection.py         LLM explanation + next-step suggestion
  └── core/memory.py             persist ExperimentRecord to SQLite
        └── core/memory_analyzer.py    aggregate summaries & failure tags

scripts/mutate_alpha.py
  └── core/mutator.py            LLM → child AlphaConfig (rule-based fallback)

scripts/plan_next.py
  └── core/planner.py            LLM → new research directions (rule-based fallback)
        └── core/memory_analyzer.py

scripts/export_graph.py
  ├── core/graph.py              build NetworkX DAG from experiment store
  └── core/visualization.py     render interactive HTML with vis-network.js
```

---

## `core/` — Library modules

### `types.py`
Defines all shared data contracts as `TypedDict`s and dataclasses. Nothing is computed here — this file is imported by almost every other module.

| Type | Purpose |
|------|---------|
| `AlphaConfig` | Input spec for one experiment (formula, features, universe, dates, etc.) |
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

- **Prices** — `yfinance`: daily adjusted close and volume for S&P 500 tickers
- **Fundamentals** — `SimFin` (TTM): Revenue, EBITDA, EPS, COGS. If SimFin is unavailable or the API key is missing, it returns an empty DataFrame and price-only features still work.
- **Universe** — scraped from Wikipedia's S&P 500 list; includes ticker and GICS sector.

Cache key is a SHA-256 hash of `(table, start_date, end_date)`. Delete the `cache/` directory to force a fresh fetch.

Output column contracts (same names used throughout `features.py` and `backtest.py`):

| DataFrame | Key columns |
|-----------|------------|
| `prices_df` | `TICKER`, `DATE`, `ADJUSTED_PRICE`, `ADJUSTED_VOLUME` |
| `fundamentals_df` | `TICKER`, `DATE`, `SALES_LTM`, `COGS_LTM`, `EPS_LTM`, `EBITDA_LTM` |
| `universe_df` | `TICKER`, `SECTOR`, `COUNTRY` |

---

### `features.py`
Transforms raw price and fundamental DataFrames into a `(DATE, TICKER)` MultiIndex panel of alpha features. Only the features listed in the alpha config's `features` field are computed.

| Feature | Description |
|---------|-------------|
| `MOM12_1` | 12-month price return excluding the last month (skip-1 momentum) |
| `MOM6_1` | 6-month price return excluding the last month |
| `VOL_20D` | 20-day rolling annualised volatility of log returns |
| `LIQUIDITY` | 20-day rolling average dollar volume |
| `EBITDA_MARGIN` | EBITDA / Revenue (LTM) |
| `SALES_GROWTH` | Year-over-year sales growth |
| `EPS_GROWTH` | Year-over-year EPS growth |
| `PRICE_TO_SALES` | Price relative to sales (cross-sectional proxy) |

All derived features are winsorised at the 1%/99% level to reduce outlier noise. Raw columns (`ADJUSTED_PRICE`, `ADJUSTED_VOLUME`, `EPS_LTM`, etc.) are passed through directly if requested.

---

### `formula_eval.py`
Sandboxed evaluator for formula strings. Formulas run inside `eval()` with `__builtins__` disabled and a restricted namespace of cross-sectional operators.

**Allowed operators:** `rank()`, `zscore()`, `log()`, `abs()`, `sign()`

**Blocked operators:** `delta()`, `ts_mean()`, `ts_std()` — these require time-series context and raise `NotImplementedError` with a clear message.

Standard arithmetic (`+`, `-`, `*`, `/`, `**`) and parentheses work as expected. The input is a dict of `{feature_name: pd.Series}` one value per stock. The output is a signal `pd.Series` indexed by `TICKER`.

---

### `validator.py`
Validates an `AlphaConfig` before any computation runs. Returns a `ValidationResult` with lists of errors (blocking) and warnings (non-blocking).

Checks performed:
1. **Required fields** — `alpha_id`, `formula`, `features`, `universe`, `start_date`, `end_date`
2. **Feature allowlist** — every feature must be in `ALLOWED_FEATURES`; NTM (forward-looking) features produce a warning
3. **Formula tokens** — every identifier in the formula must be a known feature or allowed operator
4. **Feature/formula consistency** — warns if features are declared but not in the formula or vice versa
5. **Universe** — must be one of `sp500`, `russell1000`, `russell3000`
6. **Date ordering** — `start_date` must be before `end_date`, both in `YYYY-MM-DD` format

---

### `backtest.py`
Core simulation engine. Runs a monthly-rebalanced long-short backtest and returns both summary metrics and intermediate data needed for robustness checks.

**Pipeline per rebalancing period:**
1. Pull the cross-section of features at each month-end date
2. Evaluate the formula via `formula_eval.py` → signal
3. Winsorise and z-score the signal
4. Compute Spearman IC against next-month forward returns
5. Split into quintiles; simulate Q5 − Q1 long-short portfolio return
6. Compute per-sector IC (for robustness)

**Metrics computed:**
- `IC_mean` and `ICIR` (IC / IC std)
- `Sharpe` (annualised, monthly periods)
- `monotonicity` (avg Spearman rho between quintile rank and quintile return)
- `turnover` (avg constituent replacement rate in top/bottom quintiles)
- `max_drawdown`
- `deflated_sharpe` (Sharpe adjusted for multiple testing)
- `noise_risk` (`low` / `medium` / `high` based on deflated/raw Sharpe ratio)

Raises `RuntimeError` if fewer than 3 valid periods are produced (e.g., date range too short or feature data missing).

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
Step 1   Validate formula and config
Step 1.5 Similarity check against prior alphas
Step 2   Backtest (data load + signal + IC + portfolio)
Step 3   Tier 1 decision (predictive power)
Step 4   Robustness checks (sector, subperiod, regime, placebo)
Step 5   Tier 2 decision (implementation)
Step 6   Generate LLM reflection
Step 7   Save ExperimentRecord to SQLite
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
