# AlphaResearchBot — Codebase Reference

A quick-reference guide to every Python file in `core/`, `scripts/`, and `tests/`.

---

## How the pieces fit together

```
scripts/run_loop.py                 ── V4 autonomous loop ──
  ├── core/scheduler.py             Thompson bandit: explore vs mutate:<parent>
  ├── core/planner.py               explore → new direction (LLM, rule-based fallback)
  ├── core/mutator.py               mutate  → child config  (LLM, rule-based fallback)
  └── core/experiment.py            run one experiment (shared with the CLI below)

scripts/run_experiment.py           thin CLI wrapper
  └── core/experiment.py            validate → similarity → backtest → score → reflect → save
        ├── core/formula_validator.py   validate config/formula, complexity, eval namespace
        ├── core/similarity.py          near-duplicate gate (≥0.90) + novelty input to score
        ├── core/data_loader.py         fetch prices / income TTM / cashflow TTM (cached in cache/)
        │     └── core/backtest.py      process statements → data/; join → signal → simulate
        │           ├── core/data_process.py        rename + winsorise + standardise + ffill
        │           └── core/signal_calculation.py  eval formula → signal Series
        │                 └── core/formula_validator.py  build_panel_namespace()
        ├── core/robustness.py          sector, subperiod, regime, placebo checks
        ├── core/decision.py            V4 composite score → verdict (score_alpha)
        ├── core/reflection.py          LLM explanation + next-step suggestion
        └── core/memory.py              persist ExperimentRecord + bandit state to SQLite
              └── core/memory_analyzer.py   summaries, failure tags, effective_score

scripts/mutate_alpha.py
  └── core/mutator.py

scripts/plan_next.py
  └── core/planner.py
        └── core/memory_analyzer.py

scripts/export_graph.py
  ├── core/graph.py                 build NetworkX DAG from experiment store
  └── core/visualization.py         render interactive HTML with vis-network.js
```

---

## `core/` — Library modules

### `types.py`
Defines all shared data contracts as `TypedDict`s and dataclasses. Nothing is computed here — this file is imported by almost every other module.

| Type | Purpose |
|------|---------|
| `AlphaConfig` | Input spec for one experiment: `alpha_id`, `parent_id`, `batch_id`, `hypothesis`, `formula` (executed expression over raw column DataFrames), `features`, `universe`, dates, rebalance/cost settings |
| `BacktestMetrics` | Output metrics: IC_mean, ICIR, Sharpe, Q5_Q1_return, turnover, monotonicity, max_drawdown, deflated_sharpe, noise_risk |
| `BacktestResult` | Full backtest output including IC series, portfolio returns, sector IC, and per-period signal values |
| `RobustnessResult` | Sector stability, subperiod stability, market-regime Sharpes, placebo score |
| `SubScores` | V4: the five composite-score components, each in [0, 1]: performance, implementation, robustness, simplicity, novelty |
| `AlphaScore` | V4 (dataclass): `total` (directional 0–100), `signal_strength` (best-direction 0–100), `preferred_direction` (±1), `sub_scores`, derived `verdict`, `failure_reason`, `fatal` |
| `ExperimentRecord` | Everything persisted to SQLite: config + metrics + robustness + verdict + reflection + V4 score fields (`score`, `signal_strength`, `preferred_direction`, `sub_scores` — `None` on pre-V4 rows) |
| `ValidationResult` | Dataclass holding `valid` bool, `errors` list, `warnings` list |
| `SimilarityResult` | Jaccard similarity score against the most-similar prior alpha |
| `ResearchSuggestion` | One planner suggestion: direction, hypothesis, formula, features, rationale |
| `MemorySummary` | Aggregated view of all experiments: counts, best runs, unexplored features, trend observations |
| `Verdict` | Literal `"promising" \| "revise" \| "revise_invert" \| "failed"` — `revise_invert` means the signal is real but the hypothesis direction was wrong |
| `FailureCategory` | `high_turnover`, `weak_ic`, `negative_sharpe`, `high_noise`, `poor_robustness`, `too_complex`, `low_novelty`, `wrong_direction` |

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
Single owner of all formula-related logic: config validation, operator definitions, namespace construction, formula evaluation, and structural complexity.

**Constants:**

| Name | Purpose |
|------|---------|
| `AVAILABLE_RAW_COLUMNS` | frozenset of ~29 raw column names valid in `formula`: prices (`ADJUSTED_PRICE`, `ADJUSTED_VOLUME`) plus all income-statement and cashflow TTM columns (`REVENUE_LTM`, `NET_INCOME_LTM`, `CFO_LTM`, …) |
| `ALLOWED_FUNCTION_NAMES` | ~45 operators: cross-sectional (`rank`, `zscore`, `scale`, `group_rank`, `indneutralize`, …), time-series (`ts_mean`, `ts_std`, `ts_rank`, `ts_corr`, `decay_linear`, …), technical (`ema`, `rsi`, `macd`, `boll_*`, …) |
| `FORMULA_CONSTRAINT` | Canonical operator reference injected into every LLM prompt that may produce a formula |

**`validate_alpha(alpha)`** — validates an `AlphaConfig`, returns `ValidationResult`:
1. Required fields: `alpha_id`, `formula`, `universe`, `start_date`, `end_date`
2. `formula` — checks parenthesis balance (max nesting 5), verifies at least one `AVAILABLE_RAW_COLUMNS` column is referenced, errors on unrecognised operators, warns on unknown ALL_CAPS identifiers
3. Universe and date ordering

**`build_panel_namespace(processed_df)`** — builds the eval namespace for `formula`. Each column of `processed_df` is pivoted to a full `DATE × TICKER` wide DataFrame. The namespace includes all operators above, `np`, `float`/`nan`, and `SECTOR` for group operators. Pandas methods (`.shift()`, `.rolling()`, `.pct_change()`) work inline.

**`formula_complexity(formula)`** — V4: structural complexity used by the simplicity sub-score:
`n_calls + max_expr_depth + n_distinct_columns`, computed from the AST (`ast.parse(formula, mode="eval")`; depth counts only Call/BinOp/UnaryOp/Compare nodes). Calibration: `rank(ADJUSTED_PRICE) * -1` → 4; the quality+value fallback formula → ~12; heavily-nested gamed formulas ≥ 20. `SyntaxError` fallback: token count // 2.

---

### `signal_calculation.py`
Execution layer for alpha signal generation. Wires `formula_validator.build_panel_namespace` and `data_process._winsorise/_standardise` together to produce the signal Series consumed by `backtest.py`.

**`compute_signal(processed_df, formula, post_winsorise, post_standardise)`**:
1. Call `build_panel_namespace(processed_df)` → namespace of `DATE × TICKER` DataFrames
2. `eval(formula, namespace)` → `DATE × TICKER` signal DataFrame
3. Apply cross-sectional winsorise + standardise to the result
4. Stack to `(DATE, TICKER)` MultiIndex `pd.Series` named `"signal"`, drop NaNs

There is a single formula string per alpha — the execution formula written directly against raw column names (e.g. `rank(ADJUSTED_PRICE.shift(21) / ADJUSTED_PRICE.shift(252) - 1)`). The old display/raw formula split was removed in V3.9.

---

### `backtest.py`
Core simulation engine. Orchestrates data processing, signal computation, and the monthly-rebalanced long-short simulation.

**Data pipeline (run once per `(start_date, end_date)`):**
1. Unpack 4-tuple from `data_loader.load()` → `prices_df`, `income_ttm_df`, `cashflow_ttm_df`, `universe_df`
2. Process income statement independently via `data_process.process()` → cache to `data/processed_income_*.parquet`
3. Process cashflow statement independently via `data_process.process()` → cache to `data/processed_cashflow_*.parquet`
4. Outer-join processed income + cashflow on `(DATE, TICKER)` index
5. Add `SECTOR` from universe and raw `ADJUSTED_PRICE` / `ADJUSTED_VOLUME` from prices; restrict to actual trading dates

**Signal and backtest pipeline per rebalancing period:**
1. Evaluate `formula` via `signal_calculation.py` over the full date range at once → signal Series
2. At each month-end rebalancing date, slice signal and compute Spearman IC against next-month forward returns
3. Split into quintiles (bin edges deduplicated for low-cardinality signals); simulate Q5 − Q1 long-short portfolio return
4. Compute per-sector IC (for robustness)

Cache paths come from `data_process.processed_path(statement, start_date, end_date)`. `--no-cache` skips the processed parquet cache and rebuilds from the raw `cache/` files.

**Metrics computed:**
- `IC_mean` and `ICIR` (IC / IC std)
- `Sharpe` (annualised, monthly periods)
- `Q5_Q1_return` (mean per-period long-short spread — the un-annualised Sharpe numerator)
- `monotonicity` (avg Spearman rho between quintile rank and quintile return)
- `turnover` (avg constituent replacement rate in top/bottom quintiles)
- `max_drawdown`
- `deflated_sharpe` (López de Prado SR adjusted for skewness and kurtosis)
- `noise_risk` (`low` / `medium` / `high` based on deflated/raw Sharpe ratio)

Raises `RuntimeError` if fewer than 3 valid periods are produced. The private helpers `_max_drawdown` / `_deflated_sharpe` are reused by `decision.score_alpha` to compute exact inverted-direction metrics.

---

### `decision.py`
**V4: continuous composite scoring with derived verdicts.** The V3 two-tier gates (`check_tier1` / `decide`) are kept only for reference and for the threshold constants other modules import — the pipeline calls `score_alpha()`.

**`score_alpha(metrics, robustness, formula, similarity_score, portfolio_returns=None) → AlphaScore`**

Five sub-scores in [0, 1], combined with weights (module constants):

| Sub-score | Weight | Computation |
|-----------|--------|-------------|
| performance | 0.45 | Smooth ramps over IC_mean, ICIR, Sharpe (blended 70/30 with deflated Sharpe — anti-overfit), monotonicity. Every ramp passes through 0.5 at the old soft threshold and 0.0 at the old hard threshold |
| robustness | 0.20 | Equal thirds: subperiod_stability, placebo_score, regime-Sharpe consistency (neutral 0.5 if yfinance data missing) |
| implementation | 0.15 | Turnover penalty (1.0 below 0.30, 0.5 at 0.70, 0 at 0.90) + drawdown ramp |
| simplicity | 0.10 | `1 − ramp(formula_complexity, 4, 20)` — the anti-gaming pressure |
| novelty | 0.10 | `1 − similarity_score` vs the store |

**Direction-aware scoring** — separates "a predictive signal exists" from "the hypothesis direction is correct":
- `total` = composite on the raw metrics (hypothesis evaluation, sign preserved)
- `inverted_total` = composite on sign-flipped metrics; drawdown and deflated Sharpe recomputed **exactly** from the negated `portfolio_returns` series
- `signal_strength = max(total, inverted_total − INVERSION_PENALTY(=5))`, `preferred_direction = ±1`

**Fatal gates** (only truly dead cases): `|IC_mean| < 0.005 and |Sharpe| < 0.10` (no edge in either direction) or `max_drawdown ≤ −0.40` in the *preferred* direction → verdict `failed`, `signal_strength` capped at 25.

**Verdict bands on `signal_strength`**: ≥ 65 `promising`, 35–65 `revise`, < 35 `failed`. A revise-or-better score with `preferred_direction == −1` becomes `revise_invert` — never `failed` just because the sign was wrong. `failure_reason` names the weakest sub-score (or the inversion, with a flip-the-sign instruction).

`HARD_DUPLICATE_THRESHOLD = 0.90` — similarity at or above this aborts before the backtest (enforced by `experiment.py`).

---

### `experiment.py`
**V4: the single-experiment pipeline as a callable function**, shared by the CLI (`scripts/run_experiment.py`) and the loop runner (`scripts/run_loop.py`).

**`run_single_experiment(alpha, store, loader=None, force=False, verbose=True) → ExperimentOutcome`**

```
Step 1  Validate formula/config             (formula_validator.py)
Step 2  Similarity — hard abort only ≥ 0.90 (similarity.py); below that it feeds the novelty sub-score
Step 3  Backtest                            (backtest.py)
Step 4  Robustness (skipped for dead signals) (robustness.py)
Step 5  Composite score → verdict           (decision.score_alpha)
Step 6  Reflection with score context       (reflection.py)
Step 7  Save ExperimentRecord               (memory.py)
```

`ExperimentOutcome.status` is one of `completed` / `duplicate` / `validation_failed` / `backtest_error` — no `sys.exit` inside the pipeline, so the loop runner can treat failures as zero-reward pulls.

---

### `scheduler.py`
**V4: Thompson-sampling bandit** that decides explore-vs-mutate each loop iteration.

- **Arms**: one `__explore__` arm + `mutate:<alpha_id>` for the top-5 eligible parents (verdict ∈ {promising, revise, revise_invert} and `effective_score ≥ 35`), re-checked every iteration.
- **Fractional-Beta updates for continuous rewards**: a reward r ∈ [0, 1] updates the posterior as `α += r; β += (1 − r)` — no binarization. Priors: explore Beta(2, 1) (optimistic bootstrap); parent arms Beta(1 + score/100, 1).
- **Selection**: sample `random.betavariate(α, β)` per arm, argmax. Cold start (< 3 experiments) forces explore.
- **Rewards** (all on `signal_strength` via `effective_score`): mutate `r = clip(0.5 + (child − parent)/50)`; explore `r = clip(0.5 + (child − store mean)/50)`; duplicates / validation / backtest errors → 0.0 so the bandit learns to avoid arms producing garbage.
- **Persistence**: posteriors live in the `bandit_state` SQLite table, written after every pull — crash-safe across loop sessions.

---

### `robustness.py`
Computes four robustness diagnostics from the intermediate data returned by `backtest.py`. All checks use real data — no simulation.

| Diagnostic | What it measures |
|-----------|-----------------|
| `sector_stability` | Mean IC per GICS sector; reveals if the signal is sector-concentrated |
| `subperiod_stability` | IC consistency between the first and second halves of the period; detects regime-specific alphas |
| `market_regime_sharpe` | Annualised Sharpe split across five regimes (bull/bear, high/low/neutral VIX) using SPY and VIX from yfinance |
| `placebo_score` | Shuffles forward returns within each period and computes IC against the real signal; score near 1.0 means real IC is well above chance |

SPY and VIX data are fetched once and cached in a module-level dict for the process lifetime. All robustness outputs feed the V4 robustness sub-score (previously they were display-only).

---

### `reflection.py`
Produces a structured written analysis of an experiment after scoring. Tries the DeepSeek LLM first; falls back to rule-based templates if the API key is missing or the call fails.

**Output format** (both LLM and fallback):
```
Observation: <what the numbers show>
Failure Reason: <from decision.py>
Possible Explanation: <why the alpha performed this way>
Next Mutation: <one concrete change to try>
```

V4: `generate_reflection(..., alpha_score=)` receives the `AlphaScore`; the LLM prompt gains a Composite Score block (total, signal strength, direction, all five sub-scores, "sub-scores below 0.4 are the priority to fix"). For `revise_invert` verdicts the prompt asks the LLM to restate the economic hypothesis in the opposite direction, and the rule-based fallback recommends a sign flip without added complexity.

LLM output is prefixed with `[DISCLAIMER: LLM-generated hypothesis, not validated evidence]`.

---

### `mutator.py`
Generates a child `AlphaConfig` by mutating a parent experiment. Reads the parent's metrics, robustness results, score/sub-scores, and prior reflection, then asks DeepSeek to propose one targeted change (3 validation-retry attempts).

V4 prompt additions: the parent's composite score and sub-scores, plus targeted instructions — if `preferred_direction == −1` the primary mutation must be a sign flip; if simplicity < 0.5 the mutation MUST reduce complexity; if novelty < 0.4 it must change the signal source.

**Rule-based fallback** (priority order):
1. `preferred_direction == −1` → negate the formula and invert the hypothesis
2. High turnover → switch to quarterly rebalance
3. Weak Sharpe + no quality → add EBITDA-margin overlay
4. Low ICIR → add low-volatility damper
5. Has momentum but no quality → add quality overlay
6. Default → add liquidity screen

The child gets `alpha_id = {parent_id}_mut_{timestamp}` and `parent_id` set so the graph edge is created correctly.

---

### `planner.py`
Surveys all past experiments (via `memory_analyzer.py`) and proposes N new alpha research directions. Prioritises signals that haven't been tried yet.

**LLM path:** Sends a `MemorySummary` (verdict counts, failure modes, best experiments with scores, explored/unexplored features) to DeepSeek and parses a JSON array of `ResearchSuggestion`s. Each suggestion is validated before being accepted. V4 prompt explicitly prefers simple formulas (≤ 3 operator calls) since complexity is penalised and novelty rewarded.

**Rule-based fallback:** Five pre-defined suggestions covering value, quality+value, earnings growth, low volatility, and liquidity+momentum. Suggestions whose features haven't been explored are preferred.

**`suggestion_to_config(suggestion, alpha_id, batch_id, base_config)`** — converts a suggestion into a runnable `AlphaConfig`, using a prior experiment's config as the template for universe/date/cost defaults. Shared by `plan_next.py` and `run_loop.py`.

---

### `memory.py`
SQLite persistence layer (`db/experiments.db`). Experiment records live in the `experiments` table; V4 adds the `score`, `signal_strength`, `preferred_direction`, `sub_scores` columns (added via safe try/except `ALTER TABLE` migrations, so pre-V4 databases upgrade in place) and a second `bandit_state` table for the scheduler's posteriors.

Key methods on `ExperimentStore`:
- `save_experiment(record)` — INSERT OR REPLACE by `alpha_id`
- `load_all()` — returns all records ordered by `timestamp`
- `load_by_id(alpha_id)` — returns one record or `None`
- `load_bandit_state()` / `upsert_bandit_arm(arm_id, alpha, beta, pulls)` — V4 bandit persistence

---

### `memory_analyzer.py`
Aggregates `ExperimentStore` records into a `MemorySummary` and classifies failure modes.

**`effective_score(record)`** — V4: the signal-strength score used for rankings and bandit rewards. Prefers stored `signal_strength`, then stored `score`; pre-V4 rows are rescored lazily via `score_alpha` with neutral novelty 0.5 (directional only).

**`classify_failure(record)`** assigns one `FailureCategory` (first match wins):
`wrong_direction` (any `revise_invert`) → `high_turnover` → `negative_sharpe` → `weak_ic` → `high_noise` → `poor_robustness` → `too_complex` (simplicity < 0.3) → `low_novelty` (novelty < 0.2)

**`analyze_memory(store)`** returns:
- Verdict and failure-category counts
- Top-3 promising experiments ranked by `effective_score` (with score and direction)
- Which features have been explored vs. unexplored
- Trend observations (dominant failure mode, best score, unexplored signals)

---

### `similarity.py`
Guards against re-running near-duplicate alphas and supplies the novelty input to the composite score. Compares a new alpha against every prior experiment using a combined Jaccard similarity:

```
score = 0.5 × feature_jaccard + 0.5 × formula_token_jaccard  (+0.1 config bonus)
```

V4 semantics: similarity is a **graded penalty**, not a binary gate — the score flows into the novelty sub-score (`1 − similarity`). Only near-exact duplicates (`≥ 0.90`, `decision.HARD_DUPLICATE_THRESHOLD`) abort before the backtest, overridable with `--force`.

---

### `graph.py`
Builds a `networkx.DiGraph` where each node is an alpha experiment and each directed edge is a parent → child mutation relationship. Node attributes include all metrics, the V4 score fields (`score`, `signal_strength`, `preferred_direction`, `sub_scores`; `−1.0`/`0` sentinels for pre-V4 rows), and metadata so the visualisation renders without re-querying the database.

---

### `visualization.py`
Exports the research graph as a fully self-contained, interactive HTML file using vis-network.js (CDN).

Features of the exported HTML:
- **Summary bar** — total experiments, promising / revise / invert / failed counts, best Sharpe, best IC
- **Ring-layout graph** — one ring per batch/session; nodes colour-coded by verdict (green = promising, amber = revise, **blue = revise_invert**, red = failed); failed nodes have dashed borders; scored nodes label with their signal strength
- **Retractable detail panel** — clicking a node slides in a right panel with hypothesis, formula, **Composite Score section** (big signal-strength number, "↕ inverted" badge when direction is −1, colour-coded sub-score rows: good ≥ 0.6 / warn 0.4–0.6 / bad < 0.4), tiered metrics with threshold warnings, failure reason, mutation, and reflection
- **Light mode, Inter font**

Node and edge data are serialised as JSON and embedded directly in the HTML so the file works offline.

---

## `scripts/` — CLI entry points

### `run_loop.py`
**V4: the autonomous research loop** — closes the plan → run → learn cycle.

Each iteration: `ThompsonScheduler.select_action()` → planner (explore) or mutator (mutate) generates a config → config JSON written to `experiments/` for reproducibility → `run_single_experiment()` → reward computed vs parent/baseline → posterior updated and persisted. One `loop_{timestamp}` batch_id per session (= one ring in the graph). Ctrl-C safe: all state is persisted per iteration. Ends with a summary table (per-arm posteriors, best alpha).

**Usage:**
```bash
python scripts/run_loop.py --iterations 10
python scripts/run_loop.py --iterations 5 --sleep 10 --max-consecutive-failures 3
```

Without a `DEEPSEEK_API_KEY` the loop still runs on rule-based fallbacks, but expect near-duplicate mutations (zero-reward pulls) after a few iterations — the fallbacks are finite and deterministic.

---

### `run_experiment.py`
Thin CLI wrapper around `core/experiment.run_single_experiment` — parses args, loads the config JSON, prints the score breakdown, and maps non-completed outcomes to exit code 1.

**Usage:**
```bash
python scripts/run_experiment.py --config experiments/sample_alpha_001.json
python scripts/run_experiment.py --config experiments/my_alpha.json --no-cache --force
```

Flags: `--no-cache` bypasses the parquet cache; `--force` overrides the near-duplicate abort.

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
Reads all experiments from the database, summarises what has been tried, and uses `core/planner.py` to suggest N new alpha directions. With `--save`, converts each suggestion to a ready-to-run config via `planner.suggestion_to_config` and writes it to `experiments/`, stamping one shared `batch_id`.

**Usage:**
```bash
python scripts/plan_next.py
python scripts/plan_next.py --n 5
python scripts/plan_next.py --n 3 --save
```

---

### `export_graph.py`
Loads all experiments from SQLite, builds the `ResearchGraph`, assigns ring indices from chronologically-sorted batch_ids, and calls `core/visualization.py` to write a self-contained HTML file.

**Usage:**
```bash
python scripts/export_graph.py
python scripts/export_graph.py --filter-verdict promising,revise_invert --include-ancestors
python scripts/export_graph.py --filter-top 10      # ranked by signal_strength, Sharpe fallback
python scripts/export_graph.py --filter-batch loop_20260706_012337
```

Default output: `reports/research_graph.html`.

---

## `tests/`

Run with `python -m pytest tests/`.

| File | Covers |
|------|--------|
| `test_decision.py` | Ramp anchors at old V3 thresholds, score monotonicity in every metric, fatal gates (dead signal, catastrophic drawdown), direction separation (contrarian → `revise_invert`, not failed), verdict bands, `formula_complexity` calibration |
| `test_scheduler.py` | Cold-start forced explore, exact fractional Beta updates (no binarization), reward clipping, parent-arm eligibility, convergence to the better arm, cross-instance state persistence, reward symmetry around the parent score |
