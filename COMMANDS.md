# AlphaResearchBot — Command Reference

All commands are run from the repo root. Activate the virtual environment first:

```bash
source .venv/bin/activate
```

---

## 1. Run an Experiment

Runs a full backtest on an alpha config and saves the result to the database.

```bash
python scripts/run_experiment.py --config experiments/sample_alpha_001.json
```

**What it does (6 steps):**
1. Validates the formula and config
2. Checks similarity against all prior alphas (blocks if >80% similar)
3. Runs a monthly-rebalanced long-short backtest (downloads/caches S&P 500 data)
4. Runs robustness checks (sector stability, subperiod stability, market regime, placebo)
5. Applies decision rules → verdict: `promising`, `failed`, or `inconclusive`
6. Generates a DeepSeek LLM reflection and saves everything to `db/experiments.db`

**What it prints:**
```
Hypothesis : ...
Formula    : ...

[ Step 1 ] Validating formula...
[ Step 2 ] Running backtest...
  IC_mean        : 0.0123
  ICIR           : 0.0412
  Sharpe         : -0.1100
  Turnover       : 375.9 bps/yr
  ...
[ Step 3 ] Running robustness checks...
[ Step 4 ] Verdict: FAILED
[ Step 5 ] Reflection: ...
[ Step 6 ] Saved experiment 'alpha_001' to db/experiments.db
```

**What it creates:**
- A new row in `db/experiments.db` (the experiment record)
- Nothing else on disk

**Flags:**

| Flag | Effect |
|---|---|
| `--config <path>` | **(required)** Path to the alpha JSON file |
| `--no-cache` | Ignores the parquet cache and re-downloads all market data |
| `--force` | Runs even if the alpha is too similar to a prior one |
| `--db <path>` | Use a different SQLite database (default: `db/experiments.db`) |

**Example with flags:**
```bash
python scripts/run_experiment.py --config experiments/plan_001_value_via_sales_yield.json --force
python scripts/run_experiment.py --config experiments/my_alpha.json --no-cache
```

---

## 2. Plan Next Research Directions

Reads the experiment database and suggests N new alpha ideas. Uses DeepSeek LLM (falls back to rule-based suggestions if the API key is missing).

```bash
python scripts/plan_next.py
python scripts/plan_next.py --n 5
python scripts/plan_next.py --n 3 --save
```

**What it prints:**
```
Suggestion 1: value screen
  Hypothesis : Cheap stocks outperform expensive ones.
  Formula    : rank(PRICE_TO_SALES) * -1
  Features   : ['PRICE_TO_SALES']
  Parent     : none (new branch)
  Rationale  : No value-based signal has been tested yet.
```

**What it creates (with `--save`):**
- `N` new JSON files in `experiments/`, named like:
  - `experiments/plan_001_value_screen.json`
  - `experiments/plan_002_quality_plus_value.json`
  - etc.
- These files are ready to pass directly to `run_experiment.py`

**Flags:**

| Flag | Effect |
|---|---|
| `--n <int>` | Number of suggestions to generate (default: `3`) |
| `--save` | Save each suggestion as a JSON file in `experiments/` |
| `--db <path>` | Use a different database (default: `db/experiments.db`) |

**Typical workflow:**
```bash
# 1. Generate and save 3 suggestions
python scripts/plan_next.py --n 3 --save

# 2. Pick one and run it
python scripts/run_experiment.py --config experiments/plan_001_value_screen.json
```

---

## 3. Mutate an Existing Alpha

Generates a child alpha that addresses the failure mode of a parent experiment. Uses DeepSeek LLM (falls back to rule-based mutation).

```bash
python scripts/mutate_alpha.py --parent alpha_001
python scripts/mutate_alpha.py --parent alpha_001 --run
```

**What it creates:**
- `1` new JSON file in `experiments/`, named like:
  - `experiments/alpha_001_mut_20260622134500.json`
  - (timestamp is appended to make it unique)

**What it prints:**
```
Generating mutation from parent: alpha_001

Generated child alpha:
{
  "alpha_id": "alpha_001_mut_20260622134500",
  "parent_id": "alpha_001",
  "formula": "rank(EBITDA_MARGIN)",
  "mutation": "Removed MOM12_1 to reduce turnover",
  ...
}

Saved to: experiments/alpha_001_mut_20260622134500.json
```

**With `--run`:** immediately runs the generated child through the full pipeline (same as calling `run_experiment.py` on it).

**Flags:**

| Flag | Effect |
|---|---|
| `--parent <alpha_id>` | **(required)** The `alpha_id` of the experiment to mutate |
| `--run` | Run the generated child alpha immediately after saving |
| `--db <path>` | Use a different database (default: `db/experiments.db`) |

**Typical workflow:**
```bash
# Mutate a failed experiment and run the child right away
python scripts/mutate_alpha.py --parent alpha_001 --run
```

The mutation strategy depends on the failure mode:
- High turnover → switches rebalance to quarterly
- Weak signal (low Sharpe) → adds a quality overlay (`EBITDA_MARGIN`)
- Inconsistent signal (low ICIR) → adds a volatility filter (`VOL_20D`)

---

## 4. Export Research Graph

Builds an interactive HTML graph of all experiments — nodes are alphas, edges connect parent → child.

```bash
python scripts/export_graph.py
```

**What it creates:**
- `reports/research_graph.html` — open in any browser

**Node colors:**
- **Green** — promising
- **Red** — failed
- **Orange** — inconclusive

**Node tooltip shows:**
- Formula, Sharpe, ICIR, Deflated Sharpe
- Verdict and failure reason
- Failure category (e.g. `high_turnover`, `weak_ic`)
- Mutation reason (what changed from the parent)
- LLM reflection snippet

**Flags:**

| Flag | Effect |
|---|---|
| `--db <path>` | Use a different database (default: `db/experiments.db`) |
| `--output <path>` | Write HTML to a different location (default: `reports/research_graph.html`) |

```bash
# Open after export
python scripts/export_graph.py && open reports/research_graph.html
```

---

## Writing Your Own Alpha Config

Create a `.json` file anywhere (the `experiments/` folder is conventional) with this structure:

```json
{
  "alpha_id": "my_alpha_001",
  "parent_id": null,
  "hypothesis": "One sentence describing the investment thesis.",
  "formula": "rank(EBITDA_MARGIN) + rank(PRICE_TO_SALES) * -1",
  "features": ["EBITDA_MARGIN", "PRICE_TO_SALES"],
  "mutation": "Initial quality-value hypothesis",
  "universe": "sp500",
  "start_date": "2018-01-01",
  "end_date": "2024-12-31",
  "neutralization": "sector",
  "rebalance": "monthly",
  "transaction_cost_bps": 5,
  "holding_period_days": 20
}
```

**Supported formula operators:** `rank()`, `zscore()`, `log()`, `abs()`, `sign()`, and arithmetic `+ - * / **`

**Supported features (14 total):**

| Feature | Description |
|---|---|
| `MOM12_1` | 12-month momentum excluding last month |
| `MOM6_1` | 6-month momentum excluding last month |
| `VOL_20D` | 20-day rolling volatility |
| `LIQUIDITY` | 20-day average dollar volume |
| `EBITDA_MARGIN` | EBITDA / Revenue |
| `SALES_GROWTH` | Year-over-year revenue growth |
| `EPS_GROWTH` | Year-over-year EPS growth |
| `PRICE_TO_SALES` | Price / Revenue per share |
| `EPS_LTM` | Trailing 12-month EPS |
| `SALES_LTM` | Trailing 12-month revenue |
| `EBITDA_LTM` | Trailing 12-month EBITDA |
| `COGS_LTM` | Trailing 12-month cost of goods sold |
| `ADJUSTED_PRICE` | Split/dividend-adjusted close price |
| `ADJUSTED_VOLUME` | Adjusted trading volume |

**Config options:**

| Field | Options | Notes |
|---|---|---|
| `universe` | `sp500`, `russell1000`, `russell3000` | Smaller = faster |
| `neutralization` | `sector`, `market`, `none` | `sector` removes sector-level bias |
| `rebalance` | `monthly`, `quarterly` | Quarterly reduces turnover |
| `transaction_cost_bps` | integer | Cost per trade in basis points |
| `holding_period_days` | integer | Forward return window for IC calculation |

**Decision thresholds** (from `core/decision.py`):

| Metric | Threshold | Verdict |
|---|---|---|
| Turnover | > 300 bps/yr | failed |
| Sharpe | < 0.5 | failed |
| ICIR | < 0.3 | failed |
| Noise risk | high | inconclusive |

---

## Quick Reference

```bash
# Full loop: plan → save → run best → visualize
python scripts/plan_next.py --n 3 --save
python scripts/run_experiment.py --config experiments/plan_001_<name>.json
python scripts/export_graph.py && open reports/research_graph.html

# Mutate a failed experiment and run immediately
python scripts/mutate_alpha.py --parent alpha_001 --run

# Run your own hand-crafted alpha
python scripts/run_experiment.py --config experiments/my_alpha.json

# Re-run with fresh data (ignore cache)
python scripts/run_experiment.py --config experiments/my_alpha.json --no-cache

# Force-run a near-duplicate alpha
python scripts/run_experiment.py --config experiments/my_alpha.json --force
```
