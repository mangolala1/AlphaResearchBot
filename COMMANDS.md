# AlphaResearchBot — Command Reference

All commands are run from the repo root. Activate the virtual environment first:

```bash
source .venv/bin/activate
```

---

## 1. Plan Next Research Directions

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

## 2. Run an Experiment

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

(Refer to `formula_validator.py` for allowed operators and data columns.)
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

**Config options:**

| Field | Options | Notes |
|---|---|---|
| `universe` | `sp500`| Smaller = faster |
| `neutralization` | `sector`, `market`, `none` | `sector` removes sector-level bias |
| `rebalance` | `monthly`, `quarterly` | Quarterly reduces turnover |
| `transaction_cost_bps` | integer | Cost per trade in basis points |
| `holding_period_days` | integer | Forward return window for IC calculation |


## Testing

```bash
# 1. Run a short loop against a brand-new database (file is created automatically)
python scripts/run_loop.py --iterations 5 --db db/test_experiments.db

# 2. Export the graph from ONLY that database
python scripts/export_graph.py --db db/test_experiments.db --output reports/test_graph.html
open reports/test_graph.html

# 3. Cleanup when done, clears up both experiment records and bandit posteriors
rm db/test_experiments.db

# (Optional) Created json files are harmless to keep
rm experiments/loop_*.json

# 4. Alternative (If wanting the existing experiements as bandit parents/memory)
cp db/experiments.db db/test_experiments.db                                                  
python scripts/run_loop.py --iterations 5 --db db/test_experiments.db                        
# note the batch id printed at loop start, e.g. loop_20260706_120000                         
python scripts/export_graph.py --db db/test_experiments.db --filter-batch loop_20260706_120000 --output reports/test_graph.html 
#  Here --filter-batch (existing flag in scripts/export_graph.py) shows only the loop session's ring; 
#  --include-ancestors to also show the parents that mutations branched from.
```

What to look for:
- **Scoring output**: the `[ Step 5 ] Composite scoring` block prints the composite score, predictive magnitude, direction status, and all five sub-scores; novelty ≈ 0.00 on a rerun (it's a duplicate of itself) and should be named the weakest component.
- **Loop output**: each iteration logs the scheduler's pick and the outcome, e.g. `[2/5] mutate:alpha_x → child | score 58.1 (parent 68.4) | reward 0.29 | revise`. Reward > 0.5 means the child improved on its parent; duplicates earn 0.0. The end-of-run table shows per-arm Beta posteriors. Ctrl-C is safe — all state persists per iteration, and the next loop run resumes from the saved posteriors.
- **Graph**: the loop session appears as its own ring; nodes are labeled with their composite score; clicking one shows the Composite Score panel, with a "direction contradicted" badge on signals that ran opposite to their hypothesis.

Tunables live as module constants at the top of `core/decision.py` (weights, verdict bands, fatal gates) and `core/scheduler.py` (priors, parent floor, contradicted-parent floor, top-K, reward scale). A health check worth watching early on: if the explore arm never wins, make `EXPLORE_PRIOR` more optimistic.
