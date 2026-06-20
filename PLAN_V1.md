# AlphaResearchBot V1 — Implementation Plan

## Context

Building the first working version of AlphaResearchBot: a local, end-to-end iterative alpha discovery loop. No real Snowflake queries, no real LLM calls in V1 — everything is mock/deterministic. The goal is a clean modular codebase that demonstrates the full research loop and can be upgraded to real data/LLM calls in V2.

Spec: `version1.md`. Available columns for formula validation: `Data/data_tables.md`. Existing code (`Data/config.py`, `Data/data_retrieval.py`) is not touched in V1 but serves as a future integration point.

---

## Target Directory Structure

```
AlphaResearchBot/
├── version1.md
├── PLAN_V1.md                # this file
├── requirements.txt          # NEW
├── .env                      # existing (secrets)
├── Data/                     # existing, untouched
│   ├── config.py
│   ├── data_retrieval.py
│   ├── data_tables.md
│   └── check_snowflake_access.py
├── core/                     # NEW — all business logic
│   ├── __init__.py
│   ├── types.py              # shared TypedDicts / Literals
│   ├── validator.py          # formula + alpha validation
│   ├── backtest.py           # mock deterministic backtest engine
│   ├── robustness.py         # mock robustness checks
│   ├── decision.py           # pass/fail/inconclusive logic
│   ├── reflection.py         # rule-based reflection generator
│   ├── memory.py             # SQLite persistence layer
│   ├── graph.py              # NetworkX research graph
│   └── visualization.py     # PyVis HTML export
├── experiments/              # NEW — alpha JSON configs
│   └── sample_alpha_001.json
├── reports/                  # NEW — output directory (auto-created)
├── db/                       # NEW — SQLite DB lives here (auto-created)
└── scripts/                  # NEW — CLI entry points
    ├── run_experiment.py
    └── export_graph.py
```

---

## Module Design

### `core/types.py`
Shared type definitions used across all modules. Avoids circular imports.

```python
AlphaConfig       # TypedDict — mirrors the alpha JSON schema exactly
BacktestMetrics   # TypedDict — IC_mean, ICIR, Sharpe, turnover, max_drawdown, deflated_sharpe, noise_risk
RobustnessResult  # TypedDict — sector_stability, subperiod_stability, market_regime_sharpe, placebo_score
Verdict           # Literal["promising", "failed", "inconclusive"]
ExperimentRecord  # TypedDict — all fields stored in SQLite (union of above + meta)
ValidationResult  # dataclass — valid: bool, errors: list[str]
```

---

### `core/validator.py`

Validates an alpha config dict before running any backtest.

**Constants:**
- `ALLOWED_FEATURES` — set derived from `data_tables.md`:
  `{EPS_LTM, EPS_NTM, SALES_LTM, SALES_NTM, EBITDA_LTM, EBITDA_NTM, COGS_LTM, COGS_NTM, ADJUSTED_PRICE, ADJUSTED_VOLUME, SECTOR, INDUSTRY, FACTSET_ID}`
  Plus derived/computed features commonly used in alphas: `EBITDA_MARGIN, MOM12_1, MOM6_1, SALES_GROWTH, EPS_GROWTH, PRICE_TO_SALES, VOL_20D`

- `ALLOWED_OPERATORS` — `{rank, zscore, log, abs, sign, delta, ts_mean, ts_std, +, -, *, /, (, )}`

- `FUTURE_LOOKING_FIELDS` — NTM fields that are arguably forward-looking in some contexts:
  `{EPS_NTM, SALES_NTM, EBITDA_NTM, COGS_NTM}` — these are allowed but flagged as a warning, not an error

**Function:**
```python
def validate_alpha(alpha: AlphaConfig) -> ValidationResult:
```
Checks:
1. Required keys present (alpha_id, formula, features, universe, start_date, end_date)
2. Every feature in `alpha["features"]` is in `ALLOWED_FEATURES`
3. Formula string only contains tokens from `ALLOWED_OPERATORS` + `ALLOWED_FEATURES` + numbers
4. Parenthesis depth ≤ 5 (no excessive nesting)
5. Features list matches features actually referenced in the formula
6. Universe is one of `{sp500, russell1000, russell3000}`
7. Date format is YYYY-MM-DD and start < end

Returns `ValidationResult(valid=True/False, errors=[...], warnings=[...])`

---

### `core/backtest.py`

Deterministic mock backtest — same alpha always produces the same metrics.

**Approach:** Hash the formula string + universe + neutralization using `hashlib.sha256` to get a reproducible seed, then use `random.Random(seed)` to draw from realistic metric distributions.

```python
def run_backtest(alpha: AlphaConfig) -> BacktestMetrics:
```

Realistic metric ranges:
| Metric | Range |
|---|---|
| IC_mean | -0.04 to 0.12 |
| ICIR | -0.3 to 1.8 |
| Sharpe | -0.3 to 2.2 |
| turnover | 20 to 450 |
| max_drawdown | -0.05 to -0.45 |
| deflated_sharpe | Sharpe × U(0.4, 1.0) |
| noise_risk | derived from deflated_sharpe gap |

`noise_risk` assignment:
- `deflated_sharpe / Sharpe < 0.5` → `"high"`
- `< 0.75` → `"medium"`
- else → `"low"`

---

### `core/robustness.py`

Mock robustness scores — also deterministic via hashing.

```python
def run_robustness(alpha: AlphaConfig, metrics: BacktestMetrics) -> RobustnessResult:
```

Each sub-score is a float in `[0, 1]`:
- `sector_stability` — how stable the alpha is across sectors
- `subperiod_stability` — IC consistency across sub-periods
- `market_regime_sharpe` — Sharpe across bull/bear regimes
- `placebo_score` — 1 - (score from running alpha on shuffled data) — near 0 means no overfitting

All mock but seeded deterministically from the alpha hash.

---

### `core/decision.py`

Pure rule-based decision — no randomness.

```python
def decide(metrics: BacktestMetrics, robustness: RobustnessResult) -> tuple[Verdict, str | None]:
```

Decision rules (in priority order):
1. `turnover > 300` → `("failed", "Excessive turnover: {turnover:.1f}bps/yr exceeds 300 threshold")`
2. `Sharpe < 0.5 or ICIR < 0.3` → `("failed", "Sharpe {s:.2f} or ICIR {ic:.2f} below minimum thresholds")`
3. `noise_risk == "high"` → `("inconclusive", "High noise risk: deflated Sharpe significantly below raw Sharpe")`
4. else → `("promising", None)`

Returns `(verdict, failure_reason)` where failure_reason is None for "promising".

---

### `core/reflection.py`

Rule-based mock LLM that generates a structured reflection string. Every output is clearly labeled.

```python
def generate_reflection(
    alpha: AlphaConfig,
    metrics: BacktestMetrics,
    verdict: Verdict,
    failure_reason: str | None,
) -> str:
```

Output format (multi-line string):
```
[DISCLAIMER: LLM-generated hypothesis, not validated evidence]

Observation: <describe what the metrics show>
Possible Reason: <plausible explanation based on formula features>
Failure Reason: <if failed, why it likely failed>
Next Mutation Suggestion: <one concrete change to try next>
```

Logic: uses if/elif chains on verdict + metrics to select from a small set of template strings parameterized with actual numbers.

---

### `core/memory.py`

SQLite persistence. Database lives at `db/experiments.db` (created on first run).

```python
class ExperimentStore:
    def __init__(self, db_path: str = "db/experiments.db")
    def init_db(self) -> None          # CREATE TABLE IF NOT EXISTS
    def save_experiment(self, record: ExperimentRecord) -> None   # INSERT OR REPLACE
    def load_all(self) -> list[ExperimentRecord]
    def load_by_id(self, alpha_id: str) -> ExperimentRecord | None
```

SQLite schema:
```sql
CREATE TABLE experiments (
    alpha_id       TEXT PRIMARY KEY,
    parent_id      TEXT,
    timestamp      TEXT,
    hypothesis     TEXT,
    formula        TEXT,
    features       TEXT,   -- JSON array
    mutation       TEXT,
    config         TEXT,   -- full JSON blob
    metrics        TEXT,   -- JSON object
    robustness     TEXT,   -- JSON object
    verdict        TEXT,
    failure_reason TEXT,
    reflection     TEXT
);
```

JSON-serialized fields (features, config, metrics, robustness) use `json.dumps`/`json.loads` for storage.

---

### `core/graph.py`

NetworkX directed graph — each node is an alpha, edges are parent→child.

```python
class ResearchGraph:
    def __init__(self)
    def add_experiment(self, record: ExperimentRecord) -> None
    def build_from_store(self, store: ExperimentStore) -> None   # loads all and adds
    def get_graph(self) -> nx.DiGraph
```

Node attributes stored per spec: `Sharpe`, `ICIR`, `verdict`, `failure_reason`, `mutation`, `reflection`, plus `alpha_id`, `hypothesis`, `formula`, `deflated_sharpe` (for tooltip).

Edge added when `parent_id` is not None: `graph.add_edge(parent_id, alpha_id)`.

---

### `core/visualization.py`

PyVis HTML export.

```python
def export_graph_html(graph: nx.DiGraph, output_path: str = "reports/research_graph.html") -> None:
```

Node color mapping:
- `verdict == "promising"` → `#2ecc71` (green)
- `verdict == "failed"` → `#e74c3c` (red)
- `verdict == "inconclusive"` → `#f39c12` (yellow/orange)

Node tooltip (title attribute in pyvis):
```
ID: alpha_001
Hypothesis: ...
Formula: ...
Sharpe: 1.23  |  ICIR: 0.87  |  Deflated Sharpe: 0.95
Failure: None
Reflection: ...
```

`output_path`'s parent directory is created with `Path.mkdir(parents=True, exist_ok=True)`.

---

## Sample Experiment Config

### `experiments/sample_alpha_001.json`
Exactly as specified in `version1.md`:
```json
{
  "alpha_id": "alpha_001",
  "parent_id": null,
  "hypothesis": "Stocks with improving profitability and positive momentum outperform within sectors.",
  "formula": "rank(EBITDA_MARGIN) + 0.5 * rank(MOM12_1)",
  "features": ["EBITDA_MARGIN", "MOM12_1"],
  "mutation": "Initial quality plus momentum hypothesis",
  "universe": "sp500",
  "start_date": "2018-01-01",
  "end_date": "2024-12-31",
  "neutralization": "sector",
  "rebalance": "monthly",
  "transaction_cost_bps": 5,
  "holding_period_days": 20
}
```

---

## CLI Scripts

### `scripts/run_experiment.py`

```
python scripts/run_experiment.py --config experiments/sample_alpha_001.json
```

Pipeline (in order):
1. Parse `--config` arg, load JSON
2. `validate_alpha(alpha)` — print errors and exit if invalid
3. `run_backtest(alpha)` — print metrics
4. `run_robustness(alpha, metrics)` — print scores
5. `decide(metrics, robustness)` → verdict + failure_reason
6. `generate_reflection(alpha, metrics, verdict, failure_reason)`
7. `store.save_experiment(record)` — persists full ExperimentRecord to SQLite
8. Print summary table to stdout

Uses `argparse`. All output goes to stdout; no interactive prompts.

### `scripts/export_graph.py`

```
python scripts/export_graph.py
```

Pipeline:
1. `store.load_all()` — load all experiments from SQLite
2. `graph.build_from_store(store)` — build NetworkX DiGraph
3. `export_graph_html(graph, "reports/research_graph.html")`
4. Print: `Graph exported to reports/research_graph.html (N nodes, E edges)`

---

## Dependencies (`requirements.txt`)

```
networkx>=3.0
pyvis>=0.3.2
python-dotenv>=1.0.0
```

`sqlite3` and `hashlib` are stdlib — no extra install needed. Snowflake connector stays in the existing virtual environment but is not imported in V1 core modules.

---

## Implementation Sequence

| Step | File(s) | Notes |
|---|---|---|
| 1 | `requirements.txt` | Minimal new deps |
| 2 | `core/__init__.py`, `core/types.py` | Shared types first to avoid circular imports |
| 3 | `experiments/sample_alpha_001.json` | Input fixture needed for testing |
| 4 | `core/validator.py` | Independent, testable unit |
| 5 | `core/backtest.py` | Independent, no imports from other core modules |
| 6 | `core/robustness.py` | Depends on types only |
| 7 | `core/decision.py` | Pure logic, no I/O |
| 8 | `core/reflection.py` | Pure logic, no I/O |
| 9 | `core/memory.py` | SQLite I/O, depends on types |
| 10 | `core/graph.py` | Depends on memory + types |
| 11 | `core/visualization.py` | Depends on graph |
| 12 | `scripts/run_experiment.py` | Orchestrates steps 4–9 |
| 13 | `scripts/export_graph.py` | Orchestrates steps 10–11 |

---

## Verification

End-to-end test (manual, no test framework required for V1):

```bash
# Install deps
pip install -r requirements.txt

# Run a single experiment
python scripts/run_experiment.py --config experiments/sample_alpha_001.json
# Expected: prints metrics table, verdict, reflection; db/experiments.db created

# Export graph
python scripts/export_graph.py
# Expected: reports/research_graph.html created; open in browser to verify node colors/tooltips

# Run a child experiment (after manually creating experiments/sample_alpha_002.json with parent_id="alpha_001")
python scripts/run_experiment.py --config experiments/sample_alpha_002.json
python scripts/export_graph.py
# Expected: two nodes with edge alpha_001 → alpha_002 visible in HTML graph
```
