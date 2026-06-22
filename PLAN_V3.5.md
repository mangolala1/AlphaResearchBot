# AlphaResearchBot — Version 3.5 Plan

## Context

The research loop runs experiments, stores them in SQLite, and uses DeepSeek to plan next directions and generate mutations. Three problems motivate v3.5:

1. **Planner context is raw and unstructured** — `planner.py` feeds the LLM a flat table of experiment rows. With more experiments the prompt grows noisy and the LLM can't identify failure patterns.
2. **Failure mode is opaque** — `failure_reason` is a human-readable sentence, not a machine-readable category. Neither the graph nor the planner can act on it programmatically.
3. **Mutation and planning prompts are brittle** — the list of supported operators and features is hard-coded as a static string that can silently diverge from the actual evaluator.

---

## Five Changes

### 1. Memory Analyzer (`core/memory_analyzer.py` — new file)

New module with two public functions.

**`classify_failure(record: ExperimentRecord) -> str | None`**

Returns `None` for "promising" experiments. For all others, returns one of five canonical category strings (first match wins):

| Priority | Condition | Category |
|---|---|---|
| 1 | `metrics["turnover"] > 300` | `"high_turnover"` |
| 2 | `metrics["Sharpe"] < 0` | `"negative_sharpe"` |
| 3 | `metrics["ICIR"] < 0.3` | `"weak_ic"` |
| 4 | `metrics["noise_risk"] == "high"` | `"high_noise"` |
| 5 | any robustness score < 0.3 | `"poor_robustness"` |

Thresholds imported from `core.decision` — no duplication. Covers both "failed" and "inconclusive" verdicts.

**`analyze_memory(store: ExperimentStore) -> MemorySummary`**

Aggregates all experiments into a structured summary dict:
- `total_experiments`, `verdict_counts` (promising / failed / inconclusive)
- `failure_category_counts` (how many experiments hit each taxonomy bucket)
- `best_experiments` — top-3 by Sharpe among "promising" records
- `explored_features` — union of all features used so far
- `unexplored_features` — canonical 14 LLM-safe features minus `explored_features`
- `trend_observations` — 2–4 deterministic human-readable sentences (e.g. "Dominant failure: high_turnover (67% of failures)")

Handles empty store gracefully.

---

### 2. Failure Taxonomy (`core/types.py`)

Add a `FailureCategory` Literal type and a `MemorySummary` TypedDict (consumed by the planner prompt and memory analyzer):

```python
FailureCategory = Literal[
    "high_turnover", "weak_ic", "negative_sharpe", "high_noise", "poor_robustness"
]

class MemorySummary(TypedDict):
    total_experiments: int
    verdict_counts: dict[str, int]
    failure_category_counts: dict[str, int]
    best_experiments: list[dict]       # alpha_id, formula, Sharpe, ICIR
    explored_features: list[str]
    unexplored_features: list[str]
    trend_observations: list[str]
```

`failure_category` is never stored in the database — it is always computed on-the-fly from existing fields, so no schema migration is needed.

---

### 3. Planner Context (`core/planner.py`)

Replace the flat raw-row listing in `_build_plan_prompt` with the structured `MemorySummary`. The LLM will receive:

```
== Memory Summary ==
Total: 11 experiments (1 promising, 9 failed, 1 inconclusive)
Failure patterns: high_turnover: 6  weak_ic: 2  negative_sharpe: 1
Best experiments:
  alpha_XXX | rank(EBITDA_MARGIN) | Sharpe=0.72 | ICIR=0.41
Explored features: EBITDA_MARGIN, MOM12_1, VOL_20D
Unexplored features: COGS_LTM, EPS_GROWTH, PRICE_TO_SALES, SALES_LTM, ...
Trend observations:
  - Dominant failure: high_turnover (67% of failures). Prefer quarterly rebalance.
  - Value and growth signals completely untested.
Recent mutation hints: [last 3 "Next Mutation" lines from reflections]
```

The rule-based fallback `_rule_based_plan` will source `tried_features` from `summary["explored_features"]` instead of re-scanning records.

---

### 4. Mutation Constraint (`core/mutator.py` + `core/planner.py`)

**Problem:** `ALLOWED_FUNCTION_NAMES` in `validator.py` includes `delta`, `ts_mean`, `ts_std` (they pass syntax validation but raise `NotImplementedError` at runtime). The LLM prompt hard-codes a static warning string that could silently drift.

**Fix:** Add `EVALUATOR_FEATURES` to `validator.py` (the 14 LLM-safe features, already computed inline in `mutator.py`) and derive formula constraints programmatically in both `mutator.py` and `planner.py`:

```python
# core/validator.py — new export
EVALUATOR_FEATURES: frozenset[str] = ALLOWED_FEATURES - {
    "SECTOR", "INDUSTRY", "FACTSET_ID",
    "EPS_NTM", "SALES_NTM", "EBITDA_NTM", "COGS_NTM",
}

# core/mutator.py and core/planner.py — replace static string
_SAFE_OPERATORS = ALLOWED_FUNCTION_NAMES - {"delta", "ts_mean", "ts_std"}
_FORMULA_CONSTRAINT = (
    f"IMPORTANT — supported operators ONLY: {', '.join(sorted(_SAFE_OPERATORS))}() "
    "and arithmetic (+, -, *, /, **). "
    "ts_mean(), ts_std(), delta() raise NotImplementedError — never use them. "
    f"Supported features: {', '.join(sorted(EVALUATOR_FEATURES))}."
)
```

If a new operator is added to the evaluator in the future, the prompt automatically updates.

---

### 5. Research Graph Enrichment (`core/graph.py` + `core/visualization.py`)

**`graph.py` — `add_experiment`:** Add two new node attributes:
- `failure_category` — computed via `classify_failure(record)`, stored as `"N/A"` for promising experiments
- `mutation_reason` — pulled from `record.get("mutation", "")` (already stored in the DB, just not exposed as a node attribute)

Uses a local import `from core.memory_analyzer import classify_failure` inside the method body (consistent with existing local-import pattern in this repo) to avoid circular dependencies.

**`visualization.py` — `_build_tooltip`:** Add two lines after the existing `Failure:` line:

```python
f"<b>Failure Category:</b> {attrs.get('failure_category', 'N/A')}<br>"
f"<b>Mutation Reason:</b> {attrs.get('mutation_reason', '')}<br>"
```

---

## Files Changed

| File | Change |
|---|---|
| `core/types.py` | Add `FailureCategory` Literal + `MemorySummary` TypedDict |
| `core/validator.py` | Add `EVALUATOR_FEATURES` constant |
| `core/memory_analyzer.py` | **New**: `classify_failure`, `analyze_memory` |
| `core/planner.py` | Structured prompt via `MemorySummary`; derived `_FORMULA_CONSTRAINT` |
| `core/mutator.py` | Derived `_FORMULA_CONSTRAINT` and `_AVAILABLE_FEATURES` from constants |
| `core/graph.py` | Add `failure_category`, `mutation_reason` node attributes |
| `core/visualization.py` | Add two tooltip lines |

No SQLite schema changes.

---

## Implementation Order

1. `core/types.py` — `FailureCategory` + `MemorySummary` (no deps)
2. `core/validator.py` — `EVALUATOR_FEATURES` (no deps)
3. `core/memory_analyzer.py` — new file (depends on 1, 2, `decision.py`)
4. `core/mutator.py` — derived constraints (depends on 2)
5. `core/planner.py` — structured prompt (depends on 2, 3)
6. `core/graph.py` — new node attrs (depends on 3)
7. `core/visualization.py` — extended tooltip (depends on 6)

---

## Verification

```bash
# Smoke test: classify all current experiments
python -c "
from core.memory import ExperimentStore
from core.memory_analyzer import classify_failure, analyze_memory
store = ExperimentStore()
for r in store.load_all():
    print(r['alpha_id'], '->', classify_failure(r))
print()
print(analyze_memory(store))
"
# Expected: all 4 current experiments → "high_turnover"

# Planner: confirm structured summary appears
python scripts/plan_next.py --n 2

# Graph: inspect new tooltip fields in browser
python scripts/export_graph.py
open reports/research_graph.html
# → nodes should show "Failure Category: high_turnover" and "Mutation Reason: ..."
```
