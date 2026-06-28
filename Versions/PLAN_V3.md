# AlphaResearchBot V3 — Implementation Plan

## Context

V2 made the pipeline real: live price data (yfinance), real fundamentals (SimFin), genuine IC
backtests, and a DeepSeek-powered reflection step. But the research loop is still driven
manually — a human must decide what to run next, what to mutate, and whether a new alpha is
meaningfully different from prior work.

V3 makes the research loop autonomous:

1. **Research Planner** — reads experiment memory and uses DeepSeek to suggest the next N
   directions to explore.
2. **Real LLM Reflection (update)** — the existing reflection already calls DeepSeek, but
   does not pass robustness data. Extend it to include all four robustness scores so the LLM
   can reason about sector/regime/subperiod behaviour.
3. **Alpha Mutation Generator** — given a parent alpha's full record, uses DeepSeek to
   produce a valid child AlphaConfig JSON, then validates it before returning.
4. **Similarity Check** — before running a new alpha, compare its features and formula
   tokens structurally against every prior alpha; block if too similar (>0.8 Jaccard).

LLM for all features: DeepSeek (`deepseek-chat` via OpenAI-compatible client, key already
in `.env`). All new LLM paths have rule-based fallbacks.

---

## What V3 Adds vs V2

| Module | V2 | V3 |
|---|---|---|
| `core/reflection.py` | DeepSeek; missing robustness in prompt | Add robustness scores; reorder 4-part output |
| `core/planner.py` | Does not exist | NEW: reads memory, calls DeepSeek, returns `list[ResearchSuggestion]` |
| `core/mutator.py` | Does not exist | NEW: takes parent record, calls DeepSeek, returns validated `AlphaConfig` |
| `core/similarity.py` | Does not exist | NEW: Jaccard on features + formula tokens, returns `SimilarityResult` |
| `core/types.py` | No `SimilarityResult`, `ResearchSuggestion` | Add both TypedDicts |
| `scripts/run_experiment.py` | No similarity gate; reflection misses robustness | Add Step 1.5 similarity check; pass robustness to reflection |
| `scripts/plan_next.py` | Does not exist | NEW: CLI wrapper for planner |
| `scripts/mutate_alpha.py` | Does not exist | NEW: CLI wrapper for mutator, writes JSON to experiments/ |

**Unchanged from V2:** `core/backtest.py`, `core/robustness.py`, `core/features.py`,
`core/formula_eval.py`, `core/data_loader.py`, `core/decision.py`, `core/memory.py`,
`core/graph.py`, `core/validator.py`, `core/visualization.py`, `scripts/export_graph.py`.

---

## Implementation Sequence

| Step | File | Action |
|---|---|---|
| 1 | `core/types.py` | Add `SimilarityResult` and `ResearchSuggestion` TypedDicts |
| 2 | `core/reflection.py` | Add `robustness` param; update prompt with 4 robustness scores; reorder 4-part output |
| 3 | `core/similarity.py` | New module: feature Jaccard + formula token Jaccard; threshold gate |
| 4 | `core/mutator.py` | New module: DeepSeek mutation with rule-based fallback |
| 5 | `core/planner.py` | New module: DeepSeek research planner with rule-based fallback |
| 6 | `scripts/run_experiment.py` | Add Step 1.5 similarity check; pass `robustness` to `generate_reflection` |
| 7 | `scripts/mutate_alpha.py` | New CLI script |
| 8 | `scripts/plan_next.py` | New CLI script |
| 9 | `PLAN_V3.md` | Save this plan to the repo |

---

## Step 1 — `core/types.py`

Add two new TypedDicts after the existing definitions:

```python
class SimilarityResult(TypedDict):
    is_unique: bool
    most_similar_id: str | None   # alpha_id of the closest match
    similarity_score: float        # 0.0–1.0; ≥0.8 = too similar
    reason: str                    # human-readable explanation

class ResearchSuggestion(TypedDict):
    direction: str          # short label, e.g. "value + low-vol"
    hypothesis: str         # investment thesis
    formula: str            # ready-to-run formula string
    features: list[str]     # features referenced in formula
    parent_id: str | None   # which existing alpha to branch from
    rationale: str          # why this direction given prior results
```

---

## Step 2 — `core/reflection.py`

**Signature change:**
```python
# Before
def generate_reflection(alpha, metrics, verdict, failure_reason) -> str

# After
def generate_reflection(alpha, metrics, robustness, verdict, failure_reason) -> str
```

**Prompt update** — add robustness block to `_build_prompt` and reorder the 4-part output
format the LLM is asked to follow:

```
Observation: <what the numbers show>
Failure Reason: <why it failed or "N/A">
Possible Explanation: <mechanistic hypothesis for the behaviour>
Next Mutation: <one concrete, specific change to try>

Robustness:
- Sector Stability: {robustness["sector_stability"]:.4f}
- Subperiod Stability: {robustness["subperiod_stability"]:.4f}
- Market Regime Sharpe: {robustness["market_regime_sharpe"]:.4f}
- Placebo Score: {robustness["placebo_score"]:.4f}
```

**Fallback** — update `_rule_based_reflection` to accept and surface the same 4 fields.

---

## Step 3 — `core/similarity.py` (new)

Purely structural — no LLM needed. Formulas are short and token-comparable.

```python
def check_similarity(
    new_alpha: AlphaConfig,
    store: ExperimentStore,
    threshold: float = 0.8,
) -> SimilarityResult:
```

**Algorithm:**
1. **Feature Jaccard** — `|features_new ∩ features_existing| / |features_new ∪ features_existing|`
2. **Formula token Jaccard** — tokenize formula with `re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)`;
   compute Jaccard on the resulting token sets
3. **Combined score** — `0.5 * feature_jaccard + 0.5 * formula_token_jaccard`
4. **Config bonus** — if same universe + rebalance + neutralization, add 0.1 (capped at 1.0)
5. If `combined_score >= threshold` → `is_unique=False`

Return the record with the highest combined score as `most_similar_id`.

---

## Step 4 — `core/mutator.py` (new)

```python
def generate_mutation(
    parent_id: str,
    store: ExperimentStore,
) -> AlphaConfig:
```

**LLM path:**
- Fetch parent record via `store.load_by_id(parent_id)`
- Build prompt: parent formula, features, metrics (IC, ICIR, Sharpe, turnover), verdict,
  failure_reason, full reflection, and the list of available features from `validator.ALLOWED_FEATURES`
- Instruct DeepSeek to output a single JSON object (AlphaConfig schema)
- **Critical constraint in prompt:** `ts_mean`, `ts_std`, and `delta` are NOT implemented —
  do not use them in the formula
- Parse JSON from response (strip markdown fences), call `validate_alpha()`, retry once on
  invalid JSON
- Set `parent_id` on the output config; generate `alpha_id` as `{parent_id}_mut_{timestamp}`

**Rule-based fallback** (if LLM fails):
- `turnover > 300` → change `rebalance` from `"monthly"` to `"quarterly"`
- `Sharpe < 0.3` and only momentum features → append `+ 0.3 * rank(EBITDA_MARGIN)` to formula
- `ICIR < 0.2` → append `* (1 - rank(VOL_20D))` as a volatility damper
- Otherwise → swap `MOM12_1` ↔ `MOM6_1` or add `rank(LIQUIDITY)` as a liquidity screen

---

## Step 5 — `core/planner.py` (new)

```python
def plan_next_research(
    store: ExperimentStore,
    n: int = 3,
) -> list[ResearchSuggestion]:
```

**LLM path:**
- Load all records via `store.load_all()`
- Build a compact summary table: `alpha_id | formula | verdict | Sharpe | ICIR | turnover | failure_reason`
- Include the `reflection` (Next Mutation section) for each record
- Include the full `ALLOWED_FEATURES` list with one-line descriptions
- Ask DeepSeek to return a JSON array of N `ResearchSuggestion` objects
- Validate each suggestion's formula and features with `validate_alpha()`, drop invalid ones
- If fewer than N survive validation, fall back to rule-based suggestions to fill the gap

**Rule-based fallback** (if LLM fails or returns <1 valid suggestion):
- If all prior alphas failed on turnover → suggest quarterly rebalance config
- If no quality features tried → suggest `rank(EBITDA_MARGIN) + rank(SALES_GROWTH)`
- If no value features tried → suggest `rank(PRICE_TO_SALES) * -1`

---

## Step 6 — `scripts/run_experiment.py`

Two changes:

**Change A — Similarity gate (new Step 1.5):**
```python
print("[ Step 1.5 ] Checking similarity against prior alphas...")
from core.similarity import check_similarity
sim = check_similarity(alpha, store)
if not sim["is_unique"]:
    print(f"  WARNING: Alpha is {sim['similarity_score']:.0%} similar to {sim['most_similar_id']}")
    print(f"  {sim['reason']}")
    print("  Use --force to run anyway.")
    if not getattr(args, "force", False):
        sys.exit(1)
```

Add `--force` flag to argparse.

**Change B — Pass robustness to reflection:**
```python
# Before
reflection = generate_reflection(alpha, metrics, verdict, failure_reason)

# After
reflection = generate_reflection(alpha, metrics, robustness, verdict, failure_reason)
```

---

## Step 7 — `scripts/mutate_alpha.py` (new)

```
python scripts/mutate_alpha.py --parent alpha_001 [--run] [--db db/experiments.db]
```

- Calls `generate_mutation(parent_id, store)`
- Prints the generated AlphaConfig as pretty JSON
- Saves to `experiments/<new_alpha_id>.json`
- If `--run` flag: calls `run_experiment.py` on the new config via `subprocess`

---

## Step 8 — `scripts/plan_next.py` (new)

```
python scripts/plan_next.py [--n 3] [--db db/experiments.db] [--save]
```

- Calls `plan_next_research(store, n)`
- Prints each suggestion (direction, hypothesis, formula, rationale)
- If `--save`: writes each suggestion as a JSON file to `experiments/`

---

## DeepSeek Prompt Design Notes

### Shared constraints for all LLM prompts:
- Temperature `0.3` (deterministic, focused)
- `max_tokens`: 600 for planner/mutator, 400 for reflection
- System prompt: "You are a quantitative research analyst..."
- **Formula constraints (mutation + planner):** Only use `rank`, `zscore`, `log`, `abs`,
  `sign` and arithmetic operators. `ts_mean`, `ts_std`, `delta` are NOT implemented.
- **Output format:** Always JSON — wrap in `try/except json.loads()` with one retry

### JSON extraction helper (shared across mutator + planner):
```python
def _extract_json(text: str) -> dict | list:
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    return json.loads(text)
```

---

## Critical Files

| File | Role |
|---|---|
| `core/types.py` | Add `SimilarityResult`, `ResearchSuggestion` |
| `core/reflection.py` | Add robustness param + update prompt |
| `core/similarity.py` | New — Jaccard similarity gate |
| `core/mutator.py` | New — DeepSeek alpha mutation |
| `core/planner.py` | New — DeepSeek research planner |
| `scripts/run_experiment.py` | Add similarity step 1.5 + robustness to reflection call |
| `scripts/mutate_alpha.py` | New CLI |
| `scripts/plan_next.py` | New CLI |

**Reused from existing code:**
- `ExperimentStore.load_all()` / `load_by_id()` — `core/memory.py`
- `validate_alpha()` — `core/validator.py` (validates mutator/planner output)
- `ALLOWED_FEATURES`, `ALLOWED_FUNCTION_NAMES` — `core/validator.py` (fed into LLM prompts)
- DeepSeek client pattern — `core/reflection.py` (replicate `OpenAI(api_key, base_url)` setup)

---

## Verification

```bash
# 1. Reflection now includes robustness (re-run existing experiment)
python scripts/run_experiment.py --config experiments/sample_alpha_001.json
# Expected: Step 5 reflection references sector_stability / subperiod_stability values

# 2. Similarity check blocks a duplicate
python scripts/run_experiment.py --config experiments/sample_alpha_001.json
# Expected: Step 1.5 warns "100% similar to alpha_001" and exits (unless --force)

# 3. Mutation generator produces a valid child
python scripts/mutate_alpha.py --parent alpha_001
# Expected: prints valid AlphaConfig JSON with parent_id="alpha_001", new alpha_id

# 4. Planner suggests 3 new directions
python scripts/plan_next.py --n 3
# Expected: 3 ResearchSuggestion entries, each with valid formula and features

# 5. End-to-end: mutate + run
python scripts/mutate_alpha.py --parent alpha_001 --run
# Expected: new experiment saved to DB, graph now shows edge alpha_001 → new_alpha_id
```
