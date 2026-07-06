# AlphaResearchBot — Version 4 Plan

Composite Scoring + Bandit Scheduler + Autonomous Loop

## Context

V3's fixed-threshold verdict system (`core/decision.py` hard/soft gates → promising/revise/failed) caused the LLM to game metrics with increasingly complex formulas (documented in `Reflections.md`: "the bottleneck wasn't the model's reasoning ability — it was my objective function"). Also, the explore-vs-mutate decision is currently made by a human choosing between `plan_next.py` and `mutate_alpha.py`.

V4 does two things:
1. **Objective redesign** — replace binary cliff verdicts with a continuous composite score (0–100) that also rewards formula simplicity, novelty, and robustness (the `robustness` param is currently *unused* in `decide()`).
2. **Close the loop** — a Thompson-sampling bandit scheduler decides mutate-vs-explore, driven by an autonomous loop runner.


**Design decisions** (all four are cheap to swap later):
- Continuous score + verdicts derived from score bands (downstream consumers keep working)
- Thompson sampling (Beta posteriors, stdlib `random.betavariate`, no new deps)
- Autonomous `scripts/run_loop.py` (existing CLIs stay usable manually)
- Mutation reward = improvement over parent; explore reward = score vs. running baseline

---

## Step 1 — Types (`core/types.py`)

```python
class SubScores(TypedDict):
    performance: float; implementation: float; robustness: float
    simplicity: float; novelty: float          # each in [0,1]

@dataclass
class AlphaScore:
    total: float                 # 0–100, DIRECTIONAL score (hypothesis as stated)
    signal_strength: float       # 0–100, max(total, inverted_total - INVERSION_PENALTY)
    preferred_direction: int     # +1 (as stated) or -1 (inverted works better)
    sub_scores: SubScores        # directional sub-scores
    verdict: Verdict             # derived from bands on signal_strength
    failure_reason: str | None
    fatal: bool                  # catastrophic gate fired
```

- `Verdict` literal gains `"revise_invert"` — the signal is real but the hypothesis direction was wrong; treated as revise-equivalent everywhere downstream (parent-pool eligible), but visually and semantically distinct so the system learns the economic intuition was inverted, not merely weak.
- `ExperimentRecord` gains `score: NotRequired[float | None]` (directional), `signal_strength: NotRequired[float | None]`, `preferred_direction: NotRequired[int | None]`, `sub_scores: NotRequired[dict | None]`.
- `FailureCategory` literal gains `"too_complex"`, `"low_novelty"`, `"wrong_direction"`.

## Step 2 — Formula complexity (`core/formula_validator.py`)

```python
def formula_complexity(formula: str) -> int:
    """n_calls + max_expr_depth + n_distinct_columns."""
```
- `n_calls` = count of `ast.Call` in `ast.parse(formula, mode="eval")` (formulas are eval-able Python).
- `depth` = max nesting depth over `Call/BinOp/UnaryOp/Compare` nodes.
- `n_cols` = distinct raw columns referenced (reuse existing `_tokenize` + `AVAILABLE_RAW_COLUMNS`).
- `SyntaxError` fallback: `len(_tokenize(formula)) // 2`.
- Calibration: `rank(X) * -1` → 4; the quality+value fallback formula → ~11; gamed nested formulas ≥ 20.

## Step 3 — `core/decision.py`: `score_alpha()`

Keep old constants + `check_tier1`/`decide` (deprecated comment; `memory_analyzer.py` still imports `ICIR_SOFT`/`TURNOVER_MAX`). Add linear ramp helpers `_ramp(x, lo, hi)` and `_ramp2(x, lo, mid, hi)` (0 at lo, 0.5 at mid, 1 at hi — **every mid is an old soft threshold, every lo an old hard threshold**, so the score passes through 0.5/0.0 exactly where the cliffs were).

```python
def score_alpha(metrics, robustness, formula, similarity_score: float,
                portfolio_returns: list[float] | None = None) -> AlphaScore
```
Novelty passed as a precomputed float (caller already runs `check_similarity`) — keeps it a pure, unit-testable function. `portfolio_returns` (from `BacktestResult`) enables exact inverted-direction metrics (below); when None (lazy rescore of old rows), only the directional score is computed and `signal_strength = total`, `preferred_direction = +1`.

**Direction-aware scoring — separate "predictive signal exists" from "direction is correct":**

A negative IC/Sharpe alpha may still be useful because it can be inverted — but blind `abs()` everywhere would hide a bad research story ("high profitability outperforms" scoring strongly negative means the hypothesis was *wrong*, not that the alpha is good). So compute two scores:

- `total` (directional) = composite on the raw metrics as stated. This is the **hypothesis evaluation** — sign preserved, so the system learns whether the economic intuition was correct. Stored as `score`; feeds reflection/planner prompts.
- `inverted_total` = same composite on sign-flipped metrics: `IC_mean → −IC_mean`, `ICIR → −ICIR`, `monotonicity → −monotonicity`, `Sharpe → −Sharpe`; `deflated_sharpe` and `max_drawdown` recomputed **exactly** from the negated `portfolio_returns` series (negating per-period returns changes drawdown paths and skew — they are not simple sign flips); turnover, robustness, simplicity, novelty unchanged (direction-agnostic; robustness stability/placebo treated as direction-agnostic — an approximation, noted).
- `INVERSION_PENALTY = 5.0` points — inverting is not free (the hypothesis was wrong; slight overfit risk in picking the better of two directions).

```python
signal_strength     = max(total, inverted_total - INVERSION_PENALTY)
preferred_direction = +1 if total >= inverted_total - INVERSION_PENALTY else -1
```

**Parent-pool survival uses `signal_strength`** (abs-like metrics + inversion penalty); **hypothesis evaluation uses `total`** (sign kept).

**Sub-scores:**
- *Performance* (IC .30, ICIR .25, Sharpe .30, mono .15): `_ramp2(IC_mean, 0, .02, .05)`; `_ramp2(ICIR, 0, .30, 1.0)`; Sharpe = `0.7*_ramp2(Sharpe, 0, .5, 1.5) + 0.3*_ramp2(deflated_sharpe, 0, .5, 1.5)` (deflated blend directly counters overfit gaming); `_ramp2(mono, -.2, .3, .8)`.
- *Implementation* (.5/.5): turnover 1.0 below 0.30, →0.5 over [.30,.70], →0.0 over [.70,.90]; drawdown `_ramp2(max_dd, -.40, -.25, -.10)`.
- *Robustness* (equal thirds): `subperiod_stability`, `placebo_score` (both already [0,1] per `core/robustness.py`), regime = mean of `_ramp2(v, -.5, 0, 1.0)` over `market_regime_sharpe` values, neutral 0.5 if dict empty (yfinance missing — don't punish). `robustness=None` (fatal path) → 0.0.
- *Simplicity*: `1 - _ramp(formula_complexity(formula), 4, 20)`.
- *Novelty*: `1 - similarity_score`.

**Weights** (module constants): performance .45, robustness .20, implementation .15, simplicity .10, novelty .10. `total = 100 * Σ wᵢsᵢ`.

**Fatal gates** (only truly dead cases): `abs(IC_mean) < 0.005 and abs(Sharpe) < 0.10` (dead signal — no edge in either direction, so neither the directional nor inverted score can save it) or `max_drawdown ≤ -0.40` **in the preferred direction** (a catastrophic drawdown of the raw strategy is irrelevant if we'd trade it inverted). Fires → `fatal=True`, verdict failed, `signal_strength = min(signal_strength, 25)`, sub-scores still computed/stored. Constants: `FATAL_IC_ABS = 0.005`, `FATAL_SHARPE_ABS = 0.10`.

**Verdict bands — applied to `signal_strength`**: `≥ 65 promising`, `35–65 revise`, `< 35 failed`. When the band is revise-or-better AND `preferred_direction == -1` → verdict `"revise_invert"` (never `"failed"` just because the sign was wrong). Sanity anchor: all metrics at old soft thresholds + neutral robustness + simple novel formula → ≈52 → revise (matches old judgment). A strong contrarian alpha (directional total ≈ 20, inverted ≈ 80) → signal_strength 75 → `revise_invert`, parent-pool eligible.

**failure_reason** for non-fatal rows names the weakest sub-score, e.g. `"score 41.2 — weakest: simplicity (0.21, complexity 18)"` — this string feeds the mutator prompt (the anti-gaming pressure point). For `revise_invert`: `"signal is real but direction is inverted (directional 21.3, inverted 76.1) — hypothesis direction was wrong; flip the formula sign and restate the hypothesis"`.

**Robustness always runs in V4** (score needs it; cost is seconds vs. the backtest). Exception: fatal dead-signal skips it (keep the zeroed placeholder dict from `run_experiment.py:116` for persistence). `check_tier1` no longer called by the pipeline; the tier1-revise-caps-promising reconciliation (`run_experiment.py:145-146`) is deleted.

## Step 4 — Persistence (`core/memory.py`)

- Four new try/except migrations (existing `batch_id` pattern at `memory.py:42-45`): `ALTER TABLE experiments ADD COLUMN score REAL;`, `... signal_strength REAL;`, `... preferred_direction INTEGER;`, `... sub_scores TEXT;`.
- `save_experiment` / `_row_to_record`: write/read the four fields (NULL → None for old rows).
- New table + methods (ExperimentStore stays the single SQLite owner):
  ```sql
  CREATE TABLE IF NOT EXISTS bandit_state (
      arm_id TEXT PRIMARY KEY, alpha REAL NOT NULL, beta REAL NOT NULL,
      pulls INTEGER NOT NULL DEFAULT 0, last_updated TEXT);
  ```
  `load_bandit_state() -> dict[str, dict]`, `upsert_bandit_arm(arm_id, alpha, beta, pulls)`.
- `core/memory_analyzer.py`: add `effective_score(record) -> float` — stored `signal_strength` if set, else stored `score`, else recompute `score_alpha(metrics, robustness, formula, similarity_score=0.5)` (neutral novelty; no portfolio_returns for old rows → directional only). Parent-pool ranking and bandit rewards use this (signal-strength semantics). No DB backfill needed.

## Step 5 — Refactor pipeline into `core/experiment.py`

```python
@dataclass
class ExperimentOutcome:
    status: Literal["completed", "duplicate", "validation_failed", "backtest_error"]
    record: ExperimentRecord | None
    similarity: SimilarityResult | None
    error: str | None

def run_single_experiment(alpha, store, loader=None, force=False, verbose=True) -> ExperimentOutcome
```

Moved from `scripts/run_experiment.py:main` (prints gated on `verbose`, no `sys.exit`):
validate → similarity (**hard abort only at ≥ 0.90** `HARD_DUPLICATE_THRESHOLD`; below that similarity flows into novelty sub-score instead of aborting) → backtest → fatal pre-check / robustness → `score_alpha(..., portfolio_returns=backtest_result["portfolio_returns"])` → reflection → save record with `score`/`signal_strength`/`preferred_direction`/`sub_scores`/derived verdict.

`scripts/run_experiment.py` becomes a thin CLI wrapper: same args, maps non-completed statuses to `sys.exit(1)`, prints score breakdown table. `--force` keeps working; `mutate_alpha.py --run` (subprocess) unaffected.

## Step 6 — Bandit (`core/scheduler.py`)

```python
EXPLORE_ARM = "__explore__"
MAX_PARENT_ARMS = 5          # top-K eligible parents by effective_score
PARENT_SCORE_FLOOR = 35.0    # out of the failed band
COLD_START_MIN = 3           # force explore until store has ≥ 3 experiments
REWARD_HALF_RANGE = 25.0
EXPLORE_PRIOR = (2.0, 1.0)   # optimistic bootstrap
# parent arm prior: Beta(1 + parent_score/100, 1)

class ThompsonScheduler:
    def __init__(self, store)                      # loads bandit_state
    def eligible_parent_arms(self) -> list[str]    # "mutate:<alpha_id>"
    def select_action(self) -> tuple[Literal["explore","mutate"], str | None]
    def update(self, arm_id, reward) -> None       # fractional Beta update, persist immediately
    def reward_for(self, action, parent_record, outcome) -> float
```

- **Fractional-Beta Thompson sampling for continuous rewards**: rewards are continuous in [0,1] (not Bernoulli), so `update` uses the fractional posterior update `α += r; β += (1 − r); pulls += 1`. This treats a reward r as "r successes + (1−r) failures" — the standard extension of Beta-Bernoulli TS to bounded continuous rewards (equivalent in expectation to Bernoulli-sampling with probability r, but lower variance). No binarization/thresholding of rewards anywhere. `update` must assert/clip r into [0,1] before applying.
- Eligibility: verdict ∈ {promising, revise, revise_invert} and `effective_score ≥ 35` (signal_strength semantics — an inverted-but-real alpha survives into the pool), top 5 by score. Re-checked each iteration.
- Selection: `random.betavariate(α, β)` per arm (explore + eligible parents), argmax. `< 3` experiments → forced explore.
- Rewards (symmetric, all on **signal_strength** via `effective_score` — a mutation that flips a wrong-direction parent into a correct-direction child is rewarded for any strength gained, not double-counted for the flip itself): mutate `r = clip(0.5 + (child − parent)/50, 0, 1)`; explore `r = clip(0.5 + (child − baseline)/50, 0, 1)` where baseline = mean `effective_score` over store (50.0 if empty). `duplicate`/`validation_failed`/`backtest_error` → `r = 0.0` (bandit learns to avoid arms producing garbage).

## Step 7 — Loop runner (`scripts/run_loop.py`)

Args: `--iterations` (default 10), `--db`, `--no-cache`, `--max-consecutive-failures` (default 3), `--sleep` (default 0).

Setup: one shared `ExperimentStore`, `DataLoader` (parquet cache makes iter 2+ fast), `ThompsonScheduler`, and **one `batch_id = f"loop_{ts}"` for the whole run** → one ring in the graph viz (per-iteration batches would spawn N one-node rings; lineage edges come from `parent_id`, unaffected).

Per iteration:
1. `select_action()`.
2. explore → `plan_next_research(store, n=1)`; **extract the suggestion→config logic inlined at `scripts/plan_next.py:66-85` into `core/planner.py:suggestion_to_config(suggestion, batch_id, base_config)`** and have `plan_next.py` reuse it. mutate → `generate_mutation(parent_id, store)`, stamp batch_id. Generation exception → `update(arm, 0.0)`, count consecutive failure, continue.
3. Write config JSON to `experiments/<alpha_id>.json` (reproducibility, matches existing convention).
4. `run_single_experiment(...)`.
5. `reward_for(...)` → `update(...)` (persisted per-iteration — crash-safe).
6. One-line log: `[3/10] mutate:alpha_007 → child | score 58.3 (parent 51.0) | reward 0.65 | revise`.

Stop: iterations done, N consecutive failures, or `KeyboardInterrupt` → print summary (per-arm α/β/pulls, best score, verdict counts), exit 0.

## Step 8 — Downstream updates

- `core/memory_analyzer.py`: `best_experiments` ranked by `effective_score` (add `score`/`signal_strength`/`preferred_direction` to emitted dicts); trend line becomes "Best score so far: 78 (alpha_x, Sharpe 1.2)". `classify_failure`: `revise_invert` → `"wrong_direction"` (before the metric ladder — it's the defining trait); keep existing ladder, append `sub_scores["simplicity"] < 0.3 → "too_complex"`, `sub_scores["novelty"] < 0.2 → "low_novelty"` (guard None on old rows).
- `core/reflection.py`: `generate_reflection` accepts `AlphaScore`; prompt gains a Composite Score block (directional total + signal_strength + preferred_direction + 5 sub-scores + "sub-scores below 0.4 are the priority to fix"). For `revise_invert` the prompt must ask the LLM to *restate the economic hypothesis in the opposite direction* — the point is learning why the intuition was backwards, not just flipping a sign. Fix dead branch at `reflection.py:237` (`verdict == "inconclusive"` can never fire) → `verdict == "revise" and noise_risk == "high"`.
- `core/mutator.py::_build_mutation_prompt`: parent score + sub-scores + "If simplicity < 0.5, the mutation MUST reduce formula complexity, not add terms." (directly counters the V3 failure mode). If parent `preferred_direction == -1`: "The signal direction is inverted — the primary mutation is to negate the formula (multiply by -1) AND restate the hypothesis accordingly; do not add complexity."
- `core/planner.py` prompt: best-lines include score; "Prefer simple formulas (≤ 3 operators); complexity is penalized."
- `core/graph.py`: node attrs `score`, `signal_strength` (−1.0 sentinel for None), `preferred_direction` (0 sentinel), `sub_scores`.
- `core/visualization.py`: add all to `nodes_data`; new "Composite Score" section above Tier 1 in the detail panel (big signal_strength + directional total + direction indicator "↕ inverted" when −1, one metricRow per sub-score, classes good ≥.6 / warn .4–.6 / bad <.4); node label shows rounded signal_strength when ≥ 0. New verdict color for `revise_invert` (e.g. #3498db blue) added to `_VERDICT_COLORS` and the summary-bar chips; dashed-failed border unchanged.
- `scripts/export_graph.py`: top-filtering sorts by score, Sharpe fallback.

## Files

| File | Change |
|---|---|
| `core/types.py` | AlphaScore (dual-direction), SubScores, Verdict + `revise_invert`, ExperimentRecord fields, FailureCategory |
| `core/formula_validator.py` | `formula_complexity()` |
| `core/decision.py` | `score_alpha()`, ramps, weights, bands, fatal gates |
| `core/memory.py` | score/sub_scores migrations, bandit_state table + methods |
| `core/memory_analyzer.py` | `effective_score()`, score-ranked best, new failure categories |
| `core/experiment.py` | **new** — `run_single_experiment()` |
| `core/scheduler.py` | **new** — `ThompsonScheduler` |
| `core/planner.py` | `suggestion_to_config()` extraction, prompt update |
| `core/mutator.py`, `core/reflection.py` | prompt updates, dead-branch fix |
| `core/graph.py`, `core/visualization.py`, `scripts/export_graph.py` | score display |
| `scripts/run_experiment.py` | thin CLI wrapper |
| `scripts/run_loop.py` | **new** — autonomous loop |
| `scripts/plan_next.py` | use extracted `suggestion_to_config` |

## Verification

1. **Scoring unit tests** (`tests/test_decision.py`): ramps return 0.5 at each old soft threshold / 0.0 at hard; monotonicity (raising any metric never lowers total); fatal gates — dead signal (|IC| < 0.005 and |Sharpe| < 0.10) → fatal/failed; **direction separation** — strong contrarian alpha (IC −0.03, Sharpe −0.8 with matching negative portfolio_returns): directional `total` stays low (hypothesis was wrong), `signal_strength` ≈ inverted − 5 is high, `preferred_direction == -1`, verdict `revise_invert` (NOT failed/fatal), failure_reason mentions flipping sign + restating hypothesis; symmetric strong-positive alpha → `preferred_direction == +1` and `signal_strength == total`; drawdown −0.45 in preferred direction → fatal; band edges (all-soft-thresholds alpha lands in revise, all-strong ≥ 65); `formula_complexity("rank(X) * -1") == 4` + SyntaxError fallback.
2. **Calibration check before committing bands**: back up `db/experiments.db`, run a read-only script scoring all existing rows; eyeball that old promising/revise/failed rows mostly stay in their bands — tune 65/35 if not.
3. **Migration**: load existing db → old rows have `score=None`, no exceptions; `bandit_state` exists.
4. **CLI regression**: `python scripts/run_experiment.py --config experiments/sample_alpha_001.json` end-to-end; rerun same config → duplicate abort at ≥0.9, `--force` overrides.
5. **Bandit test** (`tests/test_scheduler.py`): seeded RNG, two arms fed rewards 0.9 vs 0.1 × 30 → selection converges to good arm; fractional update check — after `update(arm, 0.7)`, α increased by exactly 0.7 and β by 0.3 (no binarization); cold start forces explore.
6. **Loop without LLM key**: unset `DEEPSEEK_API_KEY`, `python scripts/run_loop.py --iterations 3` — rule-based fallbacks carry it, 3 rows share one `loop_*` batch_id, bandit pulls updated. Then `export_graph.py` → new ring renders, score section shows, old score-less nodes still render.
7. **Ctrl-C mid-loop** → summary prints, db/bandit state consistent.

## Risks / notes

- Picking `max(directional, inverted − penalty)` is itself a mild multiple-testing step (2 hypotheses per experiment); the 5-point INVERSION_PENALTY plus the deflated-Sharpe blend are the offsets. If revise_invert verdicts dominate the pool, raise the penalty.
- Robustness sub-score is treated as direction-agnostic when scoring the inverted variant (subperiod stability / placebo are computed on the raw signal) — an approximation; exact would require re-running robustness on the negated signal, not worth the cost.
- Rule-based fallbacks (5 fixed suggestions + mutation ladder) will produce near-duplicates in a long LLM-less loop → mostly 0 rewards; fine for testing, flagged in loop output.
- Lazily-rescored old rows use neutral novelty 0.5 — not perfectly comparable to new scores; affects only sorting/baseline.
- Weights and bands (65/35) are module constants — tuning is a one-line change; verify step 2 calibrates them empirically.
- Don't run two loops on the same SQLite file concurrently.
