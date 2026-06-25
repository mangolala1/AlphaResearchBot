# AlphaResearchBot — Version 3.8 Plan

### Data Issues
SimFin provides data for past 7 years, so my start date right now is set to 2021-01-01 to avoid missing data.
Try: Databento

### Data Processing
- Add checks for missing data and handle them gracefully (e.g., skip, impute, or flag).
- Add normalization steps as well as handling of outliers after calculating feature values

### Evaluation Metrics
- Add monoticity based on quintiles: corr(rank([1,2,3,4,5]), rank(quintile_returns))
- Add fitness (WQ-style): fitness = sharpe * np.sqrt(abs(ann_return) / max(avg_turnover, 0.125))
- Change the turnover calculation to be more accurate: turnover = sum(abs(weights_t - weights_t-1)) / 2
- Change Deflated sharpe ratio calculation
- Add regression t-stats? Using Robust linear model to avoid winsorization
- Add rolling IC

### Robustness Checks
- Redesign sector / market regime / placebo score functions 
- Add volatility regimes：source VIX data from yfinance
- Redesign what should be part of the pass/fail logic, create a tiered system:
- Tier 1 (Predictive power): IC mean, ICIR, monotonicity. If this tier does not show predictive power, we reject and do 
not proceed to the next tier.
- Tier 2 (Implementation): Sharpe, Turnover, Drawdown.
- Tier 3 (Diagnostics): Bull/bear sharpe, IC by sector, rolling IC, placebo, rolling sharpe. This tier is for diagnostics 
only, not for pass/fail decisions.

---

## Implementation Plan: Tiered Decision Logic

### Verdicts
Three verdicts replace the old two:

| Verdict | Meaning |
|---------|---------|
| `failed` | No predictive evidence, or truly unworkable. Dead end — mutation agent should not build on it. |
| `revise` | Weak but alive. Predictive signal exists but implementation needs work. Best mutation candidates. |
| `promising` | Passes all tiers. |

### Architecture

```
run_backtest() → metrics
     ↓
check_tier1(metrics)
  Hard fail (dead signal) → verdict="failed", skip robustness, STOP
  Weak (alive but soft)   → verdict="revise", still compute robustness
  Pass                    → continue
     ↓
run_robustness() → robustness
     ↓
decide(metrics, robustness)   ← Tier 2 only
  Severe fail → verdict="failed"
  Tradeable issue → verdict="revise"
  All pass    → verdict="promising"
     ↓
Tier 3 in robustness + metrics — read by reflection/mutation LLM, no verdict effect
```

### Tier 1 — Predictive Power (hard fail = truly dead signal)

Hard fail conditions (any one → `"failed"`, skip robustness):
- `IC_mean <= 0` — signal has no positive predictive edge at all
- `ICIR <= 0` — IC is net negative when averaged over all periods
- `monotonicity <= -0.2` — quintile ordering is meaningfully reversed

Soft fail conditions (any one → `"revise"`, still proceed):
- `IC_mean <= 0.02` — signal is real but weak
- `ICIR <= 0.30` — IC is inconsistent
- `monotonicity <= 0.3` — quintile staircase is shallow

If all soft conditions pass → continue to Tier 2.

### Tier 2 — Implementation (severe = `"failed"`, tradeable issue = `"revise"`)

| Metric | Hard fail (`"failed"`) | Soft fail (`"revise"`) |
|--------|------------------------|------------------------|
| Sharpe | < 0 | < 0.50 |
| turnover | — | > 0.70 (fraction, 0–1) |
| max_drawdown | < -0.40 | < -0.25 |

Rationale: negative Sharpe or catastrophic drawdown is unworkable even with mutation. High turnover or moderate drawdown just needs implementation refinement.

---

## HTML Visual Design

### Context
The current graph export uses PyVis defaults: plain circles with alpha_id as label, all detail hidden behind a hover tooltip that vanishes on mouse-out. For general usability, the graph needs to show key metrics at a glance on each node, label edges with the mutation type, and reveal full details in a persistent side panel on click.

### Approach
Drop PyVis's `save_graph()` and generate a custom standalone HTML file embedding vis-network.js via CDN. `core/visualization.py` keeps the same public signature — everything inside changes. `graph.py` and `export_graph.py` are untouched.

### Page Layout

```
┌──────────────────────────────────────────┬──────────────────┐
│         GRAPH CANVAS  (70%)              │  DETAIL PANEL    │
│                                          │  (30%)           │
│  [alpha_001]────────►[alpha_002]         │  alpha_002       │
│      │                    │             │  ─────────────── │
│      ▼                    ▼             │  Verdict: REVISE │
│  [alpha_003]         [alpha_004]        │  Sharpe:  0.71   │
│                                          │  ...             │
└──────────────────────────────────────────┴──────────────────┘
```

Dark #1a1a2e background across both panels. Side panel always visible, shows placeholder until a node is clicked, scrollable.

### Node Design

- Shape: `box` (rectangular, fits multi-line text)
- Label (always visible on node):
  ```
  alpha_001
  S: 0.82  IC: 0.47
  ```
- Color: existing verdict palette — green / yellow / red
- Border: `dashes: true` for failed nodes — marks dead ends without relying on colour alone
- Size: fixed ~120px width, auto height

### Edge Design

- Label: first 35 chars of `mutation` field (e.g. `"add momentum factor"`)
- Font: 10px, #aaaaaa
- Color: #666688 (purple-tinted gray, readable on dark bg)
- Arrow: directed, to-end only

### Detail Panel (on click)

Populated by a `selectNode` vis-network event listener. Shows:
- Alpha ID + verdict badge (coloured dot)
- Hypothesis and formula (monospace block)
- Tier 1: IC_mean, ICIR, Monotonicity
- Tier 2: Sharpe, Turnover, Max Drawdown — with ⚠ next to metrics that triggered failure
- Tier 3: Subperiod Stability, Placebo Score, Deflated Sharpe
- Failure reason, mutation description, full reflection (scrollable)

Clicking empty canvas resets panel to placeholder.

### Files to Change

| File | Change |
|------|--------|
| `core/visualization.py` | Full rewrite — custom HTML generator, drop PyVis |
| `core/graph.py` | Add missing metric attrs to `add_node()`: IC_mean, monotonicity, turnover, max_drawdown |

### Verification
1. `python scripts/export_graph.py` then open `reports/research_graph.html`
2. Nodes show ID + metrics label; edges show mutation text
3. Click node → side panel populates with full detail; click canvas → resets to placeholder
4. Screenshot is clean enough for README embed

