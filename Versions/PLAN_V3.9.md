# AlphaResearchBot — Version 3.9 Plan

### data_loader.py
- Issue: 
  - lookahead bias for fundamentals data that are usually reported with a lag, need to know the release date as well.
  - Fix: SimFin provides the `REPORT_DATE` column for each fundamental data point, which can be used to filter out data that would not have been known at the time of the backtest.
  - ~~- Simple fix: shift fundamentals data by 1 quarter to avoid lookahead bias.~~
- Issue2: Data loading of fundamentals data
  - The current implementation of `_fetch_fundamentals` in `core/data_loader.py` filters the fundamentals data based on the `start_date`, which can lead to missing data for the first few months of the backtest. This is because companies may not have filed their Q4 reports until late January or February, resulting in a lack of seed values for forward-filling.
  - Fix: Extend the filter to include data from one year prior to the `start_date`, allowing for proper forward-filling and ensuring that all stocks have valid fundamental data at the start of the backtest.

Future steps: maybe create more features to take advantage of both quarterly data and TTM data.

### data_process.py
- Separately loading cashflow and income statement
- Kept more data columns and add those to available raw data in formula_validator.py 

Question: Not sure if I should standardize in data_process.py?
Future steps: might need better logic or flexibility for na values handling, maybe keep them at this stage and later on decide how to handle them in signal calculation, e.g., drop or replace with 0 or replace with mean or median by sector, etc.

### formula_validator.py
- Added more data columns 
- Added more operators
- Changed the operators' definition to also show the parameters each can take

### ~~Features.py~~ signal_calculation.py
- Issue1: Winsorization of computed features
  - The current implementation of winsorization is scattered across different feature branches, leading to inconsistencies in how outliers are handled. Some features, such as `VOL_20D`, `LIQUIDITY`, and raw fundamental pass-throughs, do not have any clipping applied, allowing extreme outliers to flow into formula evaluation.
  - Fix: Centralize the winsorization process in the `compute_features` loop, applying it to every feature panel after computation, except for raw price inputs used by momentum features. This will ensure that all computed features are consistently winsorized before use.
- Issue2: Current winsorization is applied over time per ticker, leading to lookahead bias. We want to winsorize cross-sectionally per date.
- Issue3: _compute_single_feature() feels suboptimal (hard-coded registry of each feature), potential issues for future transform to fully autonomous
  - Fix:
      - Question: should I process all the data i got at once and save it locally, or should i create a feature registry and make the winsorize and standardize part of configurations upon each prompt?
      - add a data_process.py file just to process the data columns loaded from data_loader.py, drop na rows, apply winsorization and standardization, drop na rows again in the end, and then cache the processed data locally. Maybe transform into a pivot table so that the column header is the feature name and the index is the date, and the values are the feature values for each ticker. This will make it easier to work with the data and apply transformations in a consistent manner.
      - and then evaluate the formula first, and then if valid, use the processed data and the supported operands and functions to compute
      - Keep the data multiindex panel (date, ticker), not just date index
        - create feature registry, e.g.: 
```
FEATURE_REGISTRY = {
    "EBITDA_MARGIN": {
        "inputs": ["OPER_INCOME_LTM", "DA_LTM", "SALES_LTM"],
        "process": ["winsorize", "zscore"],
        "family": "profitability",
    },
    "MOM12_1": {
        "inputs": ["ADJUSTED_PRICE"],
        "process": ["winsorize", "zscore"],
        "family": "momentum",
    },
}
```
- Issue1: na handling in signal calculation: operators like rolling() might create na values in the first few rows, and then the formula evaluation will create na values for those rows. 
- Fix: Should pre-load the data for one year so that this won't happen, and also add a check for na values before applying the formula?
- Dropping formula and raw formula design, keeping only raw formula from now on for calculation and user view

### Robustness.py
- Fixed placebo score to do multiple shuffles within a month instead of just one


### alpha layout changes
- And reinforce the edges connection alpha nodes that have mutation relationships or just simple ordinal orders, ~~I want it to look like a tree.~~ 
- Instead of a tree, we can model it as an expanding circle, with each run of plan_next.py as a ring, each ring surrounding the previous ring. so the inner most layer will be the first n alpha ideas generated from the first run of plan_next.py, and then i may choose to plan next again creating layer 2 surrounding layer 1, and i may also mutate any node in the first layer directly
  - mutation edge: parent_id → child_id
  - ~~ordinal edge: alpha_001 → alpha_002 → alpha_003~~ Dropping the ordinal edge idea, it doesn't contain information. Keep mutation edges only for now. 
  - Add a batch id instead to maintain the order of the experiments that are generated in the same batch.
  - emphasize mutation edges and make ordinal edges lighter/dashed
- I want to keep the alpha names to be alpha_xxx in a numerical order, so the name itself doesn't contain information at the layout level.
  - Maybe rename at the export to graph step, do not rename the names inside experiments.db
- I also want to add a filter on what to display on the graph, so it doesn't need to show all the nodes in experiments.db.
  - Include ancestors

- Keep an id for each alpha generated ~~OR take the index of the experiments.db as the id for each node saved in the experiments.db~~

Question: how does multiple testing bias apply? does it apply as long as I am using the same test to test? or I can reset my experiments number to 0 when I switch a direction or something?

---

#### Implementation Plan — Null Data & Winsorization Fixes

**Fix 1 — Fundamentals pre-load warm-up** (`core/data_loader.py`, `_fetch_fundamentals`)

Root cause: fundamentals are keyed by `Publish Date`. When `start_date = "2021-01-01"`, companies don't file Q4 reports until late Jan/Feb 2021. The current filter `DATE >= start_date` gives the ffill no seed value at `start_date`, so the first 1–2 monthly rebalancing periods run with almost no fundamental data and are silently skipped.

Fix: extend the SimFin filter and ffill grid 1 year back (`prefetch_start = start_date - 1 year`), then trim the output back to `start_date` before caching. The cached parquet stays the same shape (dates from `start_date` onward), but every stock now has a valid forward-filled value at `start_date` seeded from the prior year's filings.

```python
# In _fetch_fundamentals, replace the date-filter + ffill block:
prefetch_start = (pd.Timestamp(start_date) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
fund = fund[
    (fund["DATE"] >= pd.Timestamp(prefetch_start))
    & (fund["DATE"] <= pd.Timestamp(end_date))
]
trading_days = pd.bdate_range(start=prefetch_start, end=end_date)
# ... reindex + ffill on the extended grid ...
# After ffill, trim back:
fund = fund[fund["DATE"] >= pd.Timestamp(start_date)]
```

Cache action: delete `cache/fundamentals_*.parquet` and re-run with `--no-cache` to regenerate.

---

**Fix 2 — Centralize winsorization on computed features** (`core/features.py`)

Root cause: `_winsorise()` is scattered inconsistently across feature branches. `VOL_20D`, `LIQUIDITY`, and all raw fundamental pass-throughs (`EPS_LTM`, `SALES_LTM`, `NET_INCOME_LTM`, `SHARES_DILUTED`, `INV_CHANGE_LTM`) have no clipping, allowing extreme outliers to flow into formula evaluation.

Fix — two parts:

*Part A* — centralize in `compute_features` loop: apply `_winsorise` to every feature panel after computation, skipping only `ADJUSTED_PRICE` and `ADJUSTED_VOLUME` (raw price inputs used by momentum features — winsorizing prices would distort return calculations):
```python
for feat in feature_names:
    panel = _compute_single_feature(...)
    if panel is not None:
        if feat not in {"ADJUSTED_PRICE", "ADJUSTED_VOLUME"}:
            panel = _winsorise(panel)
        feature_panels[feat] = panel
```

*Part B* — strip per-feature `_winsorise()` calls inside `_compute_single_feature` (currently in `EBITDA_MARGIN`, `MOM12_1`, `MOM6_1`, `SALES_GROWTH`, `EPS_GROWTH`, `PRICE_TO_SALES`, `NET_MARGIN`, `INV_CHANGE_LTM`). Return the raw computed value; the centralized call in Part A handles it. Eliminates double-winsorization.

No changes to `backtest.py` — `_process_signal` already applies cross-sectional winsorization + z-score after formula evaluation; that layer is complementary.

---

**Verification**
1. After regenerating cache, check `fund[fund["FACTSET_ID"] == "AAPL"].head()` — `SALES_LTM` should be non-null starting from the first business day of `start_date`.
2. After winsorization fix, check `VOL_20D` and `EPS_LTM` panels — no values beyond the 1%/99% range.
3. End-to-end: `python scripts/run_experiment.py --config experiments/sample_alpha_001.json --no-cache` — IC series should start from the first rebalancing date with no silently-skipped periods.

---

## Alpha Graph Layout Redesign

### Concept
Research progresses in rounds. Each call to `plan_next.py` generates a batch of N new alpha ideas — these form **one ring**. The first batch is the innermost ring. Running `plan_next.py` again creates a surrounding ring. At any point you can mutate a node from any ring; the mutant joins the ring of whatever batch it was generated in. Mutation edges cross rings freely. The visual result is an expanding circle where radial distance = research age.

Ordinal edges are dropped — they carry no information. Sequential display names (`alpha_001`…) are assigned at export time only; experiments.db is never renamed.

---

### Files to Modify (in order)
1. `core/types.py` — add `batch_id` to `ExperimentRecord`
2. `core/memory.py` — add `batch_id` column; safe ALTER for existing DBs
3. `scripts/plan_next.py` — stamp one `batch_id` per run on all saved configs
4. `scripts/mutate_alpha.py` — accept `--batch-id` flag; stamp on saved config
5. `scripts/run_experiment.py` — read `batch_id` from config, pass to `ExperimentRecord`
6. `core/graph.py` — store `batch_id`/`timestamp`/`parent_id` on nodes; tag mutation edges
7. `scripts/export_graph.py` — filter flags, ancestor expansion, sequential rename, ring index
8. `core/visualization.py` — ring layout seeding, mutation-edge styling, detail panel update

---

### Change 1 — `core/types.py`

Add `batch_id: str | None` to `ExperimentRecord` TypedDict.

### Change 2 — `core/memory.py`

Add `batch_id TEXT` to `_CREATE_TABLE`. Safe migration in `init_db`:
```python
try:
    conn.execute("ALTER TABLE experiments ADD COLUMN batch_id TEXT")
except Exception:
    pass  # column already exists
```
Update `save_experiment` INSERT (add `batch_id` as 3rd column/value) and `_row_to_record`.

### Change 3 — `scripts/plan_next.py`

Generate one `batch_id` per invocation; stamp on every saved config JSON:
```python
batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
config["batch_id"] = batch_id   # inside save loop
```
Print the `batch_id` at the end so the user can reference it in `--filter-batch`.

### Change 4 — `scripts/mutate_alpha.py`

Add `--batch-id` flag; auto-generate if not provided:
```python
batch_id = args.batch_id or f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
child["batch_id"] = batch_id
```

### Change 5 — `scripts/run_experiment.py`

Read `batch_id` from config JSON and pass to `ExperimentRecord(batch_id=alpha.get("batch_id"), ...)`.

### Change 6 — `core/graph.py`

Store extra attrs on nodes, tag edges:
```python
self._graph.add_node(
    record["alpha_id"],
    batch_id=record.get("batch_id"),
    timestamp=record.get("timestamp", ""),
    parent_id=record.get("parent_id"),
    ...existing attrs unchanged...
)
if parent_id and parent_id in self._graph:
    self._graph.add_edge(parent_id, record["alpha_id"], type="mutation")
```

### Change 7 — `scripts/export_graph.py`

Add 5 CLI flags; implement filter → ancestor expansion → rename → ring-index in `main()`:

```python
parser.add_argument("--filter-verdict",    default=None)   # "promising,revise"
parser.add_argument("--filter-top",        type=int, default=None)
parser.add_argument("--filter-since",      default=None)   # "YYYY-MM-DD"
parser.add_argument("--filter-batch",      default=None)
parser.add_argument("--include-ancestors", action="store_true")

to_keep = set(g.nodes)
if args.filter_verdict:
    verdicts = set(args.filter_verdict.split(","))
    to_keep &= {n for n in to_keep if g.nodes[n].get("verdict") in verdicts}
if args.filter_since:
    to_keep &= {n for n in to_keep if g.nodes[n].get("timestamp", "") >= args.filter_since}
if args.filter_batch:
    to_keep &= {n for n in to_keep if g.nodes[n].get("batch_id") == args.filter_batch}
if args.filter_top:
    by_sharpe = sorted(to_keep, key=lambda n: g.nodes[n].get("Sharpe", 0.0), reverse=True)
    to_keep = set(by_sharpe[:args.filter_top])

if args.include_ancestors:
    for node in list(to_keep):
        parent = g.nodes[node].get("parent_id")
        while parent and parent in g.nodes:
            to_keep.add(parent)
            parent = g.nodes[parent].get("parent_id")

g.remove_nodes_from(set(g.nodes) - to_keep)

# Sequential rename by timestamp
for orig in g.nodes:
    g.nodes[orig]["original_id"] = orig
sorted_ids = sorted(g.nodes, key=lambda n: g.nodes[n].get("timestamp", ""))
id_map = {orig: f"alpha_{i+1:03d}" for i, orig in enumerate(sorted_ids)}
g = nx.relabel_nodes(g, id_map, copy=True)

# Ring index from batch_id
batches = sorted({g.nodes[n].get("batch_id") for n in g.nodes if g.nodes[n].get("batch_id")})
batch_order = {b: i for i, b in enumerate(batches)}
for node in g.nodes:
    g.nodes[node]["ring"] = batch_order.get(g.nodes[node].get("batch_id"), 0)
```

Need `import networkx as nx` at the top of this script.

### Change 8 — `core/visualization.py`

**Python**: add `original_id`, `batch_id`, `ring`, `timestamp` to `nodes_data`; add `edge_type` to `edges_data` (iterate `graph.edges(data=True)`).

**JS template** — seed ring positions before building `visNodes`:
```js
const ringCounts = {};
NODES_DATA.forEach(n => ringCounts[n.ring] = (ringCounts[n.ring] || 0) + 1);
const ringIdx = {};
NODES_DATA.forEach(n => {
  ringIdx[n.ring] = (ringIdx[n.ring] || 0);
  const angle = (2 * Math.PI * ringIdx[n.ring]++) / ringCounts[n.ring];
  const radius = 120 + n.ring * 220;
  n.x = radius * Math.cos(angle);
  n.y = radius * Math.sin(angle);
});
```
Pass `x`/`y` into `visNodes`. Edge styling: `color: "#e67e22"`, `width: 2.5`, `arrows: "to"`, `smooth: curvedCW`. Replace hierarchical layout + `physics: false` with:
```js
physics: {
  enabled: true,
  repulsion: { nodeDistance: 180, springLength: 200, springConstant: 0.04, damping: 0.15 },
  solver: 'repulsion',
  stabilization: { iterations: 200 },
},
layout: { improvedLayout: false },
```
Add "ID / Batch" row in `renderDetail(n)` showing `original_id` + `batch_id`.

### Verification
```bash
python -m pytest tests/test_v3_5.py -v

python scripts/plan_next.py --n 3 --save
# each experiments/*.json gets "batch_id": "batch_202606..."

python scripts/run_experiment.py --config experiments/plan_001_*.json

python scripts/export_graph.py
open reports/research_graph.html
# Inner ring = batch 1, outer rings = later batches
# Bold orange mutation edges, sequential alpha_001... labels
# Detail panel: original_id + batch_id row

python scripts/export_graph.py --filter-verdict promising
python scripts/export_graph.py --filter-top 5
python scripts/export_graph.py --filter-batch batch_20260627_...
python scripts/export_graph.py --filter-verdict promising --include-ancestors
```

### Running experiments to create a graph with multiple rings
- Issue: when the signal has duplicate quantile values (many stocks get the same rank), the 4 quantile cuts aren't all unique. duplicates="drop" silently removes duplicate bin edges, leaving fewer than 5 bins, but 5 labels are still passed, causing a mismatch.
- Fix: deduplicate the bins before calling pd.cut, then generate labels to match the actual bin count.
- Or track how often actual_bins < 5. If it is common, the alpha has weak cross-sectional resolution:
  - If this happens occasionally on a few dates, skip those dates or use fewer buckets.
  - If it happens often, the alpha is probably too discrete/sparse and quintile testing is not meaningful.
  - If many stocks have identical values because the alpha is binary / categorical / mostly zero, use group returns by raw signal value or long-short top vs bottom nonzero groups, not quintiles.
- Added one line in the prompt to llm that my processed dataset is already winsorized and standardized cross-sectionally
- TODO: add desription of data columns so LLM can understand better
- Fixed issue that reflection.py never sends llm the list of allowed operators such than llm makes up an operator that is not defined in this project.
- Issue2: ffill method kept holiday dates and filled na values in the price data, which causes the backtest to run on non-trading dates and produce incorrect results.
- Fix: joining data on prices and keeping only the dates in the price data, so that the backtest will only run on trading dates.