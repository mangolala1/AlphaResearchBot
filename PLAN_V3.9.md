# AlphaResearchBot — Version 3.9 Plan

### data_loader.py
- Issue: 
  - lookahead bias for fundamentals data that are usually reported with a lag, need to know the release date as well.
  - Fix: SimFin provides the `REPORT_DATE` column for each fundamental data point, which can be used to filter out data that would not have been known at the time of the backtest.
  - ~~- Simple fix: shift fundamentals data by 1 quarter to avoid lookahead bias.~~
- Issue2: Data loading of fundamentals data
  - The current implementation of `_fetch_fundamentals` in `core/data_loader.py` filters the fundamentals data based on the `start_date`, which can lead to missing data for the first few months of the backtest. This is because companies may not have filed their Q4 reports until late January or February, resulting in a lack of seed values for forward-filling.
  - Fix: Extend the filter to include data from one year prior to the `start_date`, allowing for proper forward-filling and ensuring that all stocks have valid fundamental data at the start of the backtest.

- Future steps: maybe create more features to take advantage of both quarterly data and TTM data.

### data_process.py
- Separately loading cashflow and income statement
- Kept more data columns and add those to available raw data in formula_validator.py
- Question: Not sure if I should standardize in data_process.py?

- Future steps: might need better logic or flexibility for na values handling, maybe keep them at this stage and later on decide how to handle them in signal calculation, e.g., drop or replace with 0 or replace with mean or median by sector, etc.


### formula_validator.py
- Added more data columns 
- Added more operators
- Changed the operators' definition to also show the parameters each can take

### ~~Features.py~~ signal calculation.py
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

### Robustness.py
- Issue: The current implementation of robustness checks may not be correctly applied to all features, leading to potential inconsistencies in the evaluation of alpha signals.


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



