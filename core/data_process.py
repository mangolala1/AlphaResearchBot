from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

_WINSOR_LOW = 0.01
_WINSOR_HIGH = 0.99
_DATA_DIR = Path("data")


def processed_path(statement: str, start_date: str, end_date: str) -> Path:
    """Return the data/ cache path for a processed statement parquet."""
    key = f"processed_{statement}|{start_date}|{end_date}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return _DATA_DIR / f"processed_{statement}_{digest}.parquet"

# Metadata columns — carried through unchanged, never winsorised/standardised.
_NON_VALUE_COLS = {"SECTOR", "COUNTRY"}

# Price columns — kept raw (not winsorised/standardised); joined back in backtest.py.
_PRICE_COLS = {"ADJUSTED_PRICE", "ADJUSTED_VOLUME"}

# SimFin column names → standardised identifiers.
# automatically (spaces/special chars → underscores, uppercased).
_COLUMN_RENAMES: dict[str, str] = {
    # Income Statement
    "Revenue":                              "REVENUE_LTM",
    "Cost of Revenue":                      "COGS_LTM",
    "Gross Profit":                         "GROSS_PROFIT_LTM",
    "Operating Expenses":                   "OPERATING_EXPENSES_LTM",
    "Selling, General & Administrative":    "SGA_EXPENSE_LTM",
    "Operating Income (Loss)":              "OPERATING_INCOME_LTM",
    "Non-Operating Income (Loss)":          "NON_OPERATING_INCOME_LTM",
    "Interest Expense, Net":                "NET_INTEREST_EXPENSE_LTM",
    "Pretax Income (Loss), Adj.":           "PRETAX_INCOME_ADJ_LTM",
    "Pretax Income (Loss)":                 "PRETAX_INCOME_LTM",
    "Income Tax (Expense) Benefit, Net":    "INCOME_TAX_LTM",
    "Income (Loss) from Continuing Operations": "CONTINUING_INCOME_LTM",
    "Net Income":                           "NET_INCOME_LTM",
    "Net Income (Common)":                  "NET_INCOME_COMMON_LTM",
    "EPS (Diluted)":                        "EPS_DILUTED",

    # Shares
    "Shares (Basic)":                       "SHARES_BASIC",
    "Shares (Diluted)":                     "SHARES_DILUTED",

    # Cash Flow
    "Net Income/Starting Line":             "NET_INCOME_START_LTM",
    "Depreciation & Amortization":          "DA_LTM",
    "Non-Cash Items":                       "NON_CASH_ITEMS_LTM",
    "Change in Working Capital":            "WORKING_CAPITAL_CHANGE_LTM",
    "Net Cash from Operating Activities":   "CFO_LTM",
    "Change in Fixed Assets & Intangibles": "FIXED_ASSET_CHANGE_LTM",
    "Net Cash from Investing Activities":   "CFI_LTM",
    "Capital Expenditures":                 "CAPEX_LTM",
    "Cash from (Repayment of) Debt":        "DEBT_FINANCING_CF_LTM",
    "Cash from (Repurchase of) Equity":     "EQUITY_FINANCING_CF_LTM",
    "Net Cash from Financing Activities":   "CFF_LTM",
    "Net Change in Cash":                   "NET_CHANGE_CASH_LTM",
}

def process(
    df: pd.DataFrame,
    value_cols: list[str] | None = None,
    winsorise: bool = True,
    standardise: bool = True,
    ffill_daily: bool = True,
) -> pd.DataFrame:
    """Rename, clean, winsorise, standardise, and forward-fill a data panel.

    Steps:
      1. Apply known SimFin → standardised column renames
      2. Sanitise any remaining non-standard column names (spaces/special chars → underscores, uppercased)
      3. Normalise to (DATE, TICKER) MultiIndex
      4. Drop rows where all value columns are NaN
      5. Cross-sectional winsorisation: clip at 1%/99% across stocks per date
      6. Cross-sectional standardisation: z-score across stocks per date
      7. Drop rows with any remaining NaN in value columns
      8. Forward-fill to day frequency within each ticker

    Args:
        df:          Long-format DataFrame with TICKER and DATE columns (or
                     a (DATE, TICKER) MultiIndex DataFrame).
        value_cols:  Columns to winsorise/standardise after renaming. Defaults to
                     all non-metadata, non-price columns present after renaming.
        winsorise:   Apply cross-sectional winsorisation.
        standardise: Apply cross-sectional z-score standardisation.
        ffill_daily: Forward-fill processed values to business-day frequency.

    Returns:
        (DATE, TICKER) MultiIndex DataFrame.
    """
    # Step 1 — apply known renames
    df = df.rename(columns=_COLUMN_RENAMES)

    # Step 2 — sanitise any remaining non-standard column names
    df = df.rename(columns={
        col: _sanitize_col(col)
        for col in df.columns
        if col not in _NON_VALUE_COLS
        and col not in _PRICE_COLS
        and col != _sanitize_col(col)
    })

    # Step 3 — normalise index
    df = _to_multiindex(df)

    # Determine which columns to process
    if value_cols is None:
        value_cols = [
            c for c in df.columns
            if c not in _NON_VALUE_COLS and c not in _PRICE_COLS
        ]

    processed: dict[str, pd.DataFrame] = {}

    for col in value_cols:
        if col not in df.columns:
            continue

        # Wide pivot: rows=DATE, cols=TICKER — enables fast vectorised row-ops
        wide = df[col].unstack(level="TICKER")

        # Step 4 — drop dates where every stock is NaN
        wide = wide.dropna(how="all")

        if winsorise:
            wide = _winsorise(wide)   # Step 5

        if standardise:
            wide = _standardise(wide) # Step 6

        processed[col] = wide

    if not processed:
        return df.iloc[0:0]

    # Stack each processed wide panel back to long form and join
    stacked = pd.concat(
        [wide.stack(future_stack=True).rename(col) for col, wide in processed.items()],
        axis=1,
    )
    stacked.index.names = ["DATE", "TICKER"]

    # Carry through non-value columns (e.g. SECTOR) aligned to the new index
    for col in _NON_VALUE_COLS:
        if col in df.columns:
            stacked[col] = df[col].reindex(stacked.index)

    # Step 7 — drop rows with any NaN remaining in the processed columns
    stacked = stacked.dropna(subset=list(processed.keys()))

    # Step 8 — forward-fill to daily frequency
    if ffill_daily:
        stacked = _ffill_to_daily(stacked)

    return stacked


def _sanitize_col(name: str) -> str:
    """Convert any column name to a valid uppercase Python identifier."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return s.strip("_").upper()


def _winsorise(wide: pd.DataFrame) -> pd.DataFrame:
    """Clip at 1%/99% across tickers per date (row-wise quantiles)."""
    lo = wide.quantile(_WINSOR_LOW, axis=1)
    hi = wide.quantile(_WINSOR_HIGH, axis=1)
    return wide.clip(lower=lo, upper=hi, axis=0)


def _standardise(wide: pd.DataFrame) -> pd.DataFrame:
    """Z-score across tickers per date (row-wise mean/std)."""
    mu = wide.mean(axis=1)
    sigma = wide.std(axis=1, ddof=1).replace(0, float("nan")).fillna(1.0) + 1e-9
    return wide.sub(mu, axis=0).div(sigma, axis=0)


def _ffill_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Expand a quarterly-frequency panel to day frequency via ffill.
    Rows before a ticker's first filing date (still NaN after ffill) are dropped.
    """
    dates = df.index.get_level_values("DATE")
    tickers = df.index.get_level_values("TICKER").unique()
    full_dates = pd.bdate_range(start=dates.min(), end=dates.max())
    full_idx = pd.MultiIndex.from_product([full_dates, tickers], names=["DATE", "TICKER"])
    return (
        df.reindex(full_idx)
        .groupby(level="TICKER")
        .ffill()
        .dropna(how="all")
    )


def _to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise input to a (DATE, TICKER) MultiIndex DataFrame."""
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if names == ["DATE", "TICKER"]:
            return df
        if set(names) == {"DATE", "TICKER"}:
            return df.reorder_levels(["DATE", "TICKER"])
        df = df.reset_index()

    df = df.copy()
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index(["DATE", "TICKER"])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from core.data_loader import DataLoader

    START, END = "2021-01-01", "2026-06-01"

    loader = DataLoader()
    _, income_raw, cashflow_raw, _ = loader.load(START, END)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    for label, raw in [("income", income_raw), ("cashflow", cashflow_raw)]:
        print(f"\n{'='*60}")
        print(f"  {label.upper()} — raw cache")
        print(f"  shape : {raw.shape}")
        print(f"  cols  : {list(raw.columns)}")

        processed = process(raw)

        out_path = processed_path(label, START, END)
        processed.to_parquet(out_path)
        print(f"\n  {label.upper()} — after process() → saved to {out_path}")
        print(f"  shape : {processed.shape}")
        print(f"  cols  : {list(processed.columns)}")
        print(f"  dates : {processed.index.get_level_values('DATE').min().date()} "
              f"→ {processed.index.get_level_values('DATE').max().date()}")
        print(f"  tickers: {processed.index.get_level_values('TICKER').nunique()}")
        print(f"\n  NaN counts per column:")
        nan_counts = processed.isna().sum()
        print(nan_counts[nan_counts > 0].to_string() if nan_counts.any() else "    none")
        print(f"\n  Sample (first 5 rows):")
        print(processed.head().to_string())
