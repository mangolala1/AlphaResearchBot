from __future__ import annotations

import pandas as pd
import numpy as np

_WINSOR_LOW = 0.01
_WINSOR_HIGH = 0.99

_NON_VALUE_COLS = {"SECTOR", "COUNTRY"}


def process(
    df: pd.DataFrame,
    value_cols: list[str] | None = None,
    winsorise: bool = True,
    standardise: bool = True,
) -> pd.DataFrame:
    """Clean, winsorise, and standardise a data panel.

    Steps:
      1. Normalise to (DATE, TICKER) MultiIndex
      2. Drop rows where all value columns are NaN
      3. Cross-sectional winsorisation: clip at 1%/99% across stocks per date
      4. Cross-sectional standardisation: z-score across stocks per date
      5. Drop rows with any remaining NaN in value columns

    Args:
        df:          Long-format DataFrame with TICKER and DATE columns (or
                     a (DATE, TICKER) MultiIndex DataFrame).
        value_cols:  Columns to winsorise/standardise. Defaults to all
                     non-metadata columns.
        winsorise:   Apply cross-sectional winsorisation.
        standardise: Apply cross-sectional z-score standardisation.

    Returns:
        (DATE, TICKER) MultiIndex DataFrame.
    """
    df = _to_multiindex(df)

    if value_cols is None:
        value_cols = [c for c in df.columns if c not in _NON_VALUE_COLS]

    processed: dict[str, pd.DataFrame] = {}

    for col in value_cols:
        if col not in df.columns:
            continue

        # Wide pivot: rows=DATE, cols=TICKER — enables fast vectorised row-ops
        wide = df[col].unstack(level="TICKER")

        # Drop dates where every stock is NaN
        wide = wide.dropna(how="all")

        if winsorise:
            wide = _winsorise(wide)

        if standardise:
            wide = _standardise(wide)

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

    # Drop rows with any NaN remaining in the processed columns
    stacked = stacked.dropna(subset=list(processed.keys()))

    return stacked


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
