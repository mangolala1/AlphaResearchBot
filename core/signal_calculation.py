"""Signal calculation — evaluates formula and returns a (DATE, TICKER) signal.

Namespace building and operator definitions live in formula_validator.py.
Data cleaning (winsorise / standardise) lives in data_process.py.
This module's sole job is to wire those two together and produce the signal
Series that backtest.py consumes.

Pipeline:
  processed_df (data_process.process output)
    → build_panel_namespace()      [formula_validator]
    → eval(formula, namespace)
    → post winsorise + standardise [data_process]
    → (DATE, TICKER) signal Series → backtest.py
"""

from __future__ import annotations

import pandas as pd

from core.data_process import _winsorise, _standardise
from core.formula_validator import build_panel_namespace


def compute_signal(
    processed_df: pd.DataFrame,
    formula: str,
    *,
    post_winsorise: bool = True,
    post_standardise: bool = True,
) -> pd.Series:
    """Evaluate formula on the processed panel and return a signal Series."""
    if not isinstance(processed_df.index, pd.MultiIndex):
        raise ValueError("processed_df must have a (DATE, TICKER) MultiIndex.")

    namespace = build_panel_namespace(processed_df)

    try:
        result = eval(formula, {"__builtins__": {}}, namespace)  # noqa: S307
    except NameError as exc:
        raise ValueError(
            f"formula references unknown column: {exc}. "
            "Check that the column exists in the processed DataFrame."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Signal formula evaluation failed: {exc}") from exc

    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"formula must evaluate to a DATE × TICKER DataFrame, "
            f"got {type(result).__name__}. Ensure all operands are full panel "
            "DataFrames (raw column name references), not scalars or Series."
        )

    if post_winsorise:
        result = _winsorise(result)
    if post_standardise:
        result = _standardise(result)

    signal = result.stack(future_stack=True)
    signal.index.names = ["DATE", "TICKER"]
    return signal.rename("signal").dropna()
