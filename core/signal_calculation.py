"""Signal calculation — evaluates raw_formula and returns a (DATE, TICKER) signal.

Namespace building and operator definitions live in formula_validator.py.
Data cleaning (winsorise / standardise) lives in data_process.py.
This module's sole job is to wire those two together and produce the signal
Series that backtest.py consumes.

Pipeline:
  processed_df (data_process.process output)
    → build_panel_namespace()      [formula_validator]
    → eval(raw_formula, namespace)
    → post winsorise + standardise [data_process]
    → (DATE, TICKER) signal Series → backtest.py
"""

from __future__ import annotations

import pandas as pd

from core.data_process import _winsorise, _standardise
from core.formula_validator import build_panel_namespace


def compute_signal(
    processed_df: pd.DataFrame,
    raw_formula: str,
    *,
    post_winsorise: bool = True,
    post_standardise: bool = True,
) -> pd.Series:
    """Evaluate raw_formula on the processed panel and return a signal Series.

    Args:
        processed_df:     (DATE, TICKER) MultiIndex DataFrame produced by
                          data_process.process(). Every value column is already
                          cross-sectionally winsorised and standardised.
        raw_formula:      Formula string where each raw column name resolves to
                          a full DATE × TICKER DataFrame. Supports:
                            - time-series pandas methods: .shift(n), .rolling(n).std()
                            - cross-sectional ops: rank(), zscore(), log(), abs(), sign()
                            - standard arithmetic: +  -  *  /  **
                          Example:
                            rank(ADJUSTED_PRICE.shift(21) / ADJUSTED_PRICE.shift(252) - 1)
                            + 0.5 * rank(OPER_INCOME_LTM / SALES_LTM)
        post_winsorise:   Clip signal at 1%/99% cross-sectionally after computation.
        post_standardise: Z-score signal cross-sectionally after computation.

    Returns:
        pd.Series with (DATE, TICKER) MultiIndex, named "signal", NaNs dropped.

    Raises:
        ValueError: unknown column, evaluation error, or wrong result type.
    """
    if not isinstance(processed_df.index, pd.MultiIndex):
        raise ValueError("processed_df must have a (DATE, TICKER) MultiIndex.")

    namespace = build_panel_namespace(processed_df)

    try:
        result = eval(raw_formula, {"__builtins__": {}}, namespace)  # noqa: S307
    except NameError as exc:
        raise ValueError(
            f"raw_formula references unknown column: {exc}. "
            "Check that the column exists in the processed DataFrame."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Signal formula evaluation failed: {exc}") from exc

    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"raw_formula must evaluate to a DATE × TICKER DataFrame, "
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
