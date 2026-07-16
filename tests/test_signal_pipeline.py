"""Pipeline-order tests: signals are computed from RAW values, then the
resulting signal — not the inputs — is winsorised + standardised."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_process import process
from core.signal_calculation import compute_signal


def _raw_panel(n_dates=6, tickers=("AAA", "BBB", "CCC", "DDD", "EEE")):
    """Tiny long-format panel with raw dollar-scale fundamentals."""
    dates = pd.bdate_range("2026-01-05", periods=n_dates)
    n = len(tickers)
    rows = []
    for d in dates:
        for j, t in enumerate(tickers):
            rows.append({
                "DATE": d,
                "TICKER": t,
                # scales differ by orders of magnitude across stocks;
                # CFO rises and REVENUE falls with j, both always positive
                "CFO_LTM": (j + 1) * 1e8,
                "REVENUE_LTM": (n - j) * 1e9 + 1e9,
            })
    return pd.DataFrame(rows)


def test_process_keeps_raw_values_by_default():
    df = _raw_panel()
    out = process(df, ffill_daily=False)
    # Raw dollar magnitudes preserved — no z-scoring happened
    assert out["CFO_LTM"].max() == pytest.approx(5e8)
    assert out["REVENUE_LTM"].min() == pytest.approx(2e9)


def test_process_optin_standardise_still_works():
    df = _raw_panel()
    out = process(df, ffill_daily=False, winsorise=True, standardise=True)
    per_date_mean = out["CFO_LTM"].groupby(level="DATE").mean()
    assert np.allclose(per_date_mean, 0.0, atol=1e-9)


def test_signal_ratio_uses_raw_values_and_is_standardised_after():
    processed = process(_raw_panel(), ffill_daily=False)
    signal = compute_signal(processed, "CFO_LTM / REVENUE_LTM")

    # Post-eval standardisation: per-date mean ~ 0, std ~ 1
    per_date = signal.groupby(level="DATE")
    assert np.allclose(per_date.mean(), 0.0, atol=1e-9)
    assert np.allclose(per_date.std(ddof=1), 1.0, atol=1e-6)

    # Raw-ratio ordering preserved: CFO rises and REVENUE falls with ticker index,
    # so the ratio (and hence the z-scored signal) must be strictly increasing.
    one_date = signal.xs(signal.index.get_level_values("DATE")[0], level="DATE")
    ordered = one_date.loc[["AAA", "BBB", "CCC", "DDD", "EEE"]].to_numpy()
    assert np.all(np.diff(ordered) > 0)


def test_signal_winsorises_outliers_before_standardising():
    df = _raw_panel(tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF",
                             "GGG", "HHH", "III", "JJJ"))
    # Give one stock an absurd ratio; winsorisation should cap its dominance
    df.loc[df["TICKER"] == "JJJ", "CFO_LTM"] = 1e13
    processed = process(df, ffill_daily=False)
    signal = compute_signal(processed, "CFO_LTM / REVENUE_LTM")
    winsorised_max = signal.groupby(level="DATE").max()

    raw = compute_signal(processed, "CFO_LTM / REVENUE_LTM",
                         post_winsorise=False, post_standardise=True)
    raw_max = raw.groupby(level="DATE").max()

    # Without winsorisation the outlier dominates the z-score far more
    assert (winsorised_max < raw_max).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
