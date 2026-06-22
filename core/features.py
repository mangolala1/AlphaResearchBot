"""Feature engineering: derive alpha features from raw price and fundamental DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

_WINSOR_LOW = 0.01
_WINSOR_HIGH = 0.99

_RAW_PRICE_COLS = {"ADJUSTED_PRICE", "ADJUSTED_VOLUME"}
_RAW_FUNDAMENTAL_COLS = {
    "EPS_LTM", "SALES_LTM", "EBITDA_LTM", "COGS_LTM",
}
_RAW_UNIVERSE_COLS = {"SECTOR", "INDUSTRY", "FACTSET_ID"}


def compute_features(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute requested features and return a MultiIndex (DATE, FACTSET_ID) DataFrame.

    Only the features listed in feature_names are computed.
    Stocks with NaN values for a feature on a given date are included with NaN
    (the backtest engine will drop them from that period's cross-section).
    """
    prices_df = _normalise_dates(prices_df)
    fundamentals_df = _normalise_dates(fundamentals_df)

    # Build a price pivot: rows=DATE, cols=FACTSET_ID
    price_pivot = prices_df.pivot_table(
        index="DATE", columns="FACTSET_ID", values="ADJUSTED_PRICE"
    )
    volume_pivot = prices_df.pivot_table(
        index="DATE", columns="FACTSET_ID", values="ADJUSTED_VOLUME"
    )

    # Build fundamentals pivots for each column
    fund_pivots: dict[str, pd.DataFrame] = {}
    for col in _RAW_FUNDAMENTAL_COLS:
        if col in fundamentals_df.columns:
            fund_pivots[col] = fundamentals_df.pivot_table(
                index="DATE", columns="FACTSET_ID", values=col
            )

    # Universe metadata (static, no date dimension)
    universe_meta = universe_df.set_index("FACTSET_ID") if "FACTSET_ID" in universe_df.columns else universe_df

    feature_panels: dict[str, pd.DataFrame] = {}

    for feat in feature_names:
        panel = _compute_single_feature(
            feat, price_pivot, volume_pivot, fund_pivots, universe_meta
        )
        if panel is not None:
            feature_panels[feat] = panel

    if not feature_panels:
        raise ValueError(f"No features could be computed from: {feature_names}")

    # Stack each panel to long form and join
    long_frames = []
    for feat_name, panel in feature_panels.items():
        long = panel.stack(future_stack=True).rename(feat_name)
        long.index.names = ["DATE", "FACTSET_ID"]
        long_frames.append(long)

    result = pd.concat(long_frames, axis=1)

    # Attach SECTOR if universe has it (needed for robustness sector_ic)
    if "SECTOR" in universe_meta.columns:
        sector_map = universe_meta["SECTOR"]
        result["SECTOR"] = result.index.get_level_values("FACTSET_ID").map(sector_map)

    return result


def _compute_single_feature(
    name: str,
    price_pivot: pd.DataFrame,
    volume_pivot: pd.DataFrame,
    fund_pivots: dict[str, pd.DataFrame],
    universe_meta: pd.DataFrame,
) -> pd.DataFrame | None:
    """Return a wide DataFrame (DATE × FACTSET_ID) for a single feature, or None if unsupported."""

    # Raw pass-through
    if name == "ADJUSTED_PRICE":
        return price_pivot
    if name == "ADJUSTED_VOLUME":
        return volume_pivot
    if name in _RAW_FUNDAMENTAL_COLS:
        return fund_pivots.get(name)

    # Derived features
    if name == "EBITDA_MARGIN":
        ebitda = fund_pivots.get("EBITDA_LTM")
        sales = fund_pivots.get("SALES_LTM")
        if ebitda is None or sales is None:
            return None
        raw = ebitda / sales.replace(0, np.nan)
        return _winsorise(raw)

    if name == "MOM12_1":
        # 12-month return excluding last month: price[t-21] / price[t-252] - 1
        return _winsorise(price_pivot.shift(21) / price_pivot.shift(252) - 1)

    if name == "MOM6_1":
        return _winsorise(price_pivot.shift(21) / price_pivot.shift(126) - 1)

    if name == "SALES_GROWTH":
        sales = fund_pivots.get("SALES_LTM")
        if sales is None:
            return None
        return _winsorise(sales / sales.shift(252).replace(0, np.nan) - 1)

    if name == "EPS_GROWTH":
        eps = fund_pivots.get("EPS_LTM")
        if eps is None:
            return None
        return _winsorise(eps / eps.shift(252).replace(0, np.nan) - 1)

    if name == "PRICE_TO_SALES":
        sales = fund_pivots.get("SALES_LTM")
        if sales is None:
            return None
        # Simple proxy: price / (sales_LTM scaled to per-share equivalent is unknown,
        # so we use price / sales_LTM_millions as a relative cross-sectional signal)
        return _winsorise(price_pivot / (sales / 1e6).replace(0, np.nan))

    if name == "VOL_20D":
        log_returns = np.log(price_pivot / price_pivot.shift(1))
        return log_returns.rolling(20).std()

    if name == "LIQUIDITY":
        dollar_vol = price_pivot * volume_pivot
        return dollar_vol.rolling(20).mean()

    return None


def _winsorise(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise each column at 1%/99% percentile."""
    return df.clip(
        lower=df.quantile(_WINSOR_LOW),
        upper=df.quantile(_WINSOR_HIGH),
        axis=1,
    )


def _normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DATE column is datetime."""
    if "DATE" in df.columns:
        df = df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df
