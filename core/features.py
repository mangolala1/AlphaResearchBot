"""Feature engineering: derive alpha features from raw price and fundamental DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

_WINSOR_LOW = 0.01
_WINSOR_HIGH = 0.99

_RAW_PRICE_COLS = {"ADJUSTED_PRICE", "ADJUSTED_VOLUME"}
_RAW_FUNDAMENTAL_COLS = {
    "SALES_LTM", "COGS_LTM", "NET_INCOME_LTM", "SHARES_DILUTED",
    "INV_CHANGE_LTM", "OPER_INCOME_LTM", "DA_LTM",
}
_RAW_UNIVERSE_COLS = {"SECTOR", "INDUSTRY", "TICKER"}


def compute_features(
    prices_df: pd.DataFrame,
    fundamentals_ttm_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute requested features and return a MultiIndex (DATE, TICKER) DataFrame.

    Only the features listed in feature_names are computed.
    Stocks with NaN values for a feature on a given date are included with NaN
    (the backtest engine will drop them from that period's cross-section).
    """
    prices_df = _normalise_dates(prices_df)
    fundamentals_df = _normalise_dates(fundamentals_ttm_df)

    # Build a price pivot: rows=DATE, cols=TICKER
    price_pivot = prices_df.pivot_table(
        index="DATE", columns="TICKER", values="ADJUSTED_PRICE"
    )
    volume_pivot = prices_df.pivot_table(
        index="DATE", columns="TICKER", values="ADJUSTED_VOLUME"
    )

    # Build fundamentals pivots for each column
    fund_pivots: dict[str, pd.DataFrame] = {}
    for col in _RAW_FUNDAMENTAL_COLS:
        if col in fundamentals_df.columns:
            fund_pivots[col] = fundamentals_df.pivot_table(
                index="DATE", columns="TICKER", values=col
            )

    # Universe metadata (static, no date dimension)
    universe_meta = universe_df.set_index("TICKER") if "TICKER" in universe_df.columns else universe_df

    feature_panels: dict[str, pd.DataFrame] = {}

    for feat in feature_names:
        panel = _compute_single_feature(
            feat, price_pivot, volume_pivot, fund_pivots, universe_meta
        )
        if panel is not None:
            if feat not in {"ADJUSTED_PRICE", "ADJUSTED_VOLUME"}:
                panel = _winsorise(panel)
            feature_panels[feat] = panel

    if not feature_panels:
        raise ValueError(f"No features could be computed from: {feature_names}")

    # Stack each panel to long form and join
    long_frames = []
    for feat_name, panel in feature_panels.items():
        long = panel.stack(future_stack=True).rename(feat_name)
        long.index.names = ["DATE", "TICKER"]
        long_frames.append(long)

    result = pd.concat(long_frames, axis=1)

    # Attach SECTOR if universe has it (needed for robustness sector_ic)
    if "SECTOR" in universe_meta.columns:
        sector_map = universe_meta["SECTOR"]
        result["SECTOR"] = result.index.get_level_values("TICKER").map(sector_map)

    return result


def _compute_single_feature(
    name: str,
    price_pivot: pd.DataFrame,
    volume_pivot: pd.DataFrame,
    fund_pivots: dict[str, pd.DataFrame],
    universe_meta: pd.DataFrame,
) -> pd.DataFrame | None:
    """Return a wide DataFrame (DATE × TICKER) for a single feature, or None if unsupported."""

    # Raw pass-through
    if name == "ADJUSTED_PRICE":
        return price_pivot
    if name == "ADJUSTED_VOLUME":
        return volume_pivot
    if name in _RAW_FUNDAMENTAL_COLS:
        return fund_pivots.get(name)

    # Derived features — computed from raw columns
    if name == "EPS_LTM":
        net_income = fund_pivots.get("NET_INCOME_LTM")
        shares = fund_pivots.get("SHARES_DILUTED")
        if net_income is None or shares is None:
            return None
        return net_income / shares.replace(0, np.nan)

    if name == "EBITDA_LTM":
        oper_income = fund_pivots.get("OPER_INCOME_LTM")
        da = fund_pivots.get("DA_LTM")
        if oper_income is None:
            return None
        return oper_income + (da.fillna(0) if da is not None else 0)

    if name == "EBITDA_MARGIN":
        oper_income = fund_pivots.get("OPER_INCOME_LTM")
        da = fund_pivots.get("DA_LTM")
        sales = fund_pivots.get("SALES_LTM")
        if oper_income is None or sales is None:
            return None
        ebitda = oper_income + (da.fillna(0) if da is not None else 0)
        return ebitda / sales.replace(0, np.nan)

    if name == "MOM12_1":
        # 12-month return excluding last month: price[t-21] / price[t-252] - 1
        return price_pivot.shift(21) / price_pivot.shift(252) - 1

    if name == "MOM6_1":
        return price_pivot.shift(21) / price_pivot.shift(126) - 1

    if name == "SALES_GROWTH":
        sales = fund_pivots.get("SALES_LTM")
        if sales is None:
            return None
        return sales / sales.shift(252).replace(0, np.nan) - 1

    if name == "EPS_GROWTH":
        net_income = fund_pivots.get("NET_INCOME_LTM")
        shares = fund_pivots.get("SHARES_DILUTED")
        if net_income is None or shares is None:
            return None
        eps = net_income / shares.replace(0, np.nan)
        return eps / eps.shift(252).replace(0, np.nan) - 1

    if name == "PRICE_TO_SALES":
        sales = fund_pivots.get("SALES_LTM")
        shares = fund_pivots.get("SHARES_DILUTED")
        if sales is None or shares is None:
            return None
        sales_per_share = sales / shares.replace(0, np.nan)
        return price_pivot / sales_per_share.replace(0, np.nan)

    if name == "VOL_20D":
        log_returns = np.log(price_pivot / price_pivot.shift(1))
        return log_returns.rolling(20).std()

    if name == "LIQUIDITY":
        dollar_vol = price_pivot * volume_pivot
        return dollar_vol.rolling(20).mean()

    if name == "NET_MARGIN":
        net_income = fund_pivots.get("NET_INCOME_LTM")
        sales = fund_pivots.get("SALES_LTM")
        if net_income is None or sales is None:
            return None
        return net_income / sales.replace(0, np.nan)

    if name == "INV_CHANGE_LTM":
        inv_change = fund_pivots.get("INV_CHANGE_LTM")
        if inv_change is None:
            return None
        # Positive = inventory drawn down (demand > supply, bullish signal)
        # Negative = inventory build-up (excess supply, bearish signal)
        return inv_change

    return None


def _winsorise(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise each column at 1%/99% percentile."""
    return df.clip(
        lower=df.quantile(_WINSOR_LOW),
        upper=df.quantile(_WINSOR_HIGH),
        axis=1,
    )

def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ross-sectional standardization on pivot table where each column is a stock."""
    return (df - df.mean()) / (df.std(ddof=1) + 1e-9)


def _normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DATE column is datetime."""
    if "DATE" in df.columns:
        df = df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df
