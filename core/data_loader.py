"""Data loader — fetches US equity data and caches locally as parquet.

Data sources (per PLAN_V2.md):
  - yfinance   : daily OHLCV prices for S&P 500 stocks
  - SimFin     : quarterly fundamentals (Revenue, EBITDA, EPS) → TTM
  - FRED       : risk-free rate and macro series (robustness support)

Cache: parquet files in cache/ directory. Delete cache/ to force a fresh fetch.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import yfinance as yf


class DataLoader:
    """Fetches US equity price, fundamentals, and universe data.

    Returns DataFrames with the same column contract as the previous
    Snowflake-backed version so that features.py and backtest.py are unchanged:

      prices_df      : FACTSET_ID, DATE, ADJUSTED_PRICE, ADJUSTED_VOLUME
      fundamentals_df: FACTSET_ID, DATE, SALES_LTM, EBITDA_LTM, EPS_LTM, COGS_LTM
      universe_df    : FACTSET_ID, SECTOR, COUNTRY

    Results are cached as parquet files so remote sources are only queried once
    per (start_date, end_date) pair. Delete the cache/ directory to force a
    fresh fetch.
    """

    def __init__(self, cache_dir: str = "cache", no_cache: bool = False) -> None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache_dir = Path(cache_dir)
        self._no_cache = no_cache

    def load(
        self,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (prices_df, fundamentals_df, universe_df).

        Fetches from yfinance + SimFin on first call for a given date range;
        reads parquet cache on subsequent calls unless no_cache=True.
        """
        prices_path = self._cache_path("prices", start_date, end_date)
        fund_path = self._cache_path("fundamentals", start_date, end_date)
        univ_path = self._cache_path("universe", start_date, end_date)

        need_prices = self._no_cache or not prices_path.exists()
        need_fund = self._no_cache or not fund_path.exists()
        need_univ = self._no_cache or not univ_path.exists()

        if not need_prices and not need_fund and not need_univ:
            return (
                pd.read_parquet(prices_path),
                pd.read_parquet(fund_path),
                pd.read_parquet(univ_path),
            )

        print("[DataLoader] Fetching universe from Wikipedia...")
        universe_df = self._fetch_universe()
        universe_df.to_parquet(univ_path, index=False)

        tickers = universe_df["FACTSET_ID"].tolist()

        print(f"[DataLoader] Fetching prices from yfinance for {len(tickers)} tickers...")
        prices_df = self._fetch_prices(tickers, start_date, end_date)
        prices_df.to_parquet(prices_path, index=False)

        print("[DataLoader] Fetching fundamentals from SimFin...")
        fundamentals_df = self._fetch_fundamentals(tickers, start_date, end_date)
        fundamentals_df.to_parquet(fund_path, index=False)

        return prices_df, fundamentals_df, universe_df

    # ------------------------------------------------------------------
    # Private fetch methods
    # ------------------------------------------------------------------

    def _fetch_universe(self) -> pd.DataFrame:
        import io
        import ssl
        import urllib.request

        import certifi

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as resp:
            html = resp.read()
        df = pd.read_html(io.BytesIO(html))[0][["Symbol", "GICS Sector"]]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df = df.rename(columns={"Symbol": "FACTSET_ID", "GICS Sector": "SECTOR"})
        df["COUNTRY"] = "US"
        return df.reset_index(drop=True)

    def _fetch_prices(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        # raw.columns is MultiIndex; level order varies by yfinance version:
        #   newer: (ticker, field) — default with group_by="ticker"
        #   older: (field, ticker)
        # Normalize to (field, ticker) by checking what's in level 0.
        if isinstance(raw.columns, pd.MultiIndex):
            sample = raw.columns[0]
            if sample[0] in ("Open", "High", "Low", "Close", "Volume", "Adj Close"):
                # (field, ticker) — older style
                close_wide = raw["Close"]
                volume_wide = raw["Volume"]
            else:
                # (ticker, field) — newer style; swap levels
                raw = raw.swaplevel(axis=1)
                close_wide = raw["Close"]
                volume_wide = raw["Volume"]

            close = close_wide.stack(future_stack=True).reset_index()
            volume = volume_wide.stack(future_stack=True).reset_index()
            close.columns = ["DATE", "FACTSET_ID", "ADJUSTED_PRICE"]
            volume.columns = ["DATE", "FACTSET_ID", "ADJUSTED_VOLUME"]
        else:
            # Single ticker
            ticker = tickers[0]
            close = raw[["Close"]].reset_index().rename(
                columns={"index": "DATE", "Date": "DATE", "Close": "ADJUSTED_PRICE"}
            )
            close["FACTSET_ID"] = ticker
            volume = raw[["Volume"]].reset_index().rename(
                columns={"index": "DATE", "Date": "DATE", "Volume": "ADJUSTED_VOLUME"}
            )
            volume["FACTSET_ID"] = ticker

        prices_df = close.merge(volume, on=["DATE", "FACTSET_ID"]).dropna(
            subset=["ADJUSTED_PRICE", "ADJUSTED_VOLUME"]
        )
        prices_df["DATE"] = pd.to_datetime(prices_df["DATE"])
        return prices_df.reset_index(drop=True)

    def _fetch_fundamentals(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        _EMPTY = pd.DataFrame(
            columns=["FACTSET_ID", "DATE", "SALES_LTM", "COGS_LTM", "EPS_LTM", "EBITDA_LTM"]
        )

        try:
            import simfin as sf

            sf.set_api_key(os.getenv("SIMFIN_API_KEY", "free"))
            sf.set_data_dir(Path.home() / ".simfin")

            income = sf.load_income(variant="ttm", market="us")
            cashflow = sf.load_cashflow(variant="ttm", market="us")

            income = income.reset_index()
            cashflow = cashflow.reset_index()

            # SimFin uses "." ticker format; universe uses "-"
            sp500_simfin = [t.replace("-", ".") for t in tickers]
            income = income[income["Ticker"].isin(sp500_simfin)]
            cashflow = cashflow[cashflow["Ticker"].isin(sp500_simfin)]

            join_keys = ["Ticker", "Fiscal Year", "Fiscal Period"]
            # Only keep D&A from cashflow to avoid column conflicts
            cf_cols = join_keys + ["Depreciation & Amortization"]
            # Keep only columns that exist
            cf_cols = [c for c in cf_cols if c in cashflow.columns]

            merged = income.merge(cashflow[cf_cols], on=join_keys, how="left")

            merged["EBITDA_LTM"] = (
                merged.get("Operating Income (Loss)", pd.Series(0, index=merged.index)).fillna(0)
                + merged.get("Depreciation & Amortization", pd.Series(0, index=merged.index)).fillna(0)
            )

            merged = merged.rename(
                columns={
                    "Ticker": "FACTSET_ID",
                    "Publish Date": "DATE",
                    "Revenue": "SALES_LTM",
                    "Cost of Revenue": "COGS_LTM",
                    "EPS Diluted": "EPS_LTM",
                }
            )

            merged["FACTSET_ID"] = merged["FACTSET_ID"].str.replace(".", "-", regex=False)

            needed = ["FACTSET_ID", "DATE", "SALES_LTM", "COGS_LTM", "EPS_LTM", "EBITDA_LTM"]
            # Keep only columns that exist after rename
            needed = [c for c in needed if c in merged.columns]
            fund = merged[needed].dropna(subset=["DATE"])
            fund["DATE"] = pd.to_datetime(fund["DATE"])
            fund = fund[
                (fund["DATE"] >= pd.Timestamp(start_date))
                & (fund["DATE"] <= pd.Timestamp(end_date))
            ]

            # Forward-fill each ticker's fundamentals to every trading day
            trading_days = pd.bdate_range(start=start_date, end=end_date)
            value_cols = [c for c in needed if c not in ("FACTSET_ID", "DATE")]

            fund_indexed = fund.set_index(["FACTSET_ID", "DATE"])[value_cols]
            # Drop duplicate (ticker, date) entries, keeping the latest
            fund_indexed = fund_indexed[~fund_indexed.index.duplicated(keep="last")]

            all_tickers = fund_indexed.index.get_level_values("FACTSET_ID").unique()
            full_idx = pd.MultiIndex.from_product(
                [all_tickers, trading_days], names=["FACTSET_ID", "DATE"]
            )
            fund = (
                fund_indexed
                .reindex(full_idx)
                .groupby(level="FACTSET_ID")
                .ffill()
                .reset_index()
            )

            return fund

        except Exception as exc:
            print(
                f"\n  [DataLoader] WARNING: SimFin fetch failed — {exc}\n"
                f"  Continuing with empty fundamentals. Price-only features "
                f"(MOM12_1, MOM6_1, VOL_20D, LIQUIDITY) will still work.\n"
                f"  Fundamental features (EBITDA_MARGIN, SALES_GROWTH, EPS_GROWTH, "
                f"PRICE_TO_SALES) will be skipped for this run.\n"
            )
            return _EMPTY

    def _cache_path(self, table: str, start_date: str, end_date: str) -> Path:
        key = f"{table}|{start_date}|{end_date}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._cache_dir / f"{table}_{digest}.parquet"
