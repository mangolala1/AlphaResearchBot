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

      prices_df     : TICKER, DATE, ADJUSTED_PRICE, ADJUSTED_VOLUME
      income_ttm_df : TICKER, DATE, <original SimFin income column names>
      cashflow_ttm_df: TICKER, DATE, <original SimFin cashflow column names>
      universe_df   : TICKER, SECTOR, COUNTRY

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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (prices_df, income_ttm_df, cashflow_ttm_df, universe_df).

        Fetches from yfinance + SimFin on first call for a given date range;
        reads parquet cache on subsequent calls unless no_cache=True.
        """
        prices_path = self._cache_path("prices", start_date, end_date)
        income_path = self._cache_path("income_ttm", start_date, end_date)
        cashflow_path = self._cache_path("cashflow_ttm", start_date, end_date)
        univ_path = self._cache_path("universe", start_date, end_date)

        need_prices = self._no_cache or not prices_path.exists()
        need_income = self._no_cache or not income_path.exists()
        need_cashflow = self._no_cache or not cashflow_path.exists()
        need_univ = self._no_cache or not univ_path.exists()

        if not need_prices and not need_income and not need_cashflow and not need_univ:
            return (
                pd.read_parquet(prices_path),
                pd.read_parquet(income_path),
                pd.read_parquet(cashflow_path),
                pd.read_parquet(univ_path),
            )

        print("[DataLoader] Fetching universe from Wikipedia...")
        universe_df = self._fetch_universe()
        universe_df.to_parquet(univ_path, index=False)

        tickers = universe_df["TICKER"].tolist()

        print(f"[DataLoader] Fetching prices from yfinance for {len(tickers)} tickers...")
        prices_df = self._fetch_prices(tickers, start_date, end_date)
        prices_df.to_parquet(prices_path, index=False)

        print("[DataLoader] Fetching income statement from SimFin...")
        income_ttm_df = self._fetch_income_TTM(tickers, start_date, end_date)
        income_ttm_df.to_parquet(income_path, index=False)

        print("[DataLoader] Fetching cash flow statement from SimFin...")
        cashflow_ttm_df = self._fetch_cashflow_TTM(tickers, start_date, end_date)
        cashflow_ttm_df.to_parquet(cashflow_path, index=False)

        return prices_df, income_ttm_df, cashflow_ttm_df, universe_df

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
        df = df.rename(columns={"Symbol": "TICKER", "GICS Sector": "SECTOR"})
        df["COUNTRY"] = "US"
        return df.reset_index(drop=True)

    def _fetch_prices(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        import logging
        _yf_log = logging.getLogger("yfinance")
        _prev_level = _yf_log.level
        _yf_log.setLevel(logging.CRITICAL)
        try:
            raw = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        finally:
            _yf_log.setLevel(_prev_level)

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
            close.columns = ["DATE", "TICKER", "ADJUSTED_PRICE"]
            volume.columns = ["DATE", "TICKER", "ADJUSTED_VOLUME"]
        else:
            # Single ticker
            ticker = tickers[0]
            close = raw[["Close"]].reset_index().rename(
                columns={"index": "DATE", "Date": "DATE", "Close": "ADJUSTED_PRICE"}
            )
            close["TICKER"] = ticker
            volume = raw[["Volume"]].reset_index().rename(
                columns={"index": "DATE", "Date": "DATE", "Volume": "ADJUSTED_VOLUME"}
            )
            volume["TICKER"] = ticker

        prices_df = close.merge(volume, on=["DATE", "TICKER"]).dropna(
            subset=["ADJUSTED_PRICE", "ADJUSTED_VOLUME"]
        )
        prices_df["DATE"] = pd.to_datetime(prices_df["DATE"])
        return prices_df.reset_index(drop=True)


    def _fetch_income_TTM(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        return self._fetch_simfin_TTM("income", tickers, start_date, end_date)

    def _fetch_cashflow_TTM(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        return self._fetch_simfin_TTM("cashflow", tickers, start_date, end_date)

    def _fetch_simfin_TTM(
        self, statement: str, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load a single SimFin TTM statement, clean it, and forward-fill to daily."""
        _EMPTY = pd.DataFrame(columns=["TICKER", "DATE"])
        _META_COLS = {"Fiscal Year", "Fiscal Period", "Currency",
                      "Report Date", "Restated Date", "SimFinId"}

        try:
            import simfin as sf

            sf.set_api_key(os.getenv("SIMFIN_API_KEY", "free"))
            sf.set_data_dir(Path.home() / ".simfin")

            if statement == "income":
                raw = sf.load_income(variant="ttm", market="us")
            else:
                raw = sf.load_cashflow(variant="ttm", market="us")

            fund = raw.reset_index()

            sp500_simfin = [t.replace("-", ".") for t in tickers]
            fund = fund[fund["Ticker"].isin(sp500_simfin)]

            fund = fund.rename(columns={"Ticker": "TICKER", "Publish Date": "DATE"})
            fund["TICKER"] = fund["TICKER"].str.replace(".", "-", regex=False)

            fund = fund.drop(columns=[c for c in fund.columns if c in _META_COLS], errors="ignore")
            fund = fund.dropna(subset=["DATE"])
            fund["DATE"] = pd.to_datetime(fund["DATE"])

            non_key = [c for c in fund.columns if c not in ("TICKER", "DATE")]
            sparse_cols = [c for c in non_key if fund[c].isna().sum() > 1000]
            if sparse_cols:
                print(f"  [DataLoader] {statement}: dropping {len(sparse_cols)} sparse columns: {sparse_cols}")
            fund = fund.drop(columns=sparse_cols)

            return self._prefetch_and_ffill(fund, start_date, end_date)

        except Exception as exc:
            print(f"\n  [DataLoader] WARNING: SimFin {statement} fetch failed — {exc}\n")
            return _EMPTY

    def _prefetch_and_ffill(
        self, fund: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Extend window 1 year back, forward-fill to business days, trim to start_date."""
        prefetch_start = (
            pd.Timestamp(start_date) - pd.DateOffset(years=1)
        ).strftime("%Y-%m-%d")
        fund = fund[
            (fund["DATE"] >= pd.Timestamp(prefetch_start))
            & (fund["DATE"] <= pd.Timestamp(end_date))
        ]

        trading_days = pd.bdate_range(start=prefetch_start, end=end_date)
        value_cols = [c for c in fund.columns if c not in ("TICKER", "DATE")]

        fund_indexed = fund.set_index(["TICKER", "DATE"])[value_cols]
        fund_indexed = fund_indexed[~fund_indexed.index.duplicated(keep="last")]

        all_tickers = fund_indexed.index.get_level_values("TICKER").unique()
        full_idx = pd.MultiIndex.from_product(
            [all_tickers, trading_days], names=["TICKER", "DATE"]
        )
        fund = (
            fund_indexed
            .reindex(full_idx)
            .groupby(level="TICKER")
            .ffill()
            .reset_index()
        )

        return fund[fund["DATE"] >= pd.Timestamp(start_date)]


    def _cache_path(self, table: str, start_date: str, end_date: str) -> Path:
        key = f"{table}|{start_date}|{end_date}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._cache_dir / f"{table}_{digest}.parquet"


if __name__ == "__main__":
    loader = DataLoader()
    prices_df, income_ttm_df, cashflow_ttm_df, universe_df = loader.load(
        start_date="2021-01-01", end_date="2026-06-01"
    )

    print(f"Prices: {len(prices_df)} rows")
    print(f"Income TTM: {len(income_ttm_df)} rows")
    print(f"Cashflow TTM: {len(cashflow_ttm_df)} rows")
    print(f"Universe: {len(universe_df)} rows")