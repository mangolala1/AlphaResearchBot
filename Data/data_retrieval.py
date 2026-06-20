"""
Data retrieval module for fetching data from Snowflake
"""
import snowflake.connector
import pandas as pd
import yfinance as yf
from typing import Dict, List
from datetime import datetime, timedelta
import config


# Note: load_ticker_list function removed - we now use all stocks in the universe
# If ticker filtering is needed in the future, this function can be restored


class SnowflakeDataRetriever:
    """Class to handle data retrieval from Snowflake"""
    
    def __init__(self, config_dict: Dict = None):
        """
        Initialize Snowflake connection
        
        Args:
            config_dict: Dictionary with Snowflake connection parameters
        """
        self.config = config_dict or config.SNOWFLAKE_CONFIG
        self.conn = None
        
    def connect(self):
        """Establish connection to Snowflake"""
        try:
            self.conn = snowflake.connector.connect(
                account=self.config['account'],
                user=self.config['user'],
                password=self.config['password'],
                warehouse=self.config['warehouse'],
                database=self.config['database'],
                schema=self.config['schema'],
                role=self.config.get('role')
            )
            print("Successfully connected to Snowflake")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Snowflake: {str(e)}")
    
    def disconnect(self):
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close()
            print("Disconnected from Snowflake")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            cursor.close()
            return df
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    def get_stock_data(self, 
                      tickers: List[str],
                      start_date: str,
                      end_date: str,
                      price_table: str = 'stock_prices',
                      fundamental_table: str = 'stock_fundamentals') -> Dict[str, pd.DataFrame]:
        """
        Retrieve stock price and fundamental data from Snowflake
        
        Args:
            tickers: List of stock tickers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            price_table: Name of price data table
            fundamental_table: Name of fundamental data table
            
        Returns:
            Dictionary with 'prices' and 'fundamentals' DataFrames
        """
        ticker_list = "', '".join(tickers)
        
        # Query for price data
        price_query = f"""
        SELECT 
            ticker,
            date,
            open,
            high,
            low,
            close,
            volume,
            adj_close
        FROM {price_table}
        WHERE ticker IN ('{ticker_list}')
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY ticker, date
        """
        
        # Query for fundamental data
        fundamental_query = f"""
        SELECT 
            ticker,
            date,
            market_cap,
            pe_ratio,
            pb_ratio,
            ev_ebitda,
            roe,
            roa,
            debt_to_equity,
            revenue,
            earnings
        FROM {fundamental_table}
        WHERE ticker IN ('{ticker_list}')
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY ticker, date
        """
        
        prices_df = self.execute_query(price_query)
        fundamentals_df = self.execute_query(fundamental_query)
        
        return {
            'prices': prices_df,
            'fundamentals': fundamentals_df
        }
    
    def get_fundamentals_data(self, start_date: str, end_date: str, 
                              factset_ids: List[str] = None) -> pd.DataFrame:
        """
        Retrieve fundamentals data from EDS_FACTORS_FUNDAMENTALS_NTM_LTM table
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (minimum enforced: '2015-01-01')
            end_date: End date in 'YYYY-MM-DD' format
            factset_ids: Optional list of FACTSET_IDs to filter by
            
        Returns:
            DataFrame with fundamentals data
        """
        # Enforce minimum date of 2015-01-01
        min_date = '2015-01-01'
        effective_start_date = max(start_date, min_date) if start_date < min_date else start_date
        
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["fundamentals"]}'
        
        # Handle large factset_id lists by chunking the query
        if factset_ids and len(factset_ids) > 1000:
            # For very large lists, process in chunks to avoid query size limits
            print(f"  Processing {len(factset_ids)} stocks in chunks of 1000...")
            chunks = [factset_ids[i:i+1000] for i in range(0, len(factset_ids), 1000)]
            all_results = []
            
            for i, chunk in enumerate(chunks):
                if (i + 1) % 10 == 0:
                    print(f"    Processed {i+1}/{len(chunks)} chunks...")
                factset_list = "', '".join(chunk)
                factset_filter = f"AND FACTSET_ID IN ('{factset_list}')"
                
                query = f"""
                SELECT 
                    FACTSET_ID,
                    DATE,
                    SALES_LTM,
                    SALES_NTM,
                    COGS_LTM,
                    COGS_NTM,
                    EBITDA_LTM,
                    EBITDA_NTM,
                    EPS_LTM,
                    EPS_NTM
                FROM {table_name}
                WHERE DATE >= '{effective_start_date}' AND DATE <= '{end_date}'
                {factset_filter}
                ORDER BY FACTSET_ID, DATE
                """
                chunk_result = self.execute_query(query)
                if len(chunk_result) > 0:
                    all_results.append(chunk_result)
            
            if all_results:
                return pd.concat(all_results, ignore_index=True)
            else:
                return pd.DataFrame(columns=['FACTSET_ID', 'DATE', 'SALES_LTM', 'SALES_NTM', 
                                            'COGS_LTM', 'COGS_NTM', 'EBITDA_LTM', 'EBITDA_NTM', 
                                            'EPS_LTM', 'EPS_NTM'])
        else:
            factset_filter = ""
            if factset_ids:
                factset_list = "', '".join(factset_ids)
                factset_filter = f"AND FACTSET_ID IN ('{factset_list}')"
        
        query = f"""
        SELECT 
            FACTSET_ID,
            DATE,
            SALES_LTM,
            SALES_NTM,
            COGS_LTM,
            COGS_NTM,
            EBITDA_LTM,
            EBITDA_NTM,
            EPS_LTM,
            EPS_NTM
        FROM {table_name}
            WHERE DATE >= '{min_date}' AND DATE <= '{end_date}'
            {factset_filter}
        ORDER BY FACTSET_ID, DATE
        """
        
        return self.execute_query(query)
    
    def get_universe_data(self, bloomberg_tickers: List[str] = None) -> pd.DataFrame:
        """
        Retrieve universe/metadata data from EDS_FACTORS_UNIVERSE table
        Filtered by BLOOMBERG_TICKER if provided
        
        Args:
            bloomberg_tickers: Optional list of Bloomberg tickers to filter by (e.g., ['AAPL US', 'MSFT US'])
            
        Returns:
            DataFrame with universe data (FACTSET_ID, COUNTRY, SECTOR only)
        """
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["universe"]}'
        
        ticker_filter = ""
        if bloomberg_tickers:
            ticker_list = "', '".join(bloomberg_tickers)
            ticker_filter = f"WHERE BLOOMBERG_TICKER IN ('{ticker_list}')"
        
        query = f"""
        SELECT 
            FACTSET_ID,
            COUNTRY,
            SECTOR
        FROM {table_name}
        {ticker_filter}
        """
        
        return self.execute_query(query)
    
    def calculate_returns_from_prices(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from prices data
        
        Args:
            prices_df: DataFrame with price data (must have FACTSET_ID, DATE, ADJUSTED_PRICE)
            
        Returns:
            DataFrame with returns data (FSYM_ID, P_DATE, ONE_DAY_PCT)
            Format matches the old returns table structure for compatibility
        """
        if prices_df.empty:
            return pd.DataFrame(columns=['FSYM_ID', 'P_DATE', 'ONE_DAY_PCT'])
        
        # Make a copy to avoid modifying original
        df = prices_df[['FACTSET_ID', 'DATE', 'ADJUSTED_PRICE']].copy()
        
        # Sort by FACTSET_ID and DATE
        df = df.sort_values(['FACTSET_ID', 'DATE'])
        
        # Calculate daily returns: (P_t / P_{t-1}) - 1
        df['ONE_DAY_PCT'] = df.groupby('FACTSET_ID')['ADJUSTED_PRICE'].pct_change()
        
        # Remove first row for each stock (no previous price)
        df = df.dropna(subset=['ONE_DAY_PCT'])
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'FACTSET_ID': 'FSYM_ID',
            'DATE': 'P_DATE'
        })
        
        # Select only required columns
        returns_df = df[['FSYM_ID', 'P_DATE', 'ONE_DAY_PCT']].copy()
        
        return returns_df
    
    def get_prices_data(self, start_date: str, end_date: str, 
                        factset_ids: List[str] = None) -> pd.DataFrame:
        """
        Retrieve split-adjusted prices from SPLIT_ADJUSTED_PRICES_EDS_FACTORS table
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (minimum enforced: '2015-01-01')
            end_date: End date in 'YYYY-MM-DD' format
            factset_ids: Optional list of FACTSET_IDs to filter by
            
        Returns:
            DataFrame with price data including all columns from the table
        """
        # Enforce minimum date of 2015-01-01
        min_date = '2015-01-01'
        effective_start_date = max(start_date, min_date) if start_date < min_date else start_date
        
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["prices"]}'
        
        # Handle large factset_id lists by chunking the query
        if factset_ids and len(factset_ids) > 1000:
            print(f"  Processing {len(factset_ids)} stocks in chunks of 1000...")
            chunks = [factset_ids[i:i+1000] for i in range(0, len(factset_ids), 1000)]
            all_results = []
            
            for i, chunk in enumerate(chunks):
                if (i + 1) % 10 == 0:
                    print(f"    Processed {i+1}/{len(chunks)} chunks...")
                factset_list = "', '".join(chunk)
                factset_filter = f"AND FACTSET_ID IN ('{factset_list}')"
                
                query = f"""
                SELECT 
                    FACTSET_ID,
                    DATE,
                    SPLIT_FACTOR,
                    SPECIAL_DIVS_FACTOR,
                    UNADJUSTED_PRICE,
                    ADJUSTED_PRICE,
                    ADJUSTED_VOLUME,
                    ADJUSTED_PRICE_DAY_HIGH,
                    ADJUSTED_PRICE_DAY_LOW,
                    CURRENCY,
                    P_DIVS_PD,
                    P_SPLIT_FACTOR,
                    IS_HOLIDAY
                FROM {table_name}
                WHERE DATE >= '{effective_start_date}' AND DATE <= '{end_date}'
                {factset_filter}
                ORDER BY FACTSET_ID, DATE
                """
                chunk_result = self.execute_query(query)
                if len(chunk_result) > 0:
                    all_results.append(chunk_result)
            
            if all_results:
                return pd.concat(all_results, ignore_index=True)
            else:
                return pd.DataFrame(columns=['FACTSET_ID', 'DATE', 'SPLIT_FACTOR', 'SPECIAL_DIVS_FACTOR',
                                            'UNADJUSTED_PRICE', 'ADJUSTED_PRICE', 'ADJUSTED_VOLUME',
                                            'ADJUSTED_PRICE_DAY_HIGH', 'ADJUSTED_PRICE_DAY_LOW', 'CURRENCY',
                                            'P_DIVS_PD', 'P_SPLIT_FACTOR', 'IS_HOLIDAY'])
        else:
            factset_filter = ""
            if factset_ids:
                factset_list = "', '".join(factset_ids)
                factset_filter = f"AND FACTSET_ID IN ('{factset_list}')"
            
            query = f"""
            SELECT 
                FACTSET_ID,
                DATE,
                SPLIT_FACTOR,
                SPECIAL_DIVS_FACTOR,
                UNADJUSTED_PRICE,
                ADJUSTED_PRICE,
                ADJUSTED_VOLUME,
                ADJUSTED_PRICE_DAY_HIGH,
                ADJUSTED_PRICE_DAY_LOW,
                CURRENCY,
                P_DIVS_PD,
                P_SPLIT_FACTOR,
                IS_HOLIDAY
            FROM {table_name}
            WHERE DATE >= '{min_date}' AND DATE <= '{end_date}'
            {factset_filter}
            ORDER BY FACTSET_ID, DATE
            """
            
            return self.execute_query(query)
    
    def get_fundamentals_data_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve fundamentals data for full universe by date range (no factset_ids filtering)
        This queries all stocks for the given date range
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (minimum enforced: '2015-01-01')
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with fundamentals data for all stocks in the date range
        """
        # Enforce minimum date of 2015-01-01
        min_date = '2015-01-01'
        effective_start_date = max(start_date, min_date) if start_date < min_date else start_date
        
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["fundamentals"]}'
        
        query = f"""
        SELECT 
            FACTSET_ID,
            DATE,
            SALES_LTM,
            SALES_NTM,
            COGS_LTM,
            COGS_NTM,
            EBITDA_LTM,
            EBITDA_NTM,
            EPS_LTM,
            EPS_NTM
        FROM {table_name}
        WHERE DATE >= '{effective_start_date}' AND DATE <= '{end_date}'
        ORDER BY DATE, FACTSET_ID
        """
        
        return self.execute_query(query)
    
    def get_prices_data_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve prices data for full universe by date range (no factset_ids filtering)
        This queries all stocks for the given date range, excluding holidays
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (minimum enforced: '2015-01-01')
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with price data for all stocks in the date range (excluding holiday rows)
        """
        # Enforce minimum date of 2015-01-01
        min_date = '2015-01-01'
        effective_start_date = max(start_date, min_date) if start_date < min_date else start_date
        
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["prices"]}'
        
        query = f"""
        SELECT 
            FACTSET_ID,
            DATE,
            SPLIT_FACTOR,
            SPECIAL_DIVS_FACTOR,
            UNADJUSTED_PRICE,
            ADJUSTED_PRICE,
            ADJUSTED_VOLUME,
            ADJUSTED_PRICE_DAY_HIGH,
            ADJUSTED_PRICE_DAY_LOW,
            CURRENCY,
            P_DIVS_PD,
            P_SPLIT_FACTOR,
            IS_HOLIDAY
        FROM {table_name}
        WHERE DATE >= '{effective_start_date}' AND DATE <= '{end_date}'
          AND IS_HOLIDAY = FALSE
        ORDER BY DATE, FACTSET_ID
        """
        
        return self.execute_query(query)
    
    def load_raw_data_for_model(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load universe metadata, fundamentals, prices, and returns once,
        for full date range including lookback.
        
        This function loads all data upfront (bulk load) without per-date loops.
        It's designed for Stage A of the two-stage pipeline.
        
        Args:
            start_date: Modeling period start date (e.g., '2020-01-01')
            end_date: Modeling period end date (e.g., '2025-12-01')
            
        Returns:
            Dictionary with keys: 'universe', 'fundamentals', 'prices', 'returns'
        """
        print("=" * 80)
        print("[STAGE A] Bulk Loading Raw Data")
        print("=" * 80)
        
        # Calculate lookback start date (need ~365 days for momentum)
        # We'll use '2019-04-01' as a safe buffer
        lookback_start_date = '2019-04-01'
        
        print(f"\n1. Loading universe metadata...")
        universe = self.get_universe_data(bloomberg_tickers=None)
        if len(universe) == 0:
            raise ValueError("No universe data found in the database.")
        
        # Add CONTINENT column if COUNTRY exists
        if 'COUNTRY' in universe.columns:
            from continent_mapping import get_continent
            universe['CONTINENT'] = universe['COUNTRY'].apply(get_continent)
        
        print(f"   ✓ Universe: {len(universe)} stocks")
        
        # Get FACTSET_IDs for filtering (optional, for smaller queries)
        factset_ids = universe['FACTSET_ID'].unique().tolist()
        print(f"   ✓ Total stocks: {len(factset_ids)}")
        
        print(f"\n2. Loading fundamentals data ({lookback_start_date} to {end_date})...")
        fundamentals = self.get_fundamentals_data_by_date_range(
            start_date=lookback_start_date,
            end_date=end_date
        )
        print(f"   ✓ Fundamentals: {len(fundamentals):,} rows")
        
        print(f"\n3. Loading prices data ({lookback_start_date} to {end_date})...")
        prices = self.get_prices_data_by_date_range(
            start_date=lookback_start_date,
            end_date=end_date
        )
        print(f"   ✓ Prices: {len(prices):,} rows")
        
        print(f"\n4. Calculating returns from prices...")
        returns = self.calculate_returns_from_prices(prices)
        print(f"   ✓ Returns: {len(returns):,} rows")
        
        # Ensure DATE columns are datetime
        for df_name, df in [('fundamentals', fundamentals), ('prices', prices)]:
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
        
        if 'P_DATE' in returns.columns:
            returns['P_DATE'] = pd.to_datetime(returns['P_DATE'])
        
        print("\n✓ Bulk data loading complete!")
        
        return {
            'universe': universe,
            'fundamentals': fundamentals,
            'prices': prices,
            'returns': returns
        }
    
    def get_all_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Retrieve all required data tables for factor model construction
        Enforces Date >= '2015-01-01'
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (minimum enforced: '2015-01-01')
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with all data tables
        """
        # Get universe data (all stocks, no ticker filtering)
        print("Loading universe data...")
        universe_df = self.get_universe_data(bloomberg_tickers=None)
        
        if len(universe_df) == 0:
            raise ValueError("No universe data found in the database.")
        
        # Extract FACTSET_IDs to filter all other tables
        factset_ids = universe_df['FACTSET_ID'].unique().tolist()
        print(f"Found {len(factset_ids)} stocks in universe")
        
        # Apply stock limit if configured (to avoid memory issues)
        if config.MAX_STOCKS is not None and len(factset_ids) > config.MAX_STOCKS:
            print(f"⚠ Limiting to {config.MAX_STOCKS} stocks (from {len(factset_ids)}) to avoid memory issues...")
            import random
            random.seed(42)  # For reproducibility
            factset_ids = random.sample(factset_ids, config.MAX_STOCKS)
            print(f"✓ Randomly sampled {len(factset_ids)} stocks")
        
        # Enforce minimum date of 2015-01-01
        min_date = '2015-01-01'
        effective_start_date = max(start_date, min_date) if start_date < min_date else start_date
        if effective_start_date != start_date:
            print(f"Note: Start date adjusted from {start_date} to {effective_start_date} (minimum: {min_date})")
        
        # Load all data tables filtered by FACTSET_IDs
        print("Loading fundamentals data...")
        fundamentals_df = self.get_fundamentals_data(effective_start_date, end_date, factset_ids=factset_ids)
        
        print("Loading prices data...")
        prices_df = self.get_prices_data(effective_start_date, end_date, factset_ids=factset_ids)
        
        print("Calculating returns from prices...")
        returns_df = self.calculate_returns_from_prices(prices_df)
        
        print("Loading exchange rates...")
        exchange_rates_df = self.get_exchange_rates(effective_start_date, end_date)
        
        print("Loading market value data...")
        market_value_df = self.get_market_value_data(effective_start_date, end_date, factset_ids=factset_ids)
        
        print("Loading enterprise value data...")
        enterprise_value_df = self.get_enterprise_value_data(effective_start_date, end_date, factset_ids=factset_ids)
        
        result = {
            'fundamentals': fundamentals_df,
            'universe': universe_df,
            'returns': returns_df,
            'prices': prices_df,
            'exchange_rates': exchange_rates_df,
            'market_value': market_value_df,
            'enterprise_value': enterprise_value_df
        }
        
        return result
    
    def get_exchange_rates(self, start_date: str, end_date: str, 
                          exchange_rate_table: str = None) -> pd.DataFrame:
        """
        Retrieve exchange rates for currency conversion to USD
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (minimum enforced: '2015-01-01')
            end_date: End date in 'YYYY-MM-DD' format
            exchange_rate_table: Optional table name for exchange rates.
                                If None, will use config.SNOWFLAKE_TABLES['exchange_rates']
            
        Returns:
            DataFrame with exchange rates (columns: DATE, CURRENCYCODE, EXCHANGERATE)
            Exchange rate is the number of USD per unit of the currency
        """
        # Enforce minimum date of 2015-01-01
        min_date = '2015-01-01'
        effective_start_date = max(start_date, min_date) if start_date < min_date else start_date
        
        # Use table name from config
        if exchange_rate_table is None:
            exchange_rate_table = config.SNOWFLAKE_TABLES.get('exchange_rates', 'EXCHANGERATES')
        
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{exchange_rate_table}'
        
        try:
            query = f"""
            SELECT 
                DATE,
                CURRENCYCODE,
                EXCHANGERATE
            FROM {table_name}
            WHERE DATE >= '{min_date}' AND DATE <= '{end_date}'
            ORDER BY DATE, CURRENCYCODE
            """
            df = self.execute_query(query)
            
            # Rename columns for consistency with convert_to_usd function
            df = df.rename(columns={
                'CURRENCYCODE': 'CURRENCY',
                'EXCHANGERATE': 'EXCHANGE_RATE_TO_USD'
            })
            
            return df
        except Exception as e:
            print(f"Warning: Could not retrieve exchange rates from table {exchange_rate_table}: {str(e)}")
            print("You may need to provide exchange rates manually or ensure the table exists.")
            print("Note: Enterprise value is in USD, but market value is in local currency.")
            print("      Currency conversion will be needed for market value calculations.")
            return pd.DataFrame(columns=['DATE', 'CURRENCY', 'EXCHANGE_RATE_TO_USD'])
    
    def get_market_value_data(self, start_date: str, end_date: str, 
                              factset_ids: List[str] = None) -> pd.DataFrame:
        """
        Retrieve market value (market cap) data from MARKET_VALUE_HISTORY table
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (minimum enforced: '2015-01-01')
            end_date: End date in 'YYYY-MM-DD' format
            factset_ids: Optional list of FACTSET_IDs to filter by
            
        Returns:
            DataFrame with market value data (FACTSET_ID, DATE, CURRENCY, MARKETCAP)
        """
        # Enforce minimum date of 2015-01-01
        min_date = '2015-01-01'
        effective_start_date = max(start_date, min_date) if start_date < min_date else start_date
        
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["market_value"]}'
        
        factset_filter = ""
        if factset_ids:
            factset_list = "', '".join(factset_ids)
            factset_filter = f"AND FACTSET_ID IN ('{factset_list}')"
        
        query = f"""
        SELECT 
            FACTSET_ID,
            DATE,
            CURRENCY,
            MARKETCAP
        FROM {table_name}
        WHERE DATE >= '{min_date}' AND DATE <= '{end_date}'
        {factset_filter}
        ORDER BY FACTSET_ID, DATE
        """
        
        return self.execute_query(query)
    
    def get_enterprise_value_data(self, start_date: str, end_date: str, 
                                  factset_ids: List[str] = None) -> pd.DataFrame:
        """
        Retrieve enterprise value data from ENTERPRISEVALUE_HISTORY table
        Enterprise value is already in USD
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (minimum enforced: '2015-01-01')
            end_date: End date in 'YYYY-MM-DD' format
            factset_ids: Optional list of FACTSET_IDs to filter by
            
        Returns:
            DataFrame with enterprise value data (FACTSET_ID, DATE, ENTERPRISE_VALUE, EV_COMPONENTS_DATE)
        """
        # Enforce minimum date of 2015-01-01
        min_date = '2015-01-01'
        effective_start_date = max(start_date, min_date) if start_date < min_date else start_date
        
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["enterprise_value"]}'
        
        factset_filter = ""
        if factset_ids:
            factset_list = "', '".join(factset_ids)
            factset_filter = f"AND FACTSET_ID IN ('{factset_list}')"
        
        query = f"""
        SELECT 
            FACTSET_ID,
            DATE,
            ENTERPRISE_VALUE,
            EV_COMPONENTS_DATE
        FROM {table_name}
        WHERE DATE >= '{min_date}' AND DATE <= '{end_date}'
        {factset_filter}
        ORDER BY FACTSET_ID, DATE
        """
        
        return self.execute_query(query)
    
    def get_sp500_returns(self, start_date: str, end_date: str) -> pd.Series:
        """
        Get S&P500 returns from Yahoo Finance
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Series with daily returns
        """
        sp500 = yf.download(config.SP500_TICKER, start=start_date, end=end_date)
        returns = sp500['Adj Close'].pct_change().dropna()
        returns.name = 'sp500_returns'
        return returns
    
    def is_holiday_date(self, date: str) -> bool:
        """
        Check if a given date is a holiday (all rows have IS_HOLIDAY=True)
        
        Args:
            date: Date in 'YYYY-MM-DD' format
            
        Returns:
            True if date is a holiday (should be skipped), False otherwise
        """
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["prices"]}'
        
        query = f"""
        SELECT DISTINCT IS_HOLIDAY
        FROM {table_name}
        WHERE DATE = '{date}'
        """
        
        result = self.execute_query(query)
        
        # If no data for this date, skip it
        if len(result) == 0:
            return True  # Skip dates with no data
        
        # If all rows have IS_HOLIDAY=True, it's a holiday
        if len(result) == 1 and result['IS_HOLIDAY'].iloc[0] == True:
            return True
        
        # If there are any rows with IS_HOLIDAY=False, it's not a holiday
        return False
    
    def get_data_for_date(self, date: str) -> Dict[str, pd.DataFrame]:
        """
        Retrieve all data for a single date from Snowflake
        
        Args:
            date: Date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with keys: 'fundamentals', 'prices', 'universe', 'returns', 
                                  'market_value', 'enterprise_value'
            Returns empty DataFrames if date is a holiday or has no data
        """
        # Check if date is a holiday
        if self.is_holiday_date(date):
            return {
                'fundamentals': pd.DataFrame(),
                'prices': pd.DataFrame(),
                'universe': pd.DataFrame(),
                'returns': pd.DataFrame(),
                'market_value': pd.DataFrame(),
                'enterprise_value': pd.DataFrame()
            }
        
        # Get prices for this date (filter out holiday rows)
        prices_df = self.get_prices_data(date, date)
        if len(prices_df) > 0:
            # Filter out holiday rows
            prices_df = prices_df[prices_df['IS_HOLIDAY'] == False]
        
        if len(prices_df) == 0:
            # No valid data for this date
            return {
                'fundamentals': pd.DataFrame(),
                'prices': pd.DataFrame(),
                'universe': pd.DataFrame(),
                'returns': pd.DataFrame(),
                'market_value': pd.DataFrame(),
                'enterprise_value': pd.DataFrame()
            }
        
        # Get FACTSET_IDs for this date
        factset_ids = prices_df['FACTSET_ID'].unique().tolist()
        
        # Get fundamentals for this date
        fundamentals_df = self.get_fundamentals_data(date, date, factset_ids=factset_ids)
        
        # Get universe data (all stocks, not filtered by date)
        universe_df = self.get_universe_data()
        # Filter to only stocks that appear in prices
        universe_df = universe_df[universe_df['FACTSET_ID'].isin(factset_ids)]
        
        # Calculate returns from prices
        returns_df = self.calculate_returns_from_prices(prices_df)
        
        # Get market value for this date
        market_value_df = pd.DataFrame()
        try:
            market_value_df = self.get_market_value_data(date, date, factset_ids=factset_ids)
        except:
            pass
        
        # Get enterprise value for this date
        enterprise_value_df = pd.DataFrame()
        try:
            enterprise_value_df = self.get_enterprise_value_data(date, date, factset_ids=factset_ids)
        except:
            pass
        
        return {
            'fundamentals': fundamentals_df,
            'prices': prices_df,
            'universe': universe_df,
            'returns': returns_df,
            'market_value': market_value_df,
            'enterprise_value': enterprise_value_df
        }
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        Get list of trading dates (non-holiday dates) between start_date and end_date
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            List of dates in 'YYYY-MM-DD' format (excluding holidays)
        """
        table_name = f'"{config.SNOWFLAKE_CONFIG["database"]}"."{config.SNOWFLAKE_CONFIG["schema"]}".{config.SNOWFLAKE_TABLES["prices"]}'
        
        query = f"""
        SELECT DISTINCT DATE
        FROM {table_name}
        WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
          AND IS_HOLIDAY = FALSE
        ORDER BY DATE
        """
        
        result = self.execute_query(query)
        
        if len(result) > 0:
            # Convert DATE column to datetime if it's not already
            result['DATE'] = pd.to_datetime(result['DATE'])
            dates = result['DATE'].dt.strftime('%Y-%m-%d').tolist()
            return dates
        else:
            return []


def convert_to_usd(value: pd.Series, currency: pd.Series, 
                   exchange_rates: pd.DataFrame, 
                   date: pd.Series = None) -> pd.Series:
    """
    Convert values from local currency to USD using exchange rates
    
    Note: Enterprise value is typically in USD, but market value is in local currency.
    Use this function to convert market value to USD for consistency.
    
    Example usage for market value conversion:
        market_value_usd = convert_to_usd(
            market_value_local_currency,
            prices_df['CURRENCY'],
            exchange_rates_df,
            date=prices_df['DATE']
        )
    
    Args:
        value: Series with values in local currency
        currency: Series with currency codes (e.g., 'USD', 'EUR', 'GBP')
        exchange_rates: DataFrame with columns (DATE, CURRENCY, EXCHANGE_RATE_TO_USD)
                       Exchange rate is the number of USD per unit of the currency
        date: Optional Series with dates for time-varying exchange rates
        
    Returns:
        Series with values converted to USD
    """
    result = value.copy()
    
    # If exchange rates are provided
    if len(exchange_rates) > 0 and 'CURRENCY' in exchange_rates.columns:
        # Create a mapping for exchange rates
        if date is not None and 'DATE' in exchange_rates.columns:
            # Time-varying exchange rates: merge on DATE and CURRENCY
            # Convert dates to same format
            exchange_rates_copy = exchange_rates.copy()
            exchange_rates_copy['DATE'] = pd.to_datetime(exchange_rates_copy['DATE'])
            date_series = pd.to_datetime(date)
            
            # Create a DataFrame for merging
            merge_df = pd.DataFrame({
                'value': value.values,
                'currency': currency.values,
                'date': date_series.values
            }, index=value.index)
            
            # Merge with exchange rates
            merged = merge_df.merge(
                exchange_rates_copy[['DATE', 'CURRENCY', 'EXCHANGE_RATE_TO_USD']],
                left_on=['date', 'currency'],
                right_on=['DATE', 'CURRENCY'],
                how='left'
            )
            
            # Convert: USD stays as is, others multiply by exchange rate
            # Fill missing exchange rates with 1.0 (assume USD if not found)
            merged['EXCHANGE_RATE_TO_USD'] = merged['EXCHANGE_RATE_TO_USD'].fillna(1.0)
            usd_mask = merged['currency'] == 'USD'
            result = merged['value'].copy()
            result[~usd_mask] = merged.loc[~usd_mask, 'value'] * merged.loc[~usd_mask, 'EXCHANGE_RATE_TO_USD']
            result.index = value.index
        else:
            # Static exchange rates (latest available per currency)
            exchange_rates_latest = exchange_rates.groupby('CURRENCY')['EXCHANGE_RATE_TO_USD'].last().to_dict()
            
            # Apply conversion
            result = value.copy()
            for curr in currency.unique():
                if curr == 'USD':
                    continue
                elif curr in exchange_rates_latest:
                    mask = currency == curr
                    result[mask] = value[mask] * exchange_rates_latest[curr]
                else:
                    # Currency not found in exchange rates - keep original value
                    mask = currency == curr
                    if mask.any():
                        print(f"Warning: Exchange rate not found for currency {curr}. Values will not be converted.")
    else:
        # No exchange rates provided - assume USD or warn
        non_usd_mask = currency != 'USD'
        if non_usd_mask.any():
            print("Warning: Exchange rates not provided. Non-USD values will not be converted.")
    
    return result


def get_date_range(lookback_days: int = 252) -> tuple:
    """
    Get start and end dates for data retrieval
    
    Args:
        lookback_days: Number of days to look back
        
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)  # Extra buffer
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


if __name__ == "__main__":
    data_retriever = SnowflakeDataRetriever()
    data_retriever.connect()
    data_retriever.disconnect()
    print("Data retrieval module executed successfully")


