"""
Configuration file for EDS Factor Model
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Snowflake Connection Configuration
SNOWFLAKE_CONFIG = {
    'account': os.getenv('SNOWFLAKE_ACCOUNT', ''),
    'user': os.getenv('SNOWFLAKE_USER', ''),
    'password': os.getenv('SNOWFLAKE_PASSWORD', ''),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', ''),
    'database': os.getenv('SNOWFLAKE_DATABASE', ''),
    'schema': os.getenv('SNOWFLAKE_SCHEMA', ''),
    'role': os.getenv('SNOWFLAKE_ROLE', '')
}

# Model Parameters
TARGET_CORRELATION = 0.70  # Minimum correlation with S&P500
LOOKBACK_PERIOD = 252  # Trading days (1 year)
MIN_DATA_POINTS = 60  # Minimum data points required for factor calculation

# Data Filtering Parameters (to avoid memory issues)
MAX_STOCKS = 500  # Maximum number of stocks to process (set to None for all stocks)
# If you have memory issues, reduce this number (e.g., 500, 1000, 5000)
# The code will randomly sample stocks if universe exceeds this limit

# S&P500 Ticker
SP500_TICKER = '^GSPC'

# Snowflake Table Names
SNOWFLAKE_TABLES = {
    'fundamentals': 'EDS_FACTORS_FUNDAMENTALS_NTM_LTM',
    'universe': 'EDS_FACTORS_UNIVERSE',
    'prices': 'SPLIT_ADJUSTED_PRICES_EDS_FACTORS',
    'exchange_rates': 'EXCHANGERATES',
    'market_value': 'MARKET_VALUE_HISTORY',
    'enterprise_value': 'ENTERPRISEVALUE_HISTORY'
}

# Factor Construction Parameters (MSCI-style)
FACTOR_PARAMS = {
    'momentum_lookback': 252,  # 12 months (252 trading days)
    'momentum_exclude': 21,  # Exclude last month (21 trading days)
    'volatility_lookback': 60,  # 60-day rolling volatility
    'liquidity_lookback': 20,  # 20-day dollar volume average
    'winsorize_lower': 0.01,  # 1% lower percentile
    'winsorize_upper': 0.99,  # 99% upper percentile
    'neutralize': False  # Whether to neutralize by industry/country
}

# Factor Return Calculation Parameters
FACTOR_RETURN_PARAMS = {
    'include_intercept': True  # Whether to include intercept in factor return regression
    # If True: r_i,t = α_t + Σ(k) β_i,k,t * f_k,t + ε_i,t
    # If False: r_i,t = Σ(k) β_i,k,t * f_k,t + ε_i,t
    # With sum-to-zero dummies, intercept represents average return
}



