### Snowflake data tables:
Query format: select * from EDS_DEV_BERKELEY.BERKELEY.Table_name

- `EDS_FACTORS_FUNDAMENTALS_NTM_LTM` (196.4M rows)
  - `FACTSET_ID`: Varchar
  - `DATE`: Date
  - `EPS_LTM`: Float
  - `EPS_NTM`: Float
  - `SALES_LTM`: Float
  - `SALES_NTM`: Float
  - `EBITDA_LTM`: Float
  - `EBITDA_NTM`: Float
  - `COGS_LTM`: Float
  - `COGS_NTM`: Float

- `EDS_FACTORS_UNIVERSE` (74.9k rows)
  - `FACTSET_ID`: Varchar
  - `COUNTRY`: Varchar
  - `SECTOR`: Varchar
  - `INDUSTRY`: Varchar
  - `INDUSTRYGROUP`: Varchar
  - `NAME`: Varchar
  - `BLOOMBERG_TICKER`: Varchar

- `SPLIT_ADJUSTED_PRICES_EDS_FACTORS` (313.3M rows)
  - `DATE`: Date
  - `FACTSET_ID`: Varchar
  - `SPECIAL_DIVS_FACTOR`: Float
  - `SPLIT_FACTOR`: Float
  - `UNADJUSTED_PRICE`: Float
  - `ADJUSTED_PRICE`: Float
  - `ADJUSTED_PRICE_DAY_HIGH`: Float
  - `ADJUSTED_PRICE_DAY_LOW`: Float
  - `ADJUSTED_VOLUME`: Float
  - `CURRENCY`: Varchar
  - `IS_HOLIDAY`: Boolean
  - `P_DIVS_PD`: Float
  - `P_SPLIT_FACTOR`: Float
