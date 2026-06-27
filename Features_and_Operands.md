# Feature Engine

## Raw data sources

### Prices — yfinance (daily, 2021-01-01 → 2026-06-01, cached as parquet)

- `ADJUSTED_PRICE`: Split- and dividend-adjusted close price 
- `ADJUSTED_VOLUME`: Daily share volume 

### Fundamentals — SimFin TTM & quarterly data

Sourced from two datasets available on the free SimFin tier:

**Income statement**

**Cash flow statement **


### Universe — Wikipedia S&P 500 list (static, cached as parquet)

- `SECTOR`: GICS sector label
- `COUNTRY`: Always `US` for this universe

### Not available on free SimFin tier (paywalled)

- Balance sheet
- `Quarterly / Annual` variants of income and cashflow — same

---

## Computable features

These are computed on demand by `core/features.py` from the raw data above. Only features listed in an alpha config's `features` field are computed.

### Price-based

| Feature | Formula | Description |
|---------|---------|-------------|
| `MOM12_1` | `price[t-21] / price[t-252] - 1` | 12-month return, skipping most recent month (standard skip-1 momentum) |
| `MOM6_1` | `price[t-21] / price[t-126] - 1` | 6-month return, skipping most recent month |
| `VOL_20D` | `std(log_returns, 20d)` | 20-day rolling realised volatility of log returns |
| `LIQUIDITY` | `mean(price × volume, 20d)` | 20-day rolling average dollar volume |

### Fundamental-based

| Feature           | Formula | Description                                                                      |
|-------------------|---------|----------------------------------------------------------------------------------|
| `EBITDA_MARGIN`   | `EBITDA_LTM / SALES_LTM` | Operating profitability margin                                                   |
| `NET_MARGIN`      | `NET_INCOME_LTM / SALES_LTM` | Bottom-line profitability margin                                                 |
| `SALES_GROWTH`    | `SALES_LTM / SALES_LTM[t-252] - 1` | Year-over-year revenue growth                                                    |
| `EPS_GROWTH`      | `EPS_LTM / EPS_LTM[t-252] - 1` | Year-over-year EPS growth                                                        |
| `SALES_PER_SHARE` | `SALES_LTM / SHARES_DILUTED` | Sales per share (intermediate; not directly used in formulas)                    |
| `PRICE_TO_SALES`  | `ADJUSTED_PRICE / (SALES_LTM / SHARES_DILUTED)` | Price-to-sales ratio on a per-share basis                           |

---

## Operands

These are available inside formula strings evaluated by `core/formula_eval.py`. All operators are cross-sectional (applied across stocks at each rebalancing date, not across time).

### Supported

| Operand | Signature | Description |
|---------|-----------|-------------|
| `rank()` | `rank(series)` | Percentile rank in [0, 1] cross-sectionally |
| `zscore()` | `zscore(series)` | Cross-sectional z-score: `(x - mean) / std` |
| `log()` | `log(series)` | Natural log, clipped at 1e-9 to avoid `-inf` |
| `abs()` | `abs(series)` | Absolute value |
| `sign()` | `sign(series)` | Sign: -1, 0, or +1 |

Standard arithmetic operators also work inside formulas: `+`, `-`, `*`, `/`, `**`, and parentheses up to a nesting depth of 5.

### Not supported (raise `NotImplementedError` at runtime)

| Operand | Reason |
|---------|--------|
| `delta()` | Requires time-series context; pre-compute as a named feature instead |
| `ts_mean()` | Same — use a rolling feature like `MOM12_1` |
| `ts_std()` | Same — use `VOL_20D` |

### Example formulas

```
rank(EBITDA_MARGIN) + 0.5 * rank(MOM12_1)
rank(NET_MARGIN) - rank(PRICE_TO_SALES)
rank(MOM6_1) * rank(LIQUIDITY)
rank(EPS_GROWTH) + 0.5 * rank(SALES_GROWTH)
rank(VOL_20D) * -1
```
