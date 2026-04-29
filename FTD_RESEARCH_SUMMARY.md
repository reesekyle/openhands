# FTD Trading Strategy Research Summary

## Overview
This document summarizes the research and analysis of Follow-Through Day (FTD) trading strategies on SPY from 2000-2024.

## Original Concept
FTD (Follow-Through Day) is a technical trading signal where:
- Oversold conditions are identified using RSI (< 30)
- Subsequent FTD requires high IBS (> 0.5) AND daily return > 1%
- Entry on close of FTD, exit after N bars or when RSI reaches threshold

## Key Findings

### 1. FTD vs No-FTD Comparison
The most significant finding: **Strategies WITHOUT FTD significantly outperform those WITH FTD**

| Metric | With FTD | Without FTD |
|--------|---------|------------|
| Entry_10_Exit_RSI_75 Trades | 5 | 38 |
| Entry_10_Exit_RSI_75 Return | 18.32% | 316.72% |
| Entry_10_Exit_RSI_75 Profit Factor | 4.48 | 19.06 |
| Win Rate | 80.0% | 89.5% |

**Conclusion**: The FTD filter (IBS > 0.5 AND daily return > 1%) is too restrictive, filtering out many profitable trades.

### 2. Entry Strategies Analyzed (All with FTD)
- **Entry_1**: RSI < 30 only
- **Entry_5**: RSI < 40 + FTD
- **Entry_7**: RSI < 30 + FTD + Volume > 1.5x average
- **Entry_8**: RSI < 30 + FTD + Gap Up > 0.5%
- **Entry_10**: RSI < 30 + FTD + Price above MA200
- **Entry_15**: RSI < 30 + FTD + Wide bar (>1.5%)

### 3. Exit Strategies Analyzed
**RSI Threshold Exits (5)**:
- Exit_RSI_55: Exit when RSI > 55
- Exit_RSI_60: Exit when RSI > 60
- Exit_RSI_65: Exit when RSI > 65
- Exit_RSI_70: Exit when RSI > 70
- Exit_RSI_75: Exit when RSI > 75

**Alternative Exits (4)**:
- Exit_Time_3: Hold for 3 bars
- Exit_Time_10: Hold for 10 bars
- Exit_RSI_Oversold: Exit when RSI drops back below 30
- Exit_MA_Cross: Exit when price crosses below MA20

### 4. Best Performing Strategy (All Entries)
**Entry_10_Exit_RSI_75 WITHOUT FTD**:
- 38 trades
- 316.72% total return
- Profit Factor: 19.06
- Win Rate: 89.5%

### 5. Bug Fixes Applied
During analysis, two bugs were discovered and fixed:

1. **Entry_7 and Entry_8 were missing FTD criteria**
   - Initially only had RSI < 30 + additional filter
   - Fixed to include FTD: RSI < 30 + FTD + additional filter

2. **Entry_15 was missing FTD criteria**
   - Updated to include FTD

### 6. Files Generated
- `ftd_strategy.py` - Full strategy code
- `ftd_table2.xlsx` - Excel with 4 sheets:
  - perf_with_FTD: 45 strategies WITH FTD
  - perf_no_FTD: 45 strategies WITHOUT FTD
  - Entries: Entry rules
  - Exits: Exit rules
- `ftd_performance_with_ftd_all.png` - Equity curves WITH FTD
- `ftd_performance_without_ftd_all.png` - Equity curves WITHOUT FTD
- `ftd_comparison.png` - Direct comparison of Entry_10 with/without FTD

## Recommendations

1. **For Practical Trading**: The No-FTD version (RSI < 30 + Price above MA200) with RSI 75 exit shows the best risk-adjusted returns

2. **FTD as Filter**: Using FTD as a filter significantly reduces trade frequency and profitability

3. **Exit Strategy**: Higher RSI thresholds (70-75) produce better risk-adjusted returns than lower thresholds

## Technical Details

- Data: SPY daily from 2000-01-03 to 2024-12-30 (6,288 trading days)
- Indicators: RSI(14), IBS, MA(20,50,200), Volume ratios, Gap
- No position sizing (binary: flat or 100% long)
- Entry on close of signal bar
- Exit on close of bar that triggers exit condition

---
*Research conducted: April 2025*
*Tools: Python, pandas, yfinance, matplotlib*