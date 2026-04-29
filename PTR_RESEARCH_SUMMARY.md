# PTR Trading Strategy Research - Session Summary

## Date Range: April 2026

---

## Executive Summary

Tested multiple iterations of the PTR (Portfolio Tactical Rotation) strategy to optimize return-to-drawdown ratio using Treasury bond momentum signals.

**Best Strategy Found: ptr4_ief (EMB + Commodity signals with IEF allocation)**

---

## Key Findings

### 1. Original PTR (4 signals)
- Annual Return: 2.46%
- Max Drawdown: -7.29%
- Return/DD Ratio: 0.34

### 2. Signal Permutation Testing (ptr_scenarios.xlsx)

| Rank | Signal Combination | N | Ann% | DD% | Ratio |
|------|-----------------|---|------|------|------|
| 1 | commodity_returns | 1 | 2.55 | -5.11 | 0.50 |
| 2 | bond_trend+commodity_returns | 2 | 2.85 | -5.73 | 0.50 |
| 3 | bond_trend+equity_returns+commodity_returns | 3 | 2.62 | -6.39 | 0.41 |
| 4 | yield_spread+bond_trend+commodity_returns | 3 | 2.56 | -6.33 | 0.40 |
| 5 | bond_trend | 1 | 3.12 | -7.78 | 0.40 |
| **6** | **yield_spread+bond_trend+equity_returns+commodity_returns** | **4** | **2.46** | **-7.29** | **0.34** |

**Finding: Fewer signals often perform better**

### 3. Tweak Testing (ptr3_scenarios.xlsx)

| Tweak | Ann% | DD% | Ratio |
|------|------|------|------|
| Tweak 1: Yield as Leading Indicator | 2.46 | -6.46 | 0.38 |
| Tweak 2: Duration Rotation | 2.22 | -5.75 | 0.39 |
| Tweak 3: Gold Cross-Asset Filter | 2.30 | -6.79 | 0.34 |
| **Tweak 4: Intl Bond Momentum (EMB)** | **2.40** | **-5.64** | **0.42** |

**Finding: EMB (international bond ETF) momentum signal improves risk-adjusted returns**

### 4. Best Combination: ptr4

**Two signals only: EMB momentum + Commodity returns**

| Metric | Value |
|--------|-------|
| Annual Return | 2.35% |
| Max Drawdown | -3.78% |
| Return/DD Ratio | **0.62** |

This is 82% better risk-adjusted than the original 4-signal PTR!

### 5. ptr4 with Different Bond ETFs

**Comparison from 1/19/2010 inception:**

| Strategy | Final Value | Ann Ret% | Max DD% | Ann Vol% | Ret/DD | Ret/Vol |
|----------|------------|---------|---------|---------|--------|--------|---------|
| ptr4_ief | 1.45 | 2.63 | -3.78 | 3.10 | 0.70 | 0.85 |
| ptr4_ust | 1.59 | 2.94 | -10.07 | 6.09 | 0.29 | 0.48 |

**Winner: ptr4_ief** - Better risk-adjusted returns despite lower absolute return.

### 6. Recent Performance

**Since January 2025 (16 months):**
| Strategy | Ann Return |
|----------|-----------|
| ptr4_ief | 5.34% |
| ptr4_ust | 2.41% |

**2026 YTD:**
| Strategy | Total Return |
|----------|-------------|
| ptr4_ief | +0.49% |
| IEF (buy&hold) | 0% |
| ptr4_ust | -0.58% |

**Key insight: ptr4_ief generated +0.49% alpha over IEF in 2026 even though IEF was flat.**

---

## Files Generated

### Core Strategy Files
- `ptr_code.py` - Original PTR (4 signals)
- `ptr_test_signals.py` - Signal permutation tester
- `ptr3_tweak1.py` - Tweak 1 (Yield as indicator)
- `ptr3_tweak2.py` - Tweak 2 (Duration rotation)
- `ptr3_tweak3.py` - Tweak 3 (Gold filter)
- `ptr3_tweak4.py` - Tweak 4 (EMB momentum)

### Data & Charts
- `ptr_scenarios.xlsx` - Signal permutation results
- `ptr3_scenarios.xlsx` - Tweak comparison results
- `ptr4_position.xlsx` / `ptr4_position.png` - ptr4 signals
- `ptr4_equity.xlsx` / `ptr4_equity.png` - ptr4 with IEF vs TLT
- `ptr4_equity_2.png` - ptr4 with IEF vs UST

---

## Signal Definitions

### Original PTR (4 signals)
1. **Yield Spread**: z-score of (SPY - BIL) 12-month return spread
2. **Bond Trend**: +1 if IEF > BIL 12m return, else -1
3. **Equity Returns**: z-score of (SPY - BIL) excess return * -1
4. **Commodity Returns**: z-score of DBC 12m return * -1

### ptr4 (2 signals) - BEST
1. **INTL Bond Momentum**: z-score of EMB 12-month return
2. **Commodity Returns**: z-score of DBC 12-month return * -1

---

## Recommendations

1. **Use ptr4_ief** - Best risk-adjusted returns
2. Allocate between IEF (intermediate Treasuries) and BIL (cash) based on signal
3. Signal is generated monthly at end of month
4. Rebalance on last trading day of month

---

*Generated: April 2026*
*Repository: https://github.com/reesekyle/openhands*