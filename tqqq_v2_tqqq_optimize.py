"""TQQQ Strategy Parameter Optimization"""
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime

TICKER_TQQQ = "TQQQ"
TICKER_QQQ = "QQQ"
START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

def download_data():
    print(f"Downloading data...")
    tqqq = yf.download(TICKER_TQQQ, start=START_DATE, end=END_DATE, progress=False)
    qqq = yf.download(TICKER_QQQ, start=START_DATE, end=END_DATE, progress=False)
    
    def get_close_price(df, ticker):
        if isinstance(df.columns, pd.MultiIndex):
            for col in df.columns:
                if col[0] == 'Close' and (len(col) == 1 or col[1] == ticker):
                    return df[col]
        else:
            if 'Close' in df.columns:
                return df['Close']
            elif 'Adj Close' in df.columns:
                return df['Adj Close']
        raise KeyError(f"No close price column found")
    
    data = pd.DataFrame({
        'TQQQ': get_close_price(tqqq, TICKER_TQQQ),
        'QQQ': get_close_price(qqq, TICKER_QQQ)
    })
    data = data.dropna()
    return data

def calculate_features(data, n_days=10, zscore_window=10):
    data = data.copy()
    data['TQQQ_log_ret'] = np.log(data['TQQQ'] / data['TQQQ'].shift(1))
    data['QQQ_log_ret'] = np.log(data['QQQ'] / data['QQQ'].shift(1))
    
    data['TQQQ_N_ret'] = data['TQQQ_log_ret'].rolling(window=n_days).sum()
    data['QQQ_N_ret'] = data['QQQ_log_ret'].rolling(window=n_days).sum()
    data['QQQ_3x_N'] = 3 * data['QQQ_N_ret']
    data['ExcessN'] = data['TQQQ_N_ret'] - data['QQQ_3x_N']
    
    data['ExcessN_mean'] = data['ExcessN'].rolling(window=zscore_window).mean()
    data['ExcessN_std'] = data['ExcessN'].rolling(window=zscore_window).std()
    data['ExcessZ'] = (data['ExcessN'] - data['ExcessN_mean']) / data['ExcessN_std']
    
    return data

def run_backtest(data, entry_threshold, exit_threshold, max_hold_bars, fail_fast_threshold):
    data = data.copy()
    data['position'] = 0
    data['strategy_ret'] = 0.0
    data['cum_strategy'] = 1.0
    data['cum_qqq'] = 1.0
    
    position = 0
    prev_position = 0
    bars_held = 0
    
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['ExcessZ']):
            continue
            
        prev_position = position
        
        if i > 0:
            prev_excess_z = data.iloc[i-1]['ExcessZ']
        else:
            prev_excess_z = 0
        
        date = data.index[i]
        row = data.iloc[i]
        
        if position == 0 and prev_excess_z <= entry_threshold:
            position = 1
            bars_held = 0
        
        elif position == 1:
            bars_held += 1
            should_exit = (prev_excess_z >= exit_threshold or 
                          bars_held >= max_hold_bars or 
                          prev_excess_z <= fail_fast_threshold)
            if should_exit:
                position = 0
                bars_held = 0
        
        if prev_position == 1:
            data.loc[date, 'strategy_ret'] = row['TQQQ_log_ret']
        else:
            data.loc[date, 'strategy_ret'] = 0
    
    data['cum_strategy'] = (1 + data['strategy_ret']).cumprod()
    data['cum_qqq'] = (1 + data['QQQ_log_ret']).cumprod()
    
    return data

def calculate_metrics(data):
    total_strategy_ret = data['cum_strategy'].iloc[-1] - 1
    total_qqq_ret = data['cum_qqq'].iloc[-1] - 1
    
    years = len(data) / 252
    cagr_strategy = (1 + total_strategy_ret) ** (1/years) - 1
    cagr_qqq = (1 + total_qqq_ret) ** (1/years) - 1
    
    strat_vol = data['strategy_ret'].std() * np.sqrt(252)
    qqq_vol = data['QQQ_log_ret'].std() * np.sqrt(252)
    
    sharpe_strategy = cagr_strategy / strat_vol if strat_vol > 0 else 0
    sharpe_qqq = cagr_qqq / qqq_vol if qqq_vol > 0 else 0
    
    strat_cummax = data['cum_strategy'].cummax()
    strat_dd = (data['cum_strategy'] - strat_cummax) / strat_cummax
    max_dd_strategy = strat_dd.min()
    
    avg_capital_use = data['position'].mean()
    
    return {
        'total_ret': total_strategy_ret,
        'cagr': cagr_strategy,
        'sharpe': sharpe_strategy,
        'max_dd': max_dd_strategy,
        'volatility': strat_vol,
        'avg_capital': avg_capital_use
    }

# Download data once
data = download_data()

# Define parameter ranges to test
entry_thresholds = [-0.3, -0.5, -0.7, -1.0, -1.5, -2.0]
exit_thresholds = [0.0, 0.2, 0.5]
max_hold_bars = [3, 5, 7, 10]
n_days_options = [5, 10, 15, 20, 30]
zscore_windows = [5, 10, 20, 30]

results = []

print("Running parameter optimization...")
total = len(entry_thresholds) * len(exit_thresholds) * len(max_hold_bars) * len(n_days_options) * len(zscore_windows)
count = 0

for entry_t, exit_t, max_bars, n_days, z_win in product(
    entry_thresholds, exit_thresholds, max_hold_bars, n_days_options, zscore_windows
):
    count += 1
    if count % 500 == 0:
        print(f"Progress: {count}/{total}")
    
    # Skip invalid combinations
    if exit_t <= entry_t:
        continue
    
    try:
        test_data = calculate_features(data.copy(), n_days=n_days, zscore_window=z_win)
        test_data = test_data.dropna()
        
        result_data = run_backtest(test_data, entry_t, exit_t, max_bars, -3.0)
        metrics = calculate_metrics(result_data)
        
        # Skip unreasonable results
        if metrics['total_ret'] < 0 or metrics['sharpe'] < -1:
            continue
            
        results.append({
            'entry_threshold': entry_t,
            'exit_threshold': exit_t,
            'max_hold_bars': max_bars,
            'n_days': n_days,
            'zscore_window': z_win,
            **metrics
        })
    except Exception as e:
        continue

# Convert to DataFrame and sort
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("TOP 10 BY SHARPE RATIO:")
print("="*80)
top_sharpe = results_df.nlargest(10, 'sharpe')
print(top_sharpe[['entry_threshold', 'exit_threshold', 'max_hold_bars', 'n_days', 'zscore_window', 'total_ret', 'cagr', 'sharpe', 'max_dd', 'volatility', 'avg_capital']].to_string(index=False))

print("\n" + "="*80)
print("TOP 10 BY TOTAL RETURN:")
print("="*80)
top_return = results_df.nlargest(10, 'total_ret')
print(top_return[['entry_threshold', 'exit_threshold', 'max_hold_bars', 'n_days', 'zscore_window', 'total_ret', 'cagr', 'sharpe', 'max_dd', 'volatility', 'avg_capital']].to_string(index=False))

print("\n" + "="*80)
print("TOP 10 BY CAGR (Risk-Adjusted Return):")
print("="*80)
top_cagr = results_df.nlargest(10, 'cagr')
print(top_cagr[['entry_threshold', 'exit_threshold', 'max_hold_bars', 'n_days', 'zscore_window', 'total_ret', 'cagr', 'sharpe', 'max_dd', 'volatility', 'avg_capital']].to_string(index=False))

# Find best balance of return and drawdown
results_df['return_dd_ratio'] = results_df['total_ret'] / abs(results_df['max_dd'])
print("\n" + "="*80)
print("TOP 10 BY RETURN/DRAWDOWN RATIO:")
print("="*80)
top_rdd = results_df.nlargest(10, 'return_dd_ratio')
print(top_rdd[['entry_threshold', 'exit_threshold', 'max_hold_bars', 'n_days', 'zscore_window', 'total_ret', 'cagr', 'sharpe', 'max_dd', 'return_dd_ratio']].to_string(index=False))

# Save results
results_df.to_csv('/workspace/project/openhands/tqqq_optimization_results.csv', index=False)
print("\nAll results saved to tqqq_optimization_results.csv")

# Compare with original
print("\n" + "="*80)
print("COMPARISON WITH ORIGINAL STRATEGY:")
print("="*80)
original = {
    'entry_threshold': -0.5,
    'exit_threshold': 0.0,
    'max_hold_bars': 5,
    'n_days': 10,
    'zscore_window': 10
}
print(f"Original: entry={original['entry_threshold']}, exit={original['exit_threshold']}, max_bars={original['max_hold_bars']}, n_days={original['n_days']}, z_win={original['zscore_window']}")
print(f"Original result: Total=486.17%, CAGR=11.69%, Sharpe=0.28, MaxDD=-66.37%")

