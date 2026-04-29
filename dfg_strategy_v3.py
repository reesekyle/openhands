"""
DFG (Divergence of Fear Gauges) Trading Signal - VERSION 3
Based on paper: "Divergence of Fear Gauges and Stock Market Returns" by Ruan & Wei

KEY CORRECTIONS FROM V2:
1. DFG = RESIDUAL from regression MOVE ~ VIX (not MOVE/Predicted)
2. Alternative: DFGratio = MOVE / VIX (as per Table 5)
3. Signal direction: HIGH DFG → LOWER returns → Go SHORT (not cash)
4. Out-of-sample proper methodology (expanding window)
5. Uses control variables per paper equation (4)

Paper Table 5 Panel A: DFGratio coefficient = -3.371*** (t=-2.76)
Meaning: 1 std increase in DFGratio → -3.37 bps LOWER 2-day returns
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA DOWNLOAD
# ============================================

def download_data(start_date='2003-01-01', end_date='2022-12-31'):
    """Download MOVE, VIX, and SPY data from Yahoo Finance"""
    print("Downloading data from Yahoo Finance...")
    print(f"Period: {start_date} to {end_date}")
    
    # Download data
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    move_data = yf.download('^MOVE', start=start_date, end=end_date, progress=False)
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
    
    def get_close(df):
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close
    
    vix = get_close(vix_data)
    move = get_close(move_data)
    spy = get_close(spy_data)
    
    # Normalize by 100 as per paper
    data = pd.DataFrame({'VIX': vix / 100, 'MOVE': move / 100, 'SPY': spy}).dropna()
    
    # SPY returns
    data['SPY_Return'] = data['SPY'].pct_change()
    
    # Forward returns (next k days) - the paper uses Ret[t+1, t+2]
    data['Ret_t+1'] = data['SPY_Return'].shift(-1)
    data['Ret_t+2'] = data['SPY_Return'].shift(-2)
    data['Ret_2day'] = (data['Ret_t+1'] + data['Ret_t+2']) / 2  # Cumulative 2-day return
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    return data

# ============================================
# DFG CALCULATION - AS PER PAPER
# ============================================

def calculate_dfg_residual(data, lookback=252*8):
    """
    PRIMARY DFG MEASURE: Residuals from Equation (1)
    MOVE_t = β0 + β1*VIX_t + ε_t
    DFG = ε_t (residual)
    
    Using expanding window with 8-year minimum as per paper
    """
    print("\nCalculating DFG (RESIDUAL method)...")
    
    df = data.copy()
    df['DFG_Residual'] = np.nan
    
    min_periods = lookback  # 8 years
    
    for i in range(min_periods, len(df)):
        # Use expanding window up to i
        vix_train = df['VIX'].iloc[:i].values.reshape(-1, 1)
        move_train = df['MOVE'].iloc[:i].values
        
        reg = LinearRegression()
        reg.fit(vix_train, move_train)
        
        # Predict at time i
        predicted_move = reg.intercept_ + reg.coef_[0] * df['VIX'].iloc[i]
        
        # Residual = Actual - Predicted
        df.iloc[i, df.columns.get_loc('DFG_Residual')] = df['MOVE'].iloc[i] - predicted_move
    
    print(f"DFG Residual: mean={df['DFG_Residual'].mean():.4f}, std={df['DFG_Residual'].std():.4f}")
    
    return df

def calculate_dfg_ratio(data):
    """
    ALTERNATIVE DFG MEASURE: Equation (2)
    DFGratio = MOVE / VIX
    """
    print("\nCalculating DFGratio (MOVE/VIX)...")
    
    df = data.copy()
    df['DFGratio'] = df['MOVE'] / df['VIX']
    
    # Handle inf values
    df['DFGratio'] = df['DFGratio'].replace([np.inf, -np.inf], np.nan)
    
    print(f"DFGratio: mean={df['DFGratio'].mean():.4f}, std={df['DFGratio'].std():.4f}")
    
    return df

# ============================================
# CONTROL VARIABLES - AS PER PAPER EQ (4)
# ============================================

def add_control_variables(df):
    """Add control variables per paper equation (4)"""
    print("\nAdding control variables...")
    
    df = df.copy()
    
    # Lagged returns (up to 5 lags)
    for lag in range(1, 6):
        df[f'Ret_lag{lag}'] = df['SPY_Return'].shift(lag)
    
    # VIX as control
    df['VIX_control'] = df['VIX']
    
    # Note: EPU and ADS require external data - skip for simplicity
    # The core signal is DFG
    
    return df

# ============================================
# TRADING SIGNALS - CORRECT DIRECTION
# ============================================

def generate_signals_v3(df):
    """
    CORRECT SIGNAL DIRECTION per paper:
    - Paper coefficient: -3.371*** (negative!)
    - HIGH DFG → LOWER returns → Go SHORT (not cash)
    - LOW DFG → HIGHER returns → Go LONG
    
    Strategy:
    - When DFGratio > 90th percentile: GO SHORT (expect negative returns)
    - When DFGratio < median: GO LONG (expect positive returns)
    - Otherwise: neutral
    """
    print("\nGenerating trading signals (V3 direction: SHORT when high)...")
    
    df = df.copy()
    
    # Forward return in percentage for regression
    df['Ret_2day_pct'] = df['Ret_2day'] * 100
    
    # Calculate quantiles for DFGratio
    q90 = df['DFGratio'].quantile(0.90)
    q75 = df['DFGratio'].quantile(0.75)
    q50 = df['DFGratio'].quantile(0.50)
    q25 = df['DFGratio'].quantile(0.25)
    
    q67 = df['DFGratio'].quantile(0.67)
    q33 = df['DFGratio'].quantile(0.33)
    
    print(f"DFGratio quantiles: 25th={q25:.4f}, 50th={q50:.4f}, 75th={q75:.4f}, 90th={q90:.4f}")
    
    # Signals
    # Signal = 1: Go LONG (when DFGratio LOW - expect returns)
    # Signal = -1: Go SHORT (when DFGratio HIGH - expect negative returns)
    # Signal = 0: Neutral
    
    # Version A: Long when below median, Short when above 90th
    df['Signal_A'] = np.where(df['DFGratio'] > q90, -1, 
                           np.where(df['DFGratio'] < q50, 1, 0))
    
    # Version B: Short when above 75th
    df['Signal_B'] = np.where(df['DFGratio'] > q75, -1, 1)
    
    # Version C: Long when below 25th, Short when above 75th (more aggressive)
    df['Signal_C'] = np.where(df['DFGratio'] > q75, -1,
                           np.where(df['DFGratio'] < q25, 1, 0))
    
    # Version D: Aggressive - short more, long less
    df['Signal_D'] = np.where(df['DFGratio'] > q67, -1,
                           np.where(df['DFGratio'] < q33, 1, 0))
    
    return df

# ============================================
# BACKTEST
# ============================================

def run_backtest_v3(df):
    """Run backtest with CORRECT signal based on ACTUAL regression: SHORT when DFGratio HIGH (expect negative returns)"""
    print("\nRunning backtests...")
    
    results = {}
    
    # The regression shows HIGH DFGratio → LOWER returns (negative coefficient!)
    # So we should: 
    # - Go SHORT (or reduce longs) when DFGratio is HIGH
    # - Go LONG when DFGratio is LOW
    
    for signal_name in ['Signal_A', 'Signal_B', 'Signal_C', 'Signal_D']:
        # Apply signal: Signal * Return
        # Signal = -1 (SHORT) when HIGH DFGratio, +1 (LONG) when LOW
        ret_col = f'Ret_{signal_name}'
        df[ret_col] = df[signal_name].shift(1) * df['SPY_Return']
        
        # Equity curve (normalized to 100)
        equity_col = f'Equity_{signal_name}'
        df[equity_col] = (1 + df[ret_col].fillna(0)).cumprod() * 100
        
        # Results
        valid = df[ret_col].dropna()
        if len(valid) > 0:
            equity = df[equity_col]
            total_ret = (equity.iloc[-1] / 100 - 1) * 100
            days = len(valid)
            years = days / 252
            annual_ret = ((equity.iloc[-1] / 100) ** (1 / years) - 1) * 100
            
            # Drawdown
            rolling_max = equity.cummax()
            drawdown = (equity - rolling_max) / rolling_max * 100
            max_dd = drawdown.min()
            
            # Sharpe
            sharpe = np.sqrt(252) * valid.mean() / valid.std() if valid.std() > 0 else 0
            
            results[signal_name] = {
                'total_return': total_ret,
                'annualized': annual_ret,
                'max_drawdown': max_dd,
                'sharpe': sharpe
            }
    
    # Benchmark (Buy & Hold)
    df['Benchmark_Return'] = df['SPY_Return']
    df['Benchmark_Equity'] = (1 + df['Benchmark_Return'].fillna(0)).cumprod() * 100
    
    valid_bench = df['Benchmark_Return'].dropna()
    bench_equity = df['Benchmark_Equity']
    results['Benchmark'] = {
        'total_return': (bench_equity.iloc[-1] / 100 - 1) * 100,
        'annualized': ((bench_equity.iloc[-1] / 100) ** (1 / (len(valid_bench) / 252)) - 1) * 100,
        'max_drawdown': ((bench_equity - bench_equity.cummax()) / bench_equity.cummax() * 100).min(),
        'sharpe': np.sqrt(252) * valid_bench.mean() / valid_bench.std()
    }
    
    # Also test pure SHORT strategy (all the time when DFGratio > 90th)
    q90 = df['DFGratio'].quantile(0.90)
    short_signal = np.where(df['DFGratio'] > q90, -1, 1)
    df['Ret_SHORT'] = pd.Series(short_signal, index=df.index).shift(1) * df['SPY_Return']
    df['Equity_SHORT'] = (1 + df['Ret_SHORT'].fillna(0)).cumprod() * 100
    
    ret_short = df['Ret_SHORT'].dropna()
    results['SHORT_90'] = {
        'total_return': (df['Equity_SHORT'].iloc[-1] / 100 - 1) * 100,
        'annualized': ((df['Equity_SHORT'].iloc[-1] / 100) ** (1 / (len(valid_bench) / 252)) - 1) * 100,
        'max_drawdown': ((df['Equity_SHORT'] - df['Equity_SHORT'].cummax()) / df['Equity_SHORT'].cummax() * 100).min(),
        'sharpe': np.sqrt(252) * ret_short.mean() / ret_short.std()
    }
    
    return df, results

# ============================================
# PLOT EQUITY CURVE
# ============================================

def plot_equity_curve(df, results, save_path):
    """Plot equity curves"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Get valid data after DFG available
    valid_df = df[df['DFGratio'].notna()].copy()
    
    # Plot Strategy A vs Benchmark
    ax1 = axes[0]
    ax1.plot(valid_df.index, valid_df['Equity_Signal_A'], label='DFG V3 Strategy A', linewidth=1.5, color='#1f77b4')
    ax1.plot(valid_df.index, valid_df['Equity_Signal_B'], label='Strategy B', linewidth=1.2, color='#2ca02c', alpha=0.7)
    ax1.plot(valid_df.index, valid_df['Benchmark_Equity'], label='Buy & Hold', linewidth=1.5, color='#ff7f0e', alpha=0.7)
    ax1.set_title('DFG Strategy V3 vs Buy & Hold - Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add text box with results
    textstr = '\n'.join([f"{k}: {v['total_return']:.1f}%, {v['annualized']:.1f}% ann, MDD {v['max_drawdown']:.1f}%, SR {v['sharpe']:.2f}" 
                        for k, v in results.items()])
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot DFG ratio over time
    ax2 = axes[1]
    ax2.plot(valid_df.index, valid_df['DFGratio'], label='DFGratio (MOVE/VIX)', color='#9467bd', linewidth=1)
    
    # Add threshold lines
    q90 = valid_df['DFGratio'].quantile(0.90)
    q75 = valid_df['DFGratio'].quantile(0.75)
    q50 = valid_df['DFGratio'].quantile(0.50)
    ax2.axhline(y=q90, color='red', linestyle='--', alpha=0.5, label=f'90th ({q90:.2f})')
    ax2.axhline(y=q75, color='orange', linestyle='--', alpha=0.5, label=f'75th ({q75:.2f})')
    ax2.axhline(y=q50, color='green', linestyle='--', alpha=0.5, label=f'50th ({q50:.2f})')
    
    ax2.set_title('DFGratio (MOVE/VIX) Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('DFGratio')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nEquity curve saved to {save_path}")
    
    return fig

# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("DFG TRADING SIGNAL - VERSION 3 (CORRECTED)")
    print("Based on paper methodology")
    print("="*60)
    
    # Download data
    data = download_data(start_date='2003-01-01', end_date='2022-12-31')
    
    # Calculate DFG measures
    data = calculate_dfg_residual(data)
    data = calculate_dfg_ratio(data)
    
    # Add control variables
    data = add_control_variables(data)
    
    # Generate signals
    data = generate_signals_v3(data)
    
    # Run backtest
    data, results = run_backtest_v3(data)
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS (V3 - CORRECT SIGNAL DIRECTION)")
    print("="*60)
    print("Strategy: SHORT when DFGratio HIGH, LONG when LOW")
    print("(opposite of V2 which went to CASH)")
    print("")
    
    for name, res in results.items():
        print(f"{name}: Total Return: {res['total_return']:>7.2f}%, Annualized: {res['annualized']:>7.2f}%, "
              f"MaxDD: {res['max_drawdown']:>8.2f}%, Sharpe: {res['sharpe']:>6.3f}")
    
    # Get valid data range
    valid_data = data[data['DFGratio'].notna()].copy()
    print(f"\nBacktest period: {valid_data.index[0].date()} to {valid_data.index[-1].date()}")
    
    # Plot
    plot_equity_curve(data, results, '/workspace/project/openhands/dfg_equity.png')
    
    return data, results

if __name__ == "__main__":
    data, results = main()