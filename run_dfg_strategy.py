"""
Run DFG Strategy and Generate Equity Curve
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

def download_data(start_date='2004-01-01', end_date='2022-12-31'):
    """Download MOVE, VIX, and SPY data from Yahoo Finance"""
    print("Downloading data from Yahoo Finance...")
    
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
    
    data = pd.DataFrame({'VIX': vix, 'MOVE': move, 'SPY': spy}).dropna()
    data['SPY_Return'] = data['SPY'].pct_change()
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    return data

# ============================================
# DFGRATIO CALCULATION
# ============================================

def calculate_dfgratio(data):
    """Calculate DFGratio using expanding window approach"""
    print("\nCalculating DFGratio with Expanding Window...")
    
    df = data.copy()
    min_periods = 252 * 8  # ~8 years
    
    df['DFG'] = np.nan
    
    for i in range(min_periods, len(df)):
        vix_train = df['VIX'].iloc[:i].values.reshape(-1, 1)
        move_train = df['MOVE'].iloc[:i].values
        
        reg = LinearRegression()
        reg.fit(vix_train, move_train)
        
        predicted_move = reg.intercept_ + reg.coef_[0] * df['VIX'].iloc[i]
        df.iloc[i, df.columns.get_loc('DFG')] = df['MOVE'].iloc[i] / predicted_move
    
    print(f"DFGratio calculated for {df['DFG'].notna().sum()} days")
    
    return df

# ============================================
# TRADING SIGNAL
# ============================================

def generate_signals(df):
    """Generate trading signals - Stay LONG, exit to CASH when DFG is HIGH"""
    
    # Calculate forward returns
    df['Ret_2day'] = (df['SPY_Return'].shift(1) + df['SPY_Return'].shift(2)) / 2
    
    # Strategy: Exit when DFG > 90th percentile
    threshold_90 = df['DFG'].quantile(0.90)
    df['Signal'] = np.where(df['DFG'] > threshold_90, 0, 1)  # 0=cash, 1=long
    
    return df

# ============================================
# BACKTEST & EQUITY CURVE
# ============================================

def run_backtest(df):
    """Run backtest and generate equity curve"""
    
    print("\nRunning backtest...")
    
    # Apply signal (shift to avoid look-ahead bias)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['SPY_Return']
    
    # Benchmark: Buy and Hold
    df['Benchmark_Return'] = df['SPY_Return']
    
    # Calculate equity curves (normalized to 100)
    df['Strategy_Equity'] = (1 + df['Strategy_Return'].fillna(0)).cumprod() * 100
    df['Benchmark_Equity'] = (1 + df['Benchmark_Return'].fillna(0)).cumprod() * 100
    
    # Fill NaN with previous value
    df['Strategy_Equity'] = df['Strategy_Equity'].ffill().fillna(100)
    df['Benchmark_Equity'] = df['Benchmark_Equity'].ffill().fillna(100)
    
    return df

# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("DFG TRADING SIGNAL - EQUITY CURVE GENERATION")
    print("="*60)
    
    # Download data
    data = download_data(start_date='2004-01-01', end_date='2022-12-31')
    
    # Calculate DFGratio
    data = calculate_dfgratio(data)
    
    # Generate signals
    data = generate_signals(data)
    
    # Run backtest
    data = run_backtest(data)
    
    # Get valid data (after DFG is available)
    valid_data = data[data['DFG'].notna()].copy()
    
    print(f"\nBacktest period: {valid_data.index[0].date()} to {valid_data.index[-1].date()}")
    
    # Calculate performance metrics
    strat_ret = valid_data['Strategy_Return'].dropna()
    bench_ret = valid_data['Benchmark_Return'].dropna()
    
    strat_equity = valid_data['Strategy_Equity']
    bench_equity = valid_data['Benchmark_Equity']
    
    # Final values
    strat_final = strat_equity.iloc[-1]
    bench_final = bench_equity.iloc[-1]
    
    strat_total_return = (strat_final / 100 - 1) * 100
    bench_total_return = (bench_final / 100 - 1) * 100
    
    days = len(strat_ret)
    years = days / 252
    
    strat_annual = ((strat_final / 100) ** (1 / years) - 1) * 100
    bench_annual = ((bench_final / 100) ** (1 / years) - 1) * 100
    
    # Max drawdown
    strat_rolling_max = strat_equity.cummax()
    strat_drawdown = (strat_equity - strat_rolling_max) / strat_rolling_max * 100
    strat_mdd = strat_drawdown.min()
    
    bench_rolling_max = bench_equity.cummax()
    bench_drawdown = (bench_equity - bench_rolling_max) / bench_rolling_max * 100
    bench_mdd = bench_drawdown.min()
    
    # Sharpe ratio
    strat_sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std() if strat_ret.std() > 0 else 0
    bench_sharpe = np.sqrt(252) * bench_ret.mean() / bench_ret.std() if bench_ret.std() > 0 else 0
    
    print(f"\nPerformance Summary:")
    print(f"  Strategy Total Return: {strat_total_return:.2f}%")
    print(f"  Strategy Annualized: {strat_annual:.2f}%")
    print(f"  Strategy Max Drawdown: {strat_mdd:.2f}%")
    print(f"  Strategy Sharpe: {strat_sharpe:.3f}")
    print(f"")
    print(f"  Benchmark Total Return: {bench_total_return:.2f}%")
    print(f"  Benchmark Annualized: {bench_annual:.2f}%")
    print(f"  Benchmark Max Drawdown: {bench_mdd:.2f}%")
    print(f"  Benchmark Sharpe: {bench_sharpe:.3f}")
    
    # ============================================
    # PLOT EQUITY CURVE
    # ============================================
    
    plt.figure(figsize=(14, 8))
    
    # Plot equity curves
    plt.subplot(2, 1, 1)
    plt.plot(strat_equity.index, strat_equity.values, label='DFG Strategy', color='#1f77b4', linewidth=1.5)
    plt.plot(bench_equity.index, bench_equity.values, label='Buy & Hold SPY', color='#ff7f0e', linewidth=1.5, alpha=0.7)
    plt.title('DFG Strategy vs Buy & Hold - Equity Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    plt.fill_between(strat_drawdown.index, strat_drawdown.values, 0, alpha=0.3, color='#1f77b4', label='Strategy')
    plt.fill_between(bench_drawdown.index, bench_drawdown.values, 0, alpha=0.3, color='#ff7f0e', label='Benchmark')
    plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/project/openhands/dfg_equity.png', dpi=150, bbox_inches='tight')
    print(f"\nEquity curve saved to dfg_equity.png")
    
    # Also save to dfg_equity.png as requested
    plt.savefig('/workspace/project/openhands/dfg_equity.png', dpi=150, bbox_inches='tight')
    
    return valid_data

if __name__ == "__main__":
    data = main()