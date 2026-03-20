"""
BBI (Bloodbath Bypass Indicator) Strategy
Based on the Vince/Williams "bloodbath sidestepping" rule from:
"The Ripple Effect of Daily New Lows" by Ralph Vince and Larry Williams

And variation from Mark Ungewitter's tweet:
https://x.com/mark_ungewitter/status/2030687520527130718

This version calculates NYSE New Lows from actual stock data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# REPRESENTATIVE NYSE STOCKS SAMPLE
# ============================================================

# A diversified sample of large-cap NYSE stocks to calculate new lows
NYSE_SAMPLE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH',
    'JNJ', 'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV',
    'MRK', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'WMT', 'MCD', 'CSCO', 'ACN',
    'ABT', 'DHR', 'NEE', 'ADBE', 'NKE', 'TXN', 'PM', 'MS', 'UPS', 'RTX',
    'LOW', 'HON', 'INTC', 'ORCL', 'IBM', 'SBUX', 'CAT', 'BA', 'GE', 'AMD',
    'QCOM', 'GS', 'BLK', 'C', 'DE', 'INTU', 'AMAT', 'MDLZ', 'GILD', 'ADP',
    'BKNG', 'ISRG', 'AXP', 'SYK', 'TJX', 'MMM', 'NOW', 'ZTS', 'REGN', 'VRTX',
    'ADI', 'LRCX', 'MU', 'MMC', 'CB', 'CI', 'CME', 'SPGI', 'MO', 'NOC',
    'DUK', 'PLD', 'SO', 'PNC', 'ETN', 'BSX', 'ITW', 'APD', 'ICE', 'EMR',
    'SHW', 'NSC', 'MCK', 'EOG', 'SLB', 'USB', 'TGT', 'MCO', 'CL', 'FIS'
]

# ============================================================
# DATA DOWNLOAD AND NYSE NEW LOWS CALCULATION
# ============================================================

def download_stock_data(tickers, start_date='1990-01-01', end_date=None):
    """Download adjusted close prices for a list of tickers."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading data for {len(tickers)} stocks...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=True)
    
    # Extract close prices
    if 'Close' in data.columns.get_level_values(0):
        close = data['Close']
    else:
        close = data
    
    return close

def calculate_nyse_new_lows_pct(price_data: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Calculate the percentage of stocks hitting new 52-week lows.
    
    For each day, calculate what % of stocks are at their 52-week low.
    """
    print("Calculating NYSE new lows percentage...")
    
    # Calculate rolling minimum (52-week low) for each stock
    rolling_min = price_data.rolling(window=lookback, min_periods=lookback//2).min()
    
    # Determine if each stock is at its 52-week low
    at_new_low = (price_data <= rolling_min)
    
    # Calculate percentage of stocks at new lows
    new_lows_pct = at_new_low.mean(axis=1) * 100
    
    return new_lows_pct

def get_market_data_with_new_lows(start_date='1995-01-01'):
    """
    Get SPY data and calculate NYSE new lows from stock sample.
    """
    # Download SPY for returns
    print("Downloading SPY data...")
    spy = yf.download('SPY', start=start_date, end='2024-12-31', progress=False)
    
    # Handle different yfinance return formats
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy['Close']['SPY']
    else:
        spy_close = spy['Close']
    
    # Ensure it's a series with proper index
    spy = pd.DataFrame({'close': spy_close})
    spy['returns'] = spy['close'].pct_change()
    
    # Download stock sample for new lows calculation
    stock_prices = download_stock_data(NYSE_SAMPLE, start_date=start_date)
    
    # Calculate new lows percentage
    nyse_new_lows = calculate_nyse_new_lows_pct(stock_prices)
    
    # Align with SPY dates
    data = spy.copy()
    data['new_lows_pct'] = nyse_new_lows
    data = data.dropna()
    
    print(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total trading days: {len(data)}")
    print(f"Max new lows %: {data['new_lows_pct'].max():.2f}%")
    print(f"Average new lows %: {data['new_lows_pct'].mean():.2f}%")
    
    return data

# ============================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================

def strategy_vince_williams(data: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """
    Original Vince/Williams Bloodbath Sidestepping Strategy.
    
    Logic:
    - Default position: LONG (1)
    - If NYSE new lows > threshold % of total issues, go FLAT (0)
    - Otherwise remain LONG
    
    Parameters:
    - threshold: Percentage threshold for new lows (default 4%)
    
    Returns:
    - DataFrame with signals and equity curve
    """
    df = data.copy()
    
    # Generate signal: 1 if new lows < threshold, 0 otherwise
    df['signal'] = (df['new_lows_pct'] < threshold).astype(int)
    
    # Shift signal to avoid look-ahead bias (trade on next day)
    df['signal'] = df['signal'].shift(1).fillna(1)
    
    # Calculate strategy returns
    df['strategy_returns'] = df['signal'] * df['returns']
    
    # Calculate cumulative equity
    df['strategy_equity'] = (1 + df['strategy_returns']).cumprod()
    df['buyhold_equity'] = (1 + df['returns']).cumprod()
    
    return df

def strategy_vince_williams_10day_avg(data: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """
    Variation based on Mark Ungewitter's tweet.
    
    Uses 10-day average of NYSE new lows to reduce false signals.
    
    Logic:
    - Default position: LONG (1)
    - If 10-day avg NYSE new lows > threshold % of total issues, go FLAT (0)
    - Otherwise remain LONG
    
    Parameters:
    - threshold: Percentage threshold for new lows (default 4%)
    - avg_days: Number of days for moving average (default 10)
    
    Returns:
    - DataFrame with signals and equity curve
    """
    df = data.copy()
    
    # Calculate 10-day moving average of new lows percentage
    df['new_lows_pct_10d_avg'] = df['new_lows_pct'].rolling(window=10, min_periods=1).mean()
    
    # Generate signal: 1 if 10-day avg new lows < threshold, 0 otherwise
    df['signal'] = (df['new_lows_pct_10d_avg'] < threshold).astype(int)
    
    # Shift signal to avoid look-ahead bias (trade on next day)
    df['signal'] = df['signal'].shift(1).fillna(1)
    
    # Calculate strategy returns
    df['strategy_returns'] = df['signal'] * df['returns']
    
    # Calculate cumulative equity
    df['strategy_equity'] = (1 + df['strategy_returns']).cumprod()
    df['buyhold_equity'] = (1 + df['returns']).cumprod()
    
    return df

# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_performance_metrics(df: pd.DataFrame, strategy_col: str = 'strategy_equity', 
                                  benchmark_col: str = 'buyhold_equity') -> dict:
    """
    Calculate annualized return and other performance metrics.
    """
    # Get clean data (no NaN)
    df_clean = df.dropna()
    
    # Calculate total return
    strategy_total_return = df_clean[strategy_col].iloc[-1] - 1
    benchmark_total_return = df_clean[benchmark_col].iloc[-1] - 1
    
    # Calculate years of data
    start_date = df_clean.index[0]
    end_date = df_clean.index[-1]
    years = (end_date - start_date).days / 365.25
    
    # Calculate annualized return
    strategy_annualized = (1 + strategy_total_return) ** (1/years) - 1
    benchmark_annualized = (1 + benchmark_total_return) ** (1/years) - 1
    
    # Calculate max drawdown
    strategy_peak = df_clean[strategy_col].cummax()
    strategy_drawdown = (df_clean[strategy_col] - strategy_peak) / strategy_peak
    max_strategy_drawdown = strategy_drawdown.min()
    
    benchmark_peak = df_clean[benchmark_col].cummax()
    benchmark_drawdown = (df_clean[benchmark_col] - benchmark_peak) / benchmark_peak
    max_benchmark_drawdown = benchmark_drawdown.min()
    
    # Calculate volatility (annualized)
    strategy_vol = df_clean['strategy_returns'].std() * np.sqrt(252)
    benchmark_vol = df_clean['returns'].std() * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    strategy_sharpe = strategy_annualized / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = benchmark_annualized / benchmark_vol if benchmark_vol > 0 else 0
    
    metrics = {
        'Strategy Total Return': f'{strategy_total_return*100:.2f}%',
        'Benchmark Total Return': f'{benchmark_total_return*100:.2f}%',
        'Strategy Annualized Return': f'{strategy_annualized*100:.2f}%',
        'Benchmark Annualized Return': f'{benchmark_annualized*100:.2f}%',
        'Strategy Max Drawdown': f'{max_strategy_drawdown*100:.2f}%',
        'Benchmark Max Drawdown': f'{max_benchmark_drawdown*100:.2f}%',
        'Strategy Volatility': f'{strategy_vol*100:.2f}%',
        'Benchmark Volatility': f'{benchmark_vol*100:.2f}%',
        'Strategy Sharpe Ratio': f'{strategy_sharpe:.2f}',
        'Benchmark Sharpe Ratio': f'{benchmark_sharpe:.2f}',
        'Years': f'{years:.1f}'
    }
    
    return metrics

# ============================================================
# PLOTTING
# ============================================================

def plot_performance(df1: pd.DataFrame, df2: pd.DataFrame, 
                     strategy1_name: str = 'Vince/Williams (4% Threshold)',
                     strategy2_name: str = '10-Day Avg (4% Threshold)',
                     save_path: str = 'bbi_performance.png'):
    """
    Plot equity curves for both strategies vs Buy & Hold.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Both strategies vs Buy & Hold
    ax1 = axes[0]
    ax1.plot(df1.index, df1['buyhold_equity'], label='Buy & Hold SPY', 
             color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(df1.index, df1['strategy_equity'], label=strategy1_name, 
             color='blue', linewidth=1.5)
    ax1.plot(df2.index, df2['strategy_equity'], label=strategy2_name, 
             color='red', linewidth=1.5, linestyle='--')
    
    ax1.set_title('BBI Strategy Performance vs Buy & Hold SPY', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Equity')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylabel('Cumulative Equity (Log Scale)')
    
    # Plot 2: Drawdown comparison
    ax2 = axes[1]
    
    strat1_dd = (df1['strategy_equity'] - df1['strategy_equity'].cummax()) / df1['strategy_equity'].cummax()
    strat2_dd = (df2['strategy_equity'] - df2['strategy_equity'].cummax()) / df2['strategy_equity'].cummax()
    bench_dd = (df1['buyhold_equity'] - df1['buyhold_equity'].cummax()) / df1['buyhold_equity'].cummax()
    
    ax2.fill_between(df1.index, bench_dd, 0, alpha=0.3, color='black', label='Buy & Hold SPY')
    ax2.fill_between(df1.index, strat1_dd, 0, alpha=0.3, color='blue', label=strategy1_name)
    ax2.fill_between(df2.index, strat2_dd, 0, alpha=0.3, color='red', label=strategy2_name)
    
    ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to {save_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("BBI (Bloodbath Bypass Indicator) Strategy Analysis")
    print("Using actual NYSE New Lows from stock sample")
    print("=" * 60)
    
    # Get market data with calculated new lows
    print("\n" + "=" * 60)
    print("Downloading and processing market data...")
    print("=" * 60)
    data = get_market_data_with_new_lows(start_date='1995-01-01')
    
    # Run Strategy 1: Original Vince/Williams
    print("\n" + "=" * 60)
    print("Strategy 1: Vince/Williams (4% Threshold)")
    print("=" * 60)
    df1 = strategy_vince_williams(data, threshold=4.0)
    metrics1 = calculate_performance_metrics(df1)
    
    print("\nPerformance Metrics:")
    for key, value in metrics1.items():
        print(f"  {key}: {value}")
    
    # Show signal statistics
    print(f"\nTime in market: {(df1['signal'] == 1).sum() / len(df1) * 100:.1f}%")
    print(f"Time flat: {(df1['signal'] == 0).sum() / len(df1) * 100:.1f}%")
    
    # Run Strategy 2: 10-Day Average Variation
    print("\n" + "=" * 60)
    print("Strategy 2: 10-Day Average (4% Threshold)")
    print("=" * 60)
    df2 = strategy_vince_williams_10day_avg(data, threshold=4.0)
    metrics2 = calculate_performance_metrics(df2)
    
    print("\nPerformance Metrics:")
    for key, value in metrics2.items():
        print(f"  {key}: {value}")
    
    print(f"\nTime in market: {(df2['signal'] == 1).sum() / len(df2) * 100:.1f}%")
    print(f"Time flat: {(df2['signal'] == 0).sum() / len(df2) * 100:.1f}%")
    
    # Benchmark comparison
    print("\n" + "=" * 60)
    print("Benchmark Comparison Summary")
    print("=" * 60)
    print(f"{'Strategy':<35} {'Annualized Return':<20}")
    print("-" * 55)
    print(f"{'Buy & Hold SPY':<35} {metrics1['Benchmark Annualized Return']}")
    print(f"{'Vince/Williams (4% Threshold)':<35} {metrics1['Strategy Annualized Return']}")
    print(f"{'10-Day Avg (4% Threshold)':<35} {metrics2['Strategy Annualized Return']}")
    
    # Generate plot
    print("\nGenerating performance plot...")
    plot_performance(df1, df2, save_path='bbi_performance.png')
    
    # Return dataframes for further analysis
    return df1, df2, metrics1, metrics2

if __name__ == "__main__":
    df1, df2, metrics1, metrics2 = main()
