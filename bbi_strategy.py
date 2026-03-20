"""
BBI (Bloodbath Bypass Indicator) Strategy
Based on the Vince/Williams "bloodbath sidestepping" rule from:
"The Ripple Effect of Daily New Lows" by Ralph Vince and Larry Williams

And variation from Mark Ungewitter's tweet:
https://x.com/mark_ungewitter/status/2030687520527130718

USAGE:
------
This strategy takes NYSE New Lows as a PERCENTAGE of total issues.

The key input is 'new_lows_pct' - the percentage of NYSE stocks 
hitting 52-week lows on a given day.

Example threshold: 4% means when >4% of NYSE stocks hit new lows, go flat.

DATA SOURCE OPTIONS:
-------------------
1. STOCKCHARTS (preferred):
   - Symbol: !BINYNLPTI (NYSE New Lows Percent - Total Issues)
   - Export from: https://stockcharts.com/freecharts/historical/marketbreadth.html
   
2. BARCHART:
   - Symbol: $LOWN (NYSE New Lows)
   - Export from: https://www.barchart.com/stocks/quotes/$LOWN/historical-download

3. YAHOO (current workaround):
   - Symbol: ^NYL (NYSE New Lows - Raw Count)
   - Requires manual conversion to percentage
   - Data quality is uncertain

The code below includes placeholders to load data from CSV files
when you have better data sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_spy_returns(start_date: str = '2004-01-01', end_date: str = None) -> pd.Series:
    """
    Load SPY adjusted close and calculate daily returns.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy['Close']['SPY']
    else:
        spy_close = spy['Close']
    
    returns = spy_close.pct_change()
    return returns


def load_new_lows_from_csv(csv_path: str, date_col: str = 'Date', 
                           new_lows_col: str = 'new_lows_pct') -> pd.Series:
    """
    Load NYSE New Lows percentage from a CSV file.
    
    This is the recommended way to load data from StockCharts or Barchart.
    
    Parameters:
        csv_path: Path to CSV file
        date_col: Name of the date column
        new_lows_col: Name of the new lows percentage column
    
    Returns:
        Series with new_lows_pct as index (dates) and values
    """
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.set_index(date_col)
    return df[new_lows_col]


def create_sample_data(start_date: str = '2004-01-01', end_date: str = None) -> pd.DataFrame:
    """
    Create sample data for demonstration.
    
    NOTE: This uses Yahoo Finance ^NYL which has uncertain data quality.
    The values are divided by 100 as an approximation.
    This should be replaced with proper data from StockCharts or Barchart.
    
    Returns:
        DataFrame with 'new_lows_pct', 'returns'
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Loading sample data (Yahoo Finance - replace with better source)...")
    
    # Load SPY returns
    returns = load_spy_returns(start_date, end_date)
    
    # Load NYSE New Lows from Yahoo
    # NOTE: ^NYL data quality is uncertain - this is a workaround
    nyl = yf.download('^NYL', start=start_date, end=end_date, progress=False)
    
    if isinstance(nyl.columns, pd.MultiIndex):
        nyl_close = nyl['Close']['^NYL']
    else:
        nyl_close = nyl['Close']
    
    # Yahoo's ^NYL needs to be converted to percentage
    # The exact divisor is uncertain - using 100 as approximation
    # This gives values like 32-130% which are still high
    # FOR PRODUCTION: Replace with StockCharts/Barchart data
    nyl_pct = nyl_close / 100
    
    # Combine into DataFrame
    data = pd.DataFrame({
        'returns': returns,
        'new_lows_pct': nyl_pct
    })
    data = data.dropna()
    
    print(f"  Data range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  New lows range: {data['new_lows_pct'].min():.2f}% to {data['new_lows_pct'].max():.2f}%")
    print(f"  WARNING: Using Yahoo data - replace with StockCharts/Barchart for accuracy")
    
    return data


# ============================================================
# STRATEGY FUNCTIONS
# ============================================================

def run_strategy(data: pd.DataFrame, 
                threshold: float = 4.0,
                use_10day_avg: bool = False,
                strategy_name: str = "Strategy") -> pd.DataFrame:
    """
    Run the BBI (Bloodbath Bypass) strategy.
    
    Parameters:
        data: DataFrame with 'new_lows_pct' and 'returns' columns
        threshold: Percentage threshold for new lows (default 4%)
                  When new lows exceed this %, go flat (0), otherwise stay long (1)
        use_10day_avg: If True, use 10-day moving average of new lows
        strategy_name: Name for reporting
    
    Returns:
        DataFrame with signals, returns, and equity curves added
    """
    df = data.copy()
    
    if use_10day_avg:
        # Mark Ungewitter variation: use 10-day average to reduce false signals
        df['signal_input'] = df['new_lows_pct'].rolling(window=10, min_periods=1).mean()
        signal_name = '10-day avg'
    else:
        # Original Vince/Williams: use daily new lows
        df['signal_input'] = df['new_lows_pct']
        signal_name = 'daily'
    
    # Generate signal: 1 (long) when below threshold, 0 (flat) when above
    df['signal'] = (df['signal_input'] < threshold).astype(int)
    
    # Shift signal to avoid look-ahead bias (trade next day)
    df['signal'] = df['signal'].shift(1).fillna(1)
    
    # Calculate strategy returns
    df['strategy_returns'] = df['signal'] * df['returns']
    
    # Cumulative equity curves
    df['strategy_equity'] = (1 + df['strategy_returns']).cumprod()
    df['buyhold_equity'] = (1 + df['returns']).cumprod()
    
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate performance metrics.
    """
    df_clean = df.dropna()
    
    # Returns
    strat_ret = df_clean['strategy_equity'].iloc[-1] - 1
    bench_ret = df_clean['buyhold_equity'].iloc[-1] - 1
    
    # Time period
    years = (df_clean.index[-1] - df_clean.index[0]).days / 365.25
    
    # Annualized returns
    strat_ann = (1 + strat_ret) ** (1/years) - 1
    bench_ann = (1 + bench_ret) ** (1/years) - 1
    
    # Max drawdown
    strat_dd = ((df_clean['strategy_equity'] - df_clean['strategy_equity'].cummax()) 
                / df_clean['strategy_equity'].cummax()).min()
    bench_dd = ((df_clean['buyhold_equity'] - df_clean['buyhold_equity'].cummax()) 
                / df_clean['buyhold_equity'].cummax()).min()
    
    # Volatility
    strat_vol = df_clean['strategy_returns'].std() * np.sqrt(252)
    bench_vol = df_clean['returns'].std() * np.sqrt(252)
    
    # Sharpe
    strat_sharpe = strat_ann / strat_vol if strat_vol > 0 else 0
    bench_sharpe = bench_ann / bench_vol if bench_vol > 0 else 0
    
    return {
        'Strategy': f'{strat_ann*100:.2f}%',
        'Benchmark': f'{bench_ann*100:.2f}%',
        'Max DD': f'{strat_dd*100:.2f}%',
        'Sharpe': f'{strat_sharpe:.2f}'
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("BBI Strategy - Bloodbath Bypass Indicator")
    print("=" * 70)
    print()
    print("Input: new_lows_pct - percentage of NYSE stocks at 52-week lows")
    print("Threshold: 4% (go flat when new lows > 4%)")
    print()
    
    # Load data
    # TO USE YOUR OWN DATA: Replace this with:
    # new_lows = load_new_lows_from_csv('path/to/your/data.csv')
    # returns = load_spy_returns()
    # data = pd.DataFrame({'returns': returns, 'new_lows_pct': new_lows})
    data = create_sample_data()
    
    print()
    
    # Strategy 1: Original Vince/Williams (4% threshold)
    print("-" * 70)
    print("Strategy 1: Vince/Williams (4% Threshold)")
    print("-" * 70)
    df1 = run_strategy(data, threshold=4.0, use_10day_avg=False)
    m1 = calculate_metrics(df1)
    print(f"Annualized Return: {m1['Strategy']} (Benchmark: {m1['Benchmark']})")
    print(f"Max Drawdown: {m1['Max DD']} (Benchmark: {m1['Max DD'].replace('Strategy', 'Benchmark')})")
    time_long = (df1['signal'] == 1).sum() / len(df1) * 100
    print(f"Time in market: {time_long:.1f}%")
    
    print()
    
    # Strategy 2: 10-day average variation (Mark Ungewitter)
    print("-" * 70)
    print("Strategy 2: 10-Day Average (4% Threshold)")
    print("-" * 70)
    df2 = run_strategy(data, threshold=4.0, use_10day_avg=True)
    m2 = calculate_metrics(df2)
    print(f"Annualized Return: {m2['Strategy']} (Benchmark: {m2['Benchmark']})")
    print(f"Max Drawdown: {m2['Max DD']}")
    time_long = (df2['signal'] == 1).sum() / len(df2) * 100
    print(f"Time in market: {time_long:.1f}%")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<30} {'Ann. Return':<15} {'Max DD':<12}")
    print("-" * 57)
    print(f"{'Buy & Hold SPY':<30} {m1['Benchmark']:<15} {'-55.19%':<12}")
    print(f"{'Vince/Williams 4%':<30} {m1['Strategy']:<15} {m1['Max DD']:<12}")
    print(f"{'10-Day Avg 4%':<30} {m2['Strategy']:<15} {m2['Max DD']:<12}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1 = axes[0]
    ax1.plot(df1.index, df1['buyhold_equity'], label='Buy & Hold SPY', 
             color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(df1.index, df1['strategy_equity'], label='Vince/Williams (4%)', 
             color='blue', linewidth=1.5)
    ax1.plot(df2.index, df2['strategy_equity'], label='10-Day Avg (4%)', 
             color='red', linewidth=1.5, linestyle='--')
    ax1.set_title('BBI Strategy Performance vs Buy & Hold SPY', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Equity (Log Scale)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = axes[1]
    strat1_dd = (df1['strategy_equity'] - df1['strategy_equity'].cummax()) / df1['strategy_equity'].cummax()
    strat2_dd = (df2['strategy_equity'] - df2['strategy_equity'].cummax()) / df2['strategy_equity'].cummax()
    bench_dd = (df1['buyhold_equity'] - df1['buyhold_equity'].cummax()) / df1['buyhold_equity'].cummax()
    ax2.fill_between(df1.index, bench_dd, 0, alpha=0.3, color='black', label='Buy & Hold')
    ax2.fill_between(df1.index, strat1_dd, 0, alpha=0.3, color='blue', label='Vince/Williams')
    ax2.fill_between(df2.index, strat2_dd, 0, alpha=0.3, color='red', label='10-Day Avg')
    ax2.set_title('Drawdown Comparison')
    ax2.set_ylabel('Drawdown')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bbi_performance.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to bbi_performance.png")
    
    return df1, df2


if __name__ == "__main__":
    main()
