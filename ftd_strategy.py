"""
FTD (Follow-Through Day) Trading Strategy

This strategy identifies oversold conditions using RSI, then identifies subsequent
FTD using high IBS (Intra-Bar Strength) value and high daily return (>1%).

5 Entry Variations:
1. RSI < 30 + FTD (IBS > 0.5 AND daily return > 1%)
2. RSI < 35 + FTD (IBS > 0.7 AND daily return > 1.5%)
3. RSI < 25 + FTD (IBS > 0.5 AND daily return > 1%)
4. RSI crosses above 30 + FTD (RSI reversal)
5. RSI < 40 + FTD (IBS > 0.5, any positive return)

2 Exit Variations:
1. Time-based: Hold for N bars (e.g., 5 bars)
2. RSI threshold: Exit when RSI > 65

Total: 10 strategy variations (5 entries x 2 exits)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
TICKER = 'SPY'
START_DATE = '2000-01-01'
END_DATE = '2024-12-31'


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ibs(df):
    """Calculate Intra-Bar Strength: (close - low) / (high - low)"""
    ibs = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    return ibs


def calculate_daily_return(df):
    """Calculate daily return"""
    return df['Close'].pct_change()


def download_spy_data():
    """Download SPY data from Yahoo Finance"""
    print(f"Downloading {TICKER} data...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    df = df.reset_index()
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' or col[1] == TICKER else col[0] for col in df.columns]
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['IBS'] = calculate_ibs(df)
    df['Daily_Return'] = calculate_daily_return(df)
    
    # Calculate prior RSI (for crossover detection)
    df['Prior_RSI'] = df['RSI'].shift(1)
    
    print(f"Downloaded {len(df)} trading days of data")
    return df


# Entry condition functions
def entry_condition_1(df):
    """Entry 1: RSI < 30 (oversold) + FTD (IBS > 0.5 AND daily return > 1%)"""
    oversold = df['RSI'] < 30
    ftd = (df['IBS'] > 0.5) & (df['Daily_Return'] > 0.01)
    return oversold & ftd


def entry_condition_2(df):
    """Entry 2: RSI < 35 + FTD (IBS > 0.7 AND daily return > 1.5%)"""
    oversold = df['RSI'] < 35
    ftd = (df['IBS'] > 0.7) & (df['Daily_Return'] > 0.015)
    return oversold & ftd


def entry_condition_3(df):
    """Entry 3: RSI < 25 (more oversold) + FTD (IBS > 0.5 AND daily return > 1%)"""
    oversold = df['RSI'] < 25
    ftd = (df['IBS'] > 0.5) & (df['Daily_Return'] > 0.01)
    return oversold & ftd


def entry_condition_4(df):
    """Entry 4: RSI crosses above 30 (RSI reversal) + FTD"""
    rsi_cross = (df['Prior_RSI'] <= 30) & (df['RSI'] > 30)
    ftd = (df['IBS'] > 0.5) & (df['Daily_Return'] > 0.01)
    return rsi_cross & ftd


def entry_condition_5(df):
    """Entry 5: RSI < 40 + FTD (IBS > 0.5, any positive return)"""
    oversold = df['RSI'] < 40
    ftd = (df['IBS'] > 0.5) & (df['Daily_Return'] > 0)
    return oversold & ftd


# Exit condition functions
def exit_condition_time(df, hold_bars=5):
    """Exit: Hold for N bars"""
    # This will be handled in the backtest loop
    return hold_bars


def exit_condition_rsi(df, threshold=65):
    """Exit: RSI reaches threshold"""
    return df['RSI'] > threshold


# Entry condition mapping
ENTRY_CONDITIONS = {
    'Entry_1': entry_condition_1,
    'Entry_2': entry_condition_2,
    'Entry_3': entry_condition_3,
    'Entry_4': entry_condition_4,
    'Entry_5': entry_condition_5,
}


def run_backtest(df, entry_name, exit_name):
    """
    Run backtest for a specific entry/exit combination.
    
    Returns:
        df_trades: DataFrame with trade signals
        equity_curve: Series with cumulative equity
    """
    entry_func = ENTRY_CONDITIONS[entry_name]
    
    # Get entry signals
    entry_signal = entry_func(df).astype(int)
    
    # Initialize position (0 = flat, 1 = long)
    position = pd.Series(0, index=df.index)
    
    # Handle exit conditions
    hold_bars = 5  # Default for time-based exit
    rsi_threshold = 65  # Default for RSI threshold exit
    
    in_position = False
    entry_bar = 0
    bars_held = 0
    
    for i in range(len(df)):
        if not in_position:
            # Try to enter on entry signal
            if entry_signal.iloc[i] == 1:
                position.iloc[i] = 1
                in_position = True
                entry_bar = i
                bars_held = 1
        else:
            # Currently in position - check exit conditions
            should_exit = False
            
            if exit_name == 'Exit_Time':
                # Time-based exit: hold for N bars
                if bars_held >= hold_bars:
                    should_exit = True
            elif exit_name == 'Exit_RSI':
                # RSI threshold exit
                if df['RSI'].iloc[i] > rsi_threshold:
                    should_exit = True
            
            if should_exit:
                position.iloc[i] = 0
                in_position = False
                bars_held = 0
                # Check if we can re-enter on same bar
                if entry_signal.iloc[i] == 1:
                    position.iloc[i] = 1
                    in_position = True
                    entry_bar = i
                    bars_held = 1
            else:
                position.iloc[i] = 1
                bars_held += 1
    
    # Calculate returns
    df['Position'] = position
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
    df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
    
    # Cumulative returns (equity curve)
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_BuyAndHold'] = (1 + df['Daily_Return']).cumprod()
    
    # Extract trades
    trades = []
    in_trade = False
    for i in range(len(df)):
        if df['Position'].iloc[i] == 1 and not in_trade:
            # Entry
            trades.append({
                'Entry_Date': df['Date'].iloc[i],
                'Entry_Price': df['Close'].iloc[i],
                'Entry_RSI': df['RSI'].iloc[i],
                'Entry_IBS': df['IBS'].iloc[i],
            })
            in_trade = True
        elif df['Position'].iloc[i] == 0 and in_trade:
            # Exit
            trades[-1]['Exit_Date'] = df['Date'].iloc[i]
            trades[-1]['Exit_Price'] = df['Close'].iloc[i]
            trades[-1]['Exit_RSI'] = df['RSI'].iloc[i]
            trades[-1]['Return'] = (trades[-1]['Exit_Price'] / trades[-1]['Entry_Price']) - 1
            in_trade = False
    
    # Close any open position at the end
    if in_trade and len(trades) > 0:
        trades[-1]['Exit_Date'] = df['Date'].iloc[-1]
        trades[-1]['Exit_Price'] = df['Close'].iloc[-1]
        trades[-1]['Exit_RSI'] = df['RSI'].iloc[-1]
        trades[-1]['Return'] = (trades[-1]['Exit_Price'] / trades[-1]['Entry_Price']) - 1
    
    df_trades = pd.DataFrame(trades)
    
    return df, df_trades


def calculate_metrics(df_trades, df_full):
    """Calculate performance metrics"""
    if len(df_trades) == 0:
        return {
            'Num_Trades': 0,
            'Total_Return': 0,
            'Profit_Factor': 0,
            'Win_Rate': 0,
        }
    
    # Calculate returns
    trades_returns = df_trades['Return'].dropna()
    
    if len(trades_returns) == 0:
        return {
            'Num_Trades': 0,
            'Total_Return': 0,
            'Profit_Factor': 0,
            'Win_Rate': 0,
        }
    
    wins = trades_returns[trades_returns > 0]
    losses = trades_returns[trades_returns < 0]
    
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
    
    win_rate = len(wins) / len(trades_returns) * 100 if len(trades_returns) > 0 else 0
    
    # Total return from equity curve
    final_return = df_full['Cumulative_Strategy_Return'].iloc[-1] - 1
    
    return {
        'Num_Trades': len(trades_returns),
        'Total_Return': final_return * 100,  # As percentage
        'Profit_Factor': profit_factor,
        'Win_Rate': win_rate,
    }


def run_all_strategies(df):
    """Run all 10 strategy combinations"""
    results = []
    equity_curves = {}
    
    exit_names = ['Exit_Time', 'Exit_RSI']
    entry_names = ['Entry_1', 'Entry_2', 'Entry_3', 'Entry_4', 'Entry_5']
    
    for entry_name in entry_names:
        for exit_name in exit_names:
            strategy_name = f"{entry_name}_{exit_name}"
            print(f"Running {strategy_name}...")
            
            df_result, df_trades = run_backtest(df, entry_name, exit_name)
            metrics = calculate_metrics(df_trades, df_result)
            metrics['Strategy'] = strategy_name
            
            results.append(metrics)
            equity_curves[strategy_name] = df_result['Cumulative_Strategy_Return']
            
            print(f"  Trades: {metrics['Num_Trades']}, Return: {metrics['Total_Return']:.2f}%, PF: {metrics['Profit_Factor']:.2f}")
    
    return results, equity_curves


def create_performance_table(results):
    """Create a formatted performance table"""
    df_results = pd.DataFrame(results)
    df_results = df_results[['Strategy', 'Num_Trades', 'Total_Return', 'Profit_Factor', 'Win_Rate']]
    df_results = df_results.sort_values('Total_Return', ascending=False)
    
    # Format columns
    df_results['Total_Return'] = df_results['Total_Return'].apply(lambda x: f"{x:.2f}%")
    df_results['Profit_Factor'] = df_results['Profit_Factor'].apply(lambda x: f"{x:.2f}")
    df_results['Win_Rate'] = df_results['Win_Rate'].apply(lambda x: f"{x:.1f}%")
    
    return df_results


def plot_performance(equity_curves, save_path='ftd_performance.png'):
    """Create performance plot with log scale"""
    plt.figure(figsize=(14, 8))
    
    # Use log scale for y-axis
    plt.yscale('log')
    
    # Plot each strategy
    for name, curve in equity_curves.items():
        plt.plot(curve, label=name, linewidth=1.5)
    
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Cumulative Return (Log Scale)', fontsize=12)
    plt.title('FTD Strategy Performance - All 10 Variations', fontsize=14)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to {save_path}")


def main():
    print("=" * 60)
    print("FTD (Follow-Through Day) Trading Strategy Analysis")
    print("=" * 60)
    
    # Download data
    df = download_spy_data()
    
    # Run all strategies
    results, equity_curves = run_all_strategies(df)
    
    # Create and display performance table
    df_results = create_performance_table(results)
    print("\n" + "=" * 60)
    print("Performance Table (sorted by Total Return)")
    print("=" * 60)
    print(df_results.to_string(index=False))
    
    # Save performance table to CSV
    df_results.to_csv('ftd_performance_table.csv', index=False)
    print("\nPerformance table saved to ftd_performance_table.csv")
    
    # Create performance plot
    plot_performance(equity_curves, 'ftd_performance.png')
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return df_results, equity_curves


if __name__ == "__main__":
    df_results, equity_curves = main()