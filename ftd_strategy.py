"""
FTD (Follow-Through Day) Trading Strategy

This strategy identifies oversold conditions using RSI, then identifies subsequent
FTD using high IBS (Intra-Bar Strength) value and high daily return (>1%).

Original 5 Entry Variations:
1. RSI < 30 + FTD (IBS > 0.5 AND daily return > 1%)
2. RSI < 35 + FTD (IBS > 0.7 AND daily return > 1.5%)
3. RSI < 25 + FTD (IBS > 0.5 AND daily return > 1%)
4. RSI crosses above 30 + FTD (RSI reversal)
5. RSI < 40 + FTD (IBS > 0.5, any positive return)

2 Exit Variations:
1. Time-based: Hold for N bars (e.g., 5 bars)
2. RSI threshold: Exit when RSI > 65

Additional 10 Social Media Strategies (paired with FTD concept):
6. MA Crossover: Price crosses above 50-day MA after oversold RSI
7. Volume Spike: Volume > 1.5x average after RSI oversold
8. Gap Up: Gap up > 0.5% after RSI oversold
9. Support Bounce: Price bounces from daily VWAP after oversold
10. Trend Confirmation: Price above 200-day MA after RSI oversold
11. Momentum: RSI crosses above 40 with strongIBS (>0.7)
12. Pullback: Price retraces to 20-day MA after FTD
13. Swing High Break: Break of prior swing high after oversold
14. VWAP Support: Price bounces from VWAP after oversold
15. Wide Bar: Large green bar (>1.5%) after RSI oversold

Total: 20 strategy variations
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


def calculate_indicators(df):
    """Calculate all additional technical indicators"""
    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # VWAP (approximate using typical price - cumulative calculation)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Cum_Volume'] = df['Volume'].cumsum()
    df['Cum_TP_Volume'] = (df['Typical_Price'] * df['Volume']).cumsum()
    df['VWAP'] = df['Cum_TP_Volume'] / df['Cum_Volume']
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Gap calculation
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Prior day data for swing high/low
    df['Prior_High'] = df['High'].shift(1)
    df['Prior_Low'] = df['Low'].shift(1)
    
    # Swing high (last 5 days)
    df['Swing_High'] = df['High'].rolling(window=5).max().shift(1)
    
    # Prior RSI
    df['Prior_RSI'] = df['RSI'].shift(1)
    
    return df


def download_spy_data():
    """Download SPY data from Yahoo Finance"""
    print(f"Downloading {TICKER} data...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    df = df.reset_index()
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' or col[1] == TICKER else col[0] for col in df.columns]
    
    # Calculate base indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['IBS'] = calculate_ibs(df)
    df['Daily_Return'] = calculate_daily_return(df)
    
    # Calculate additional indicators
    df = calculate_indicators(df)
    
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


# Additional Entry conditions (6-15) - Social Media strategies
def entry_condition_6(df):
    """Entry 6: MA Crossover - Price crosses above 50-day MA after oversold RSI"""
    rsi_oversold = df['RSI'] < 30
    ma_cross = df['Close'] > df['MA_50']
    prior_below = df['Close'].shift(1) <= df['MA_50'].shift(1)
    return rsi_oversold & ma_cross & prior_below


def entry_condition_7(df):
    """Entry 7: Volume Spike - Volume > 1.5x average after RSI oversold"""
    rsi_oversold = df['RSI'] < 30
    volume_spike = df['Volume_Ratio'] > 1.5
    return rsi_oversold & volume_spike


def entry_condition_8(df):
    """Entry 8: Gap Up - Gap up > 0.5% after RSI oversold"""
    rsi_oversold = df['RSI'] < 30
    gap_up = df['Gap'] > 0.005
    return rsi_oversold & gap_up


def entry_condition_9(df):
    """Entry 9: Support Bounce - Price bounces from daily VWAP after oversold"""
    rsi_oversold = df['RSI'] < 30
    # Close is above VWAP today but was below yesterday
    above_vwap = df['Close'] > df['VWAP']
    below_yesterday = df['Close'].shift(1) <= df['VWAP'].shift(1)
    return rsi_oversold & above_vwap & below_yesterday


def entry_condition_10(df):
    """Entry 10: Trend Confirmation - Price above 200-day MA after RSI oversold"""
    rsi_oversold = df['RSI'] < 30
    above_ma200 = df['Close'] > df['MA_200']
    return rsi_oversold & above_ma200


def entry_condition_11(df):
    """Entry 11: Momentum - RSI crosses above 40 with strong IBS (>0.7)"""
    rsi_cross = (df['Prior_RSI'] <= 40) & (df['RSI'] > 40)
    strong_ibs = df['IBS'] > 0.7
    return rsi_cross & strong_ibs


def entry_condition_12(df):
    """Entry 12: Pullback - Price retraces to 20-day MA after FTD"""
    ftd = (df['IBS'] > 0.5) & (df['Daily_Return'] > 0.01)
    # Price is at or near 20-day MA (within 1%)
    at_ma20 = abs(df['Close'] - df['MA_20']) / df['MA_20'] < 0.01
    return ftd & at_ma20


def entry_condition_13(df):
    """Entry 13: Swing High Break - Break of prior swing high after oversold"""
    rsi_oversold = df['RSI'] < 30
    break_swing = df['Close'] > df['Swing_High']
    return rsi_oversold & break_swing


def entry_condition_14(df):
    """Entry 14: VWAP Support - Price bounces from VWAP after oversold"""
    rsi_oversold = df['RSI'] < 30
    # Price is above VWAP today but was below yesterday (bounce)
    bounce = (df['Close'] > df['VWAP']) & (df['Close'].shift(1) <= df['VWAP'].shift(1))
    return rsi_oversold & bounce


def entry_condition_15(df):
    """Entry 15: Wide Bar - Large green bar (>1.5%) after RSI oversold"""
    rsi_oversold = df['RSI'] < 30
    wide_bar = df['Daily_Return'] > 0.015
    return rsi_oversold & wide_bar


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
    'Entry_6': entry_condition_6,
    'Entry_7': entry_condition_7,
    'Entry_8': entry_condition_8,
    'Entry_9': entry_condition_9,
    'Entry_10': entry_condition_10,
    'Entry_11': entry_condition_11,
    'Entry_12': entry_condition_12,
    'Entry_13': entry_condition_13,
    'Entry_14': entry_condition_14,
    'Entry_15': entry_condition_15,
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
    """Run all 20 strategy combinations (15 entries x 2 exits)"""
    results = []
    equity_curves = {}
    dates = df['Date']  # Store dates for plotting
    
    exit_names = ['Exit_Time', 'Exit_RSI']
    # First 5 are original FTD strategies, 6-15 are social media strategies
    entry_names = [f'Entry_{i}' for i in range(1, 16)]
    
    for entry_name in entry_names:
        for exit_name in exit_names:
            strategy_name = f"{entry_name}_{exit_name}"
            print(f"Running {strategy_name}...")
            
            df_result, df_trades = run_backtest(df, entry_name, exit_name)
            metrics = calculate_metrics(df_trades, df_result)
            metrics['Strategy'] = strategy_name
            
            results.append(metrics)
            # Store equity curve with dates
            equity_curves[strategy_name] = pd.Series(
                df_result['Cumulative_Strategy_Return'].values, 
                index=dates
            )
            
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
    """Create performance plot with log scale and date x-axis"""
    plt.figure(figsize=(14, 8))
    
    # Use log scale for y-axis
    plt.yscale('log')
    
    # Plot each strategy with dates on x-axis
    for name, curve in equity_curves.items():
        plt.plot(curve.index, curve.values, label=name, linewidth=1.5)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Log Scale)', fontsize=12)
    plt.title('FTD Strategy Performance - All 20 Variations', fontsize=14)
    plt.legend(loc='upper left', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to {save_path}")


def create_rules_table():
    """Create a table of all strategy rules"""
    rules = [
        # Original FTD strategies (1-5)
        {'Strategy': 'Entry_1', 'Description': 'RSI < 30 (oversold) + FTD (IBS > 0.5 AND daily return > 1%)', 'Category': 'Original FTD'},
        {'Strategy': 'Entry_2', 'Description': 'RSI < 35 + FTD (IBS > 0.7 AND daily return > 1.5%)', 'Category': 'Original FTD'},
        {'Strategy': 'Entry_3', 'Description': 'RSI < 25 (more oversold) + FTD (IBS > 0.5 AND daily return > 1%)', 'Category': 'Original FTD'},
        {'Strategy': 'Entry_4', 'Description': 'RSI crosses above 30 (RSI reversal) + FTD', 'Category': 'Original FTD'},
        {'Strategy': 'Entry_5', 'Description': 'RSI < 40 + FTD (IBS > 0.5, any positive return)', 'Category': 'Original FTD'},
        # Social media strategies (6-15)
        {'Strategy': 'Entry_6', 'Description': 'MA Crossover - Price crosses above 50-day MA after oversold RSI', 'Category': 'Social Media'},
        {'Strategy': 'Entry_7', 'Description': 'Volume Spike - Volume > 1.5x average after RSI oversold', 'Category': 'Social Media'},
        {'Strategy': 'Entry_8', 'Description': 'Gap Up - Gap up > 0.5% after RSI oversold', 'Category': 'Social Media'},
        {'Strategy': 'Entry_9', 'Description': 'Support Bounce - Price bounces from daily VWAP after oversold', 'Category': 'Social Media'},
        {'Strategy': 'Entry_10', 'Description': 'Trend Confirmation - Price above 200-day MA after RSI oversold', 'Category': 'Social Media'},
        {'Strategy': 'Entry_11', 'Description': 'Momentum - RSI crosses above 40 with strong IBS (>0.7)', 'Category': 'Social Media'},
        {'Strategy': 'Entry_12', 'Description': 'Pullback - Price retraces to 20-day MA after FTD', 'Category': 'Social Media'},
        {'Strategy': 'Entry_13', 'Description': 'Swing High Break - Break of prior swing high after oversold', 'Category': 'Social Media'},
        {'Strategy': 'Entry_14', 'Description': 'VWAP Support - Price bounces from VWAP after oversold', 'Category': 'Social Media'},
        {'Strategy': 'Entry_15', 'Description': 'Wide Bar - Large green bar (>1.5%) after RSI oversold', 'Category': 'Social Media'},
    ]
    
    # Add exit rules
    exit_rules = [
        {'Strategy': 'Exit_Time', 'Description': 'Hold for 5 bars then exit', 'Category': 'Exit'},
        {'Strategy': 'Exit_RSI', 'Description': 'Exit when RSI > 65', 'Category': 'Exit'},
    ]
    
    return pd.DataFrame(rules), pd.DataFrame(exit_rules)


def save_to_excel(df_results, df_rules, exit_rules, filename='ftd_table.xlsx'):
    """Save performance and rules to Excel with formatting"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Write performance sheet
        df_results.to_excel(writer, sheet_name='Perf', index=False)
        
        # Write rules sheet
        df_rules.to_excel(writer, sheet_name='Rules', index=False)
        
        # Get workbook and worksheets
        workbook = writer.book
        perf_sheet = writer.sheets['Perf']
        rules_sheet = writer.sheets['Rules']
        
        # Format Performance sheet
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        
        # Header styling
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_alignment = Alignment(horizontal='center', vertical='center')
        
        for cell in perf_sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
        
        # Adjust column widths
        perf_sheet.column_dimensions['A'].width = 25  # Strategy
        perf_sheet.column_dimensions['B'].width = 12  # Num_Trades
        perf_sheet.column_dimensions['C'].width = 14  # Total_Return
        perf_sheet.column_dimensions['D'].width = 14  # Profit_Factor
        perf_sheet.column_dimensions['E'].width = 12  # Win_Rate
        
        # Format Rules sheet
        for cell in rules_sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
        
        rules_sheet.column_dimensions['A'].width = 15  # Strategy
        rules_sheet.column_dimensions['B'].width = 70  # Description
        rules_sheet.column_dimensions['C'].width = 15  # Category
        
        # Add borders to cells
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for row in perf_sheet.iter_rows(min_row=2, max_row=perf_sheet.max_row, min_col=1, max_col=5):
            for cell in row:
                cell.border = thin_border
        
        for row in rules_sheet.iter_rows(min_row=2, max_row=rules_sheet.max_row, min_col=1, max_col=3):
            for cell in row:
                cell.border = thin_border
    
    print(f"Excel file saved to {filename}")


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
    
    # Create rules tables
    df_rules, exit_rules = create_rules_table()
    
    # Save to Excel
    save_to_excel(df_results, df_rules, exit_rules, 'ftd_table.xlsx')
    
    # Create performance plot
    plot_performance(equity_curves, 'ftd_performance.png')
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return df_results, equity_curves


if __name__ == "__main__":
    df_results, equity_curves = main()