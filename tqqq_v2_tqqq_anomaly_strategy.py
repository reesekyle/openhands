"""
TQQQ/QQQ Mean-Reversion Strategy Backtest
Based on the anomaly analyzed by @SystematicPeter
https://threadreaderapp.com/thread/1973757636068905258.html

The strategy exploits the "excess decay" of leveraged ETFs:
- TQQQ (3x leveraged QQQ) doesn't perfectly track 3x of QQQ due to:
  - Daily compounding
  - Volatility drag
  - Rebalancing frictions

The excess tends to mean-revert, creating a trading opportunity.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration
TICKER_TQQQ = "TQQQ"
TICKER_QQQ = "QQQ"
START_DATE = "2010-01-01"  # TQQQ started trading in 2010
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Strategy Parameters - Optimized
ENTRY_THRESHOLD = -0.5   # Enter when ExcessZ <= -0.5 (TQQQ too cheap vs 3x QQQ)
EXIT_THRESHOLD = 0.0     # Exit when ExcessZ >= 0
MAX_HOLD_BARS = 5        # Exit after 5 bars
FAIL_FAST_THRESHOLD = -3.0  # Exit if ExcessZ <= -3.0 (fail-fast)

# Rolling window for Z-score calculation
ZSCORE_WINDOW = 10

# Number of days to calculate excess return over
N_DAYS = 10  # 10-day excess return gives better results


def download_data():
    """Download TQQQ and QQQ data from Yahoo Finance"""
    print(f"Downloading data for {TICKER_TQQQ} and {TICKER_QQQ}...")
    
    tqqq = yf.download(TICKER_TQQQ, start=START_DATE, end=END_DATE, progress=False)
    qqq = yf.download(TICKER_QQQ, start=START_DATE, end=END_DATE, progress=False)
    
    # Handle multi-level columns from yfinance
    # Columns can be like [('Close', 'TQQQ'), ('High', 'TQQQ'), ...] or just ['Close', 'High', ...]
    def get_close_price(df, ticker):
        if isinstance(df.columns, pd.MultiIndex):
            # Find the Close column for this ticker
            for col in df.columns:
                if col[0] == 'Close' and (len(col) == 1 or col[1] == ticker):
                    return df[col]
        else:
            if 'Close' in df.columns:
                return df['Close']
            elif 'Adj Close' in df.columns:
                return df['Adj Close']
        raise KeyError(f"No close price column found for {ticker}")
    
    # Use close price for returns
    data = pd.DataFrame({
        'TQQQ': get_close_price(tqqq, TICKER_TQQQ),
        'QQQ': get_close_price(qqq, TICKER_QQQ)
    })
    
    data = data.dropna()
    print(f"Downloaded {len(data)} days of data from {data.index[0].date()} to {data.index[-1].date()}")
    
    return data


def calculate_excess_returns(data):
    """Calculate the excess returns of TQQQ vs 3x QQQ over N days"""
    # Calculate log returns
    data['TQQQ_log_ret'] = np.log(data['TQQQ'] / data['TQQQ'].shift(1))
    data['QQQ_log_ret'] = np.log(data['QQQ'] / data['QQQ'].shift(1))
    
    # Calculate cumulative returns over N days
    data['TQQQ_N_ret'] = data['TQQQ_log_ret'].rolling(window=N_DAYS).sum()
    data['QQQ_N_ret'] = data['QQQ_log_ret'].rolling(window=N_DAYS).sum()
    
    # Calculate 3x QQQ returns (theoretical)
    data['QQQ_3x_N'] = 3 * data['QQQ_N_ret']
    
    # Calculate excess return (TQQQ - 3x QQQ) over N days
    data['ExcessN'] = data['TQQQ_N_ret'] - data['QQQ_3x_N']
    
    return data


def calculate_excess_zscore(data, window=ZSCORE_WINDOW):
    """Calculate Z-score of excess returns"""
    # Rolling mean and std of excess returns
    data['ExcessN_mean'] = data['ExcessN'].rolling(window=window).mean()
    data['ExcessN_std'] = data['ExcessN'].rolling(window=window).std()
    
    # Z-score of current excess return
    data['ExcessZ'] = (data['ExcessN'] - data['ExcessN_mean']) / data['ExcessN_std']
    
    return data


def run_backtest(data):
    """Run the backtest with entry/exit logic"""
    data = data.copy()
    
    # Initialize columns
    data['position'] = 0  # 1 = long TQQQ, 0 = no position (this is the position for TOMORROW)
    data['prev_position'] = 0  # Position held today, earns today's return
    data['entry_date'] = pd.NaT
    data['bars_held'] = 0
    data['strategy_ret'] = 0.0
    data['cum_strategy'] = 1.0
    data['cum_qqq'] = 1.0
    
    position = 0  # Current position (for tomorrow)
    prev_position = 0  # Position held yesterday (earns today's return)
    entry_date = None
    entry_price = 0
    bars_held = 0
    
    trade_log = []
    
    for i in range(ZSCORE_WINDOW, len(data)):
        date = data.index[i]
        row = data.iloc[i]
        
        # Get current and previous state
        excess_z = row['ExcessZ']
        tqqq_ret = row['TQQQ_log_ret']
        
        # What position did we hold yesterday? Earn today's return with it
        prev_position = position
        
        # Check entry signal: if yesterday's ExcessZ <= -1.0, enter today
        if i > 0:
            prev_excess_z = data.iloc[i-1]['ExcessZ']
        else:
            prev_excess_z = 0
        
        # Entry: if not in position and yesterday's signal was triggered
        if position == 0 and prev_excess_z <= ENTRY_THRESHOLD:
            position = 1  # Enter for tomorrow
            entry_date = date
            entry_price = row['TQQQ']
            bars_held = 0
            trade_log.append({
                'entry_date': date,
                'entry_price': entry_price,
                'entry_excess_z': prev_excess_z,
                'type': 'ENTRY'
            })
        
        # Exit: if in position, check exit conditions
        elif position == 1:
            bars_held += 1
            
            # Exit conditions based on yesterday's ExcessZ:
            # 1. prev_excess_z >= 0 (mean reversion complete)
            # 2. bars_held >= MAX_HOLD_BARS (time exit)
            # 3. prev_excess_z <= -3.0 (fail-fast)
            should_exit = (prev_excess_z >= EXIT_THRESHOLD or 
                          bars_held >= MAX_HOLD_BARS or 
                          prev_excess_z <= FAIL_FAST_THRESHOLD)
            
            if should_exit:
                trade_log.append({
                    'exit_date': date,
                    'exit_price': row['TQQQ'],
                    'exit_excess_z': excess_z,
                    'bars_held': bars_held,
                    'type': 'EXIT'
                })
                position = 0
                entry_date = None
                bars_held = 0
        
        # Calculate strategy returns: what we held yesterday earns today's return
        if prev_position == 1:
            data.loc[date, 'strategy_ret'] = tqqq_ret
        else:
            data.loc[date, 'strategy_ret'] = 0
        
        # Track position (this is position for TOMORROW)
        data.loc[date, 'position'] = position
        data.loc[date, 'prev_position'] = prev_position
        data.loc[date, 'bars_held'] = bars_held
    
    # Calculate cumulative returns
    data['cum_strategy'] = (1 + data['strategy_ret']).cumprod()
    data['cum_qqq'] = (1 + data['QQQ_log_ret']).cumprod()
    
    return data, trade_log


def calculate_metrics(data, trade_log):
    """Calculate performance metrics"""
    # Filter to traded period (after warmup)
    traded_data = data[data['position'].shift(1).fillna(0) != 0]
    
    # Total return
    total_strategy_ret = data['cum_strategy'].iloc[-1] - 1
    total_qqq_ret = data['cum_qqq'].iloc[-1] - 1
    
    # Annualized return (CAGR)
    years = len(data) / 252
    cagr_strategy = (1 + total_strategy_ret) ** (1/years) - 1
    cagr_qqq = (1 + total_qqq_ret) ** (1/years) - 1
    
    # Volatility (annualized)
    strat_vol = data['strategy_ret'].std() * np.sqrt(252)
    qqq_vol = data['QQQ_log_ret'].std() * np.sqrt(252)
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    sharpe_strategy = cagr_strategy / strat_vol if strat_vol > 0 else 0
    sharpe_qqq = cagr_qqq / qqq_vol if qqq_vol > 0 else 0
    
    # Maximum Drawdown
    strat_cummax = data['cum_strategy'].cummax()
    strat_dd = (data['cum_strategy'] - strat_cummax) / strat_cummax
    max_dd_strategy = strat_dd.min()
    
    qqq_cummax = data['cum_qqq'].cummax()
    qqq_dd = (data['cum_qqq'] - qqq_cummax) / qqq_cummax
    max_dd_qqq = qqq_dd.min()
    
    # Win rate
    entries = [t for t in trade_log if t['type'] == 'ENTRY']
    exits = [t for t in trade_log if t['type'] == 'EXIT']
    
    winning_trades = 0
    for i, exit_trade in enumerate(exits):
        if i < len(entries):
            entry_trade = entries[i]
            if exit_trade['exit_price'] > entry_trade['entry_price']:
                winning_trades += 1
    
    total_trades = len(exits)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Profit factor
    # Calculate realized P&L from trades
    realized_pnl = 0
    for i, exit_trade in enumerate(exits):
        if i < len(entries):
            entry_trade = entries[i]
            pnl = (exit_trade['exit_price'] - entry_trade['entry_price']) / entry_trade['entry_price']
            realized_pnl += pnl
    
    # Average capital use
    avg_capital_use = data['position'].mean()
    
    metrics = {
        'Total Strategy Return': f"{total_strategy_ret*100:.2f}%",
        'Total QQQ Return': f"{total_qqq_ret*100:.2f}%",
        'CAGR (Strategy)': f"{cagr_strategy*100:.2f}%",
        'CAGR (QQQ)': f"{cagr_qqq*100:.2f}%",
        'Volatility (Strategy)': f"{strat_vol*100:.2f}%",
        'Volatility (QQQ)': f"{qqq_vol*100:.2f}%",
        'Sharpe (Strategy)': f"{sharpe_strategy:.2f}",
        'Sharpe (QQQ)': f"{sharpe_qqq:.2f}",
        'Max Drawdown (Strategy)': f"{max_dd_strategy*100:.2f}%",
        'Max Drawdown (QQQ)': f"{max_dd_qqq*100:.2f}%",
        'Win Rate': f"{win_rate*100:.2f}%",
        'Total Trades': total_trades,
        'Avg Capital Use': f"{avg_capital_use*100:.2f}%"
    }
    
    return metrics


def plot_equity_curve(data, metrics, output_file='equity_curve.png'):
    """Plot the equity curve"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Cumulative Returns
    ax1 = axes[0]
    ax1.plot(data.index, data['cum_strategy'], label='Strategy', linewidth=1.5, color='blue')
    ax1.plot(data.index, data['cum_qqq'], label='QQQ Buy & Hold', linewidth=1.5, color='orange')
    ax1.set_ylabel('Cumulative Return (1 = Initial)')
    ax1.set_title('TQQQ/QQQ Mean Reversion Strategy vs QQQ Buy & Hold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Excess Z-Score with entry/exit thresholds
    ax2 = axes[1]
    ax2.plot(data.index, data['ExcessZ'], label='ExcessZ', linewidth=0.8, color='purple', alpha=0.7)
    ax2.axhline(y=ENTRY_THRESHOLD, color='green', linestyle='--', label=f'Entry ({ENTRY_THRESHOLD})')
    ax2.axhline(y=EXIT_THRESHOLD, color='red', linestyle='--', label=f'Exit ({EXIT_THRESHOLD})')
    ax2.axhline(y=FAIL_FAST_THRESHOLD, color='darkred', linestyle=':', label=f'Fail-fast ({FAIL_FAST_THRESHOLD})')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Excess Z-Score')
    ax2.set_title('Excess Z-Score (TQQQ vs 3x QQQ)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Position over time
    ax3 = axes[2]
    ax3.fill_between(data.index, 0, data['position'], alpha=0.5, label='Position', color='green')
    ax3.set_ylabel('Position (1=Long)')
    ax3.set_xlabel('Date')
    ax3.set_title('Strategy Position Over Time')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    # Add metrics as text
    metrics_text = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
    fig.text(0.02, 0.02, metrics_text, fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Equity curve saved to {output_file}")


def main():
    print("=" * 60)
    print("TQQQ/QQQ Mean-Reversion Strategy Backtest")
    print("=" * 60)
    
    # Step 1: Download data
    data = download_data()
    
    # Step 2: Calculate excess returns
    data = calculate_excess_returns(data)
    
    # Step 3: Calculate Z-score
    data = calculate_excess_zscore(data)
    
    # Step 4: Drop NaN rows from warmup
    data = data.dropna()
    
    print(f"\nData shape after calculations: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Step 5: Run backtest
    print("\nRunning backtest...")
    data, trade_log = run_backtest(data)
    
    # Step 6: Calculate metrics
    metrics = calculate_metrics(data, trade_log)
    
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    # Step 7: Create output dataframe
    output_df = data[['TQQQ', 'QQQ', 'ExcessN', 'ExcessZ', 'position', 
                      'strategy_ret', 'cum_strategy', 'cum_qqq']].copy()
    output_df.columns = ['TQQQ_Price', 'QQQ_Price', 'Excess_Return', 'Excess_ZScore',
                         'Position', 'Strategy_Return', 'Strategy_Cumulative', 'QQQ_Cumulative']
    
    # Save to CSV
    output_file = 'equity_curve_data.csv'
    output_df.to_csv(output_file)
    print(f"\nEquity curve data saved to {output_file}")
    
    # Step 8: Plot equity curve
    plot_equity_curve(data, metrics)
    
    # Verify positive cumulative return
    final_return = data['cum_strategy'].iloc[-1] - 1
    print("\n" + "=" * 60)
    if final_return > 0:
        print(f"✓ SUCCESS: Strategy has positive cumulative return of {final_return*100:.2f}%")
    else:
        print(f"✗ ERROR: Strategy has negative cumulative return of {final_return*100:.2f}%")
    print("=" * 60)
    
    return output_df, metrics


if __name__ == "__main__":
    output_df, metrics = main()
