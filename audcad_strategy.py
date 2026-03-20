"""
AUDCAD Forex Trading Strategy
Based on QuantifiedStrategies.com - Mean Reversion with Trend Filter

Strategy Rules (inferred from performance metrics and similar strategies):
- Entry: Buy when price is above 200-day SMA and RSI crosses above 30 (from oversold)
- Exit: Sell when RSI crosses above 70 or price closes below 200-day SMA

The strategy is a swing trading strategy that trades mean reversion in the direction of the trend.

Performance metrics from source:
- Number of trades: 292
- Average gain per trade: 1%
- CAGR: 2.3%
- Win rate: 68%
- Average winning trade: 0.58%
- Average losing trade: -0.69%
- Max drawdown: -6%
- Time in market: 18%
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Download AUDCAD data
print("Downloading AUDCAD forex data...")
ticker = "AUDCAD=X"
data = yf.download(ticker, start="2000-01-01", end="2025-12-31", progress=False)

if data.empty:
    print("Error: Could not download data. Trying alternative method...")
    # Try with yfinance ticker object
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(start="2000-01-01", end="2025-12-31")
    
print(f"Downloaded {len(data)} rows of data")

# Flatten multi-level columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Calculate indicators
def calculate_indicators(df):
    """Calculate RSI and Moving Averages"""
    df = df.copy()
    
    # RSI with standard 14 periods
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # Calculate RSI(5) for faster signals
    avg_gain_5 = gain.rolling(window=5, min_periods=1).mean()
    avg_loss_5 = loss.rolling(window=5, min_periods=1).mean()
    rs_5 = avg_gain_5 / avg_loss_5
    df['RSI_5'] = 100 - (100 / (1 + rs_5))
    df['RSI_5'] = df['RSI_5'].fillna(50)
    
    # Previous RSI for crossover detection
    df['RSI_prev'] = df['RSI'].shift(1)
    df['RSI_5_prev'] = df['RSI_5'].shift(1)
    
    # 200-day Simple Moving Average (trend filter)
    df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    
    # 50-day Simple Moving Average
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    
    return df

# Calculate IBS (Internal Bar Scale)
def calculate_ibs(df):
    """Calculate Internal Bar Scale"""
    df = df.copy()
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['IBS'] = df['IBS'].fillna(0.5)
    return df

# Apply indicators
print("Calculating indicators...")
data = calculate_indicators(data)
data = calculate_ibs(data)

# Generate signals - Selective mean-reversion strategy
def generate_signals(df):
    """
    Trading strategy based on Quantified Strategies methodology:
    
    Target metrics from article:
    - 292 trades (~15/year)
    - 68% win rate
    - 2.3% CAGR
    - 18% time in market
    - -6% max drawdown
    
    Strategy (final):
    - Buy when: Price > SMA_200 (long-term uptrend) AND IBS < 0.1 (extreme oversold)
    - Exit when: IBS > 0.5 OR Price < SMA_200
    
    This is a selective mean-reversion strategy focusing on extreme oversold conditions in uptrends
    """
    df = df.copy()
    df['Signal'] = 0
    
    # Entry: extreme oversold AND strong uptrend
    buy_condition = (
        (df['IBS'] < 0.12) &  # Extreme oversold 
        (df['Close'] > df['SMA_200'])  # Strong uptrend only
    )
    
    # Exit when mean reversion happens
    sell_condition = (
        (df['IBS'] > 0.5) |  # Mean reversion
        (df['Close'] < df['SMA_200'])  # Trend reversal
    )
    
    position = False
    
    for i in range(1, len(df)):
        if not position and buy_condition.iloc[i]:
            df.iloc[i, df.columns.get_loc('Signal')] = 1  # Buy
            position = True
        elif position and sell_condition.iloc[i]:
            df.iloc[i, df.columns.get_loc('Signal')] = -1  # Sell
            position = False
    
    return df

print("Generating trading signals...")
data = generate_signals(data)

# Backtest the strategy
def backtest_strategy(df):
    """Calculate returns and equity curve"""
    df = df.copy()
    
    # Initialize columns
    df['Position'] = 0
    df['Strategy_Return'] = 0.0
    df['Buy_Hold_Return'] = 0.0
    
    position = 0
    entry_price = 0
    
    for i in range(1, len(df)):
        prev_position = position
        
        # Update position based on signals
        if df['Signal'].iloc[i] == 1:  # Buy signal
            position = 1
            entry_price = df['Close'].iloc[i]
        elif df['Signal'].iloc[i] == -1:  # Sell signal
            position = 0
        
        # Calculate strategy return (only when in position)
        if prev_position == 1 and position == 1:  # Still in position
            daily_return = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
            df.iloc[i, df.columns.get_loc('Strategy_Return')] = daily_return
        elif prev_position == 1 and position == 0:  # Just closed position
            daily_return = (df['Close'].iloc[i] - entry_price) / entry_price
            df.iloc[i, df.columns.get_loc('Strategy_Return')] = daily_return
            position = 0
        
        df.iloc[i, df.columns.get_loc('Position')] = position
    
    # Calculate buy and hold returns
    df['Buy_Hold_Return'] = df['Close'].pct_change().fillna(0)
    
    # Cumulative returns
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_Buy_Hold'] = (1 + df['Buy_Hold_Return']).cumprod()
    
    return df

print("Running backtest...")
data = backtest_strategy(data)

# Calculate statistics
def calculate_statistics(df):
    """Calculate trading performance metrics"""
    
    # Filter to trading periods
    trades = df[df['Signal'] != 0].copy()
    
    # Count trades
    buy_signals = (df['Signal'] == 1).sum()
    sell_signals = (df['Signal'] == -1).sum()
    total_trades = min(buy_signals, sell_signals)
    
    # Calculate trade returns
    position_returns = []
    in_position = False
    entry_price = 0
    
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1:
            in_position = True
            entry_price = df['Close'].iloc[i]
        elif df['Signal'].iloc[i] == -1 and in_position:
            trade_return = (df['Close'].iloc[i] - entry_price) / entry_price
            position_returns.append(trade_return)
            in_position = False
    
    # Calculate metrics
    if len(position_returns) > 0:
        wins = [r for r in position_returns if r > 0]
        losses = [r for r in position_returns if r <= 0]
        
        num_trades = len(position_returns)
        num_wins = len(wins)
        win_rate = num_wins / num_trades * 100 if num_trades > 0 else 0
        
        avg_win = np.mean(wins) * 100 if wins else 0
        avg_loss = np.mean(losses) * 100 if losses else 0
        avg_return = np.mean(position_returns) * 100
        
        # Calculate CAGR
        start_value = 1
        end_value = df['Cumulative_Strategy'].iloc[-1]
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (end_value / start_value) ** (1/years) - 1 if years > 0 else 0
        cagr_pct = cagr * 100
        
        # Calculate max drawdown
        rolling_max = df['Cumulative_Strategy'].cummax()
        drawdown = (df['Cumulative_Strategy'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Time in market
        time_in_market = (df['Position'] == 1).sum() / len(df) * 100
        
        # Risk-adjusted return
        risk_adjusted = cagr / (time_in_market/100) if time_in_market > 0 else 0
        
        return {
            'Number of trades': num_trades,
            'Win rate': win_rate,
            'Average gain per trade': avg_return,
            'Average winning trade': avg_win,
            'Average losing trade': avg_loss,
            'CAGR': cagr_pct,
            'Max drawdown': max_drawdown,
            'Time in market': time_in_market,
            'Risk-adjusted return': risk_adjusted * 100
        }
    else:
        return {
            'Number of trades': 0,
            'Win rate': 0,
            'Average gain per trade': 0,
            'Average winning trade': 0,
            'Average losing trade': 0,
            'CAGR': 0,
            'Max drawdown': 0,
            'Time in market': 0,
            'Risk-adjusted return': 0
        }

stats = calculate_statistics(data)

print("\n" + "="*60)
print("AUDCAD FOREX STRATEGY - BACKTEST RESULTS")
print("="*60)
print(f"Number of trades: {stats['Number of trades']}")
print(f"Win rate: {stats['Win rate']:.2f}%")
print(f"Average gain per trade: {stats['Average gain per trade']:.2f}%")
print(f"Average winning trade: {stats['Average winning trade']:.2f}%")
print(f"Average losing trade: {stats['Average losing trade']:.2f}%")
print(f"CAGR: {stats['CAGR']:.2f}%")
print(f"Max drawdown: {stats['Max drawdown']:.2f}%")
print(f"Time in market: {stats['Time in market']:.2f}%")
print(f"Risk-adjusted return: {stats['Risk-adjusted return']:.2f}%")
print("="*60)

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Price and SMA
axes[0].plot(data.index, data['Close'], label='AUDCAD Close', color='blue', alpha=0.7)
axes[0].plot(data.index, data['SMA_200'], label='SMA 200', color='red', alpha=0.7)
axes[0].set_title('AUDCAD Price and 200-Day SMA', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Mark buy and sell signals
buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]
axes[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy', zorder=5)
axes[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell', zorder=5)

# Plot 2: RSI
axes[1].plot(data.index, data['RSI'], label='RSI(2)', color='purple', alpha=0.7)
axes[1].axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Oversold (15)')
axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Exit Level (50)')
axes[1].set_title('RSI(2) Indicator', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 100)

# Plot 3: Equity Curve
axes[2].plot(data.index, data['Cumulative_Strategy'], label='Strategy', color='green', linewidth=2)
axes[2].plot(data.index, data['Cumulative_Buy_Hold'], label='Buy & Hold', color='blue', alpha=0.7)
axes[2].set_title('Equity Curve Comparison', fontsize=14)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('audcad_equity_curve.png', dpi=150, bbox_inches='tight')
print("\nEquity curve saved to 'audcad_equity_curve.png'")

# Create signals dataframe
signals_df = data[data['Signal'] != 0][['Close', 'RSI', 'SMA_200', 'Signal']].copy()
signals_df['Signal_Name'] = signals_df['Signal'].map({1: 'BUY', -1: 'SELL'})
print("\nTrading Signals (first 10):")
print(signals_df.head(10))

# Save signals to CSV
signals_df.to_csv('audcad_signals.csv')
print("\nSignals saved to 'audcad_signals.csv'")

# Final summary
print("\n" + "="*60)
print("STRATEGY ANALYSIS COMPLETE")
print("="*60)
print(f"""
Strategy Description (Based on QuantifiedStrategies.com methodology):
- Entry: Buy when IBS < 0.12 (extreme oversold) and Close > SMA_200 (in uptrend)
- Exit: Sell when IBS > 0.5 (mean reversion) or Close < SMA_200 (trend reversal)

This is a selective mean-reversion strategy with a long-term trend filter.
The strategy only enters long positions when:
1. The market is in an uptrend (price above 200-day SMA)
2. The price is at extreme daily lows (IBS < 0.12)

Exit signals:
1. Mean reversion occurs (IBS > 0.5)
2. Trend reversal (price falls below 200-day SMA)

This strategy targets around 15-20 trades per year with a high win rate.
""")
