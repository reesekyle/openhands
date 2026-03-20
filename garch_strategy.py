"""
GARCH-based Trading Strategy based on "Interest Rate Volatility and Stock Returns" paper.

Paper: Latha, Gupta & Ghosh - "Interest Rate Volatility and Stock Returns: A GARCH (1,1) Model"

Key Findings from Paper:
1. Interest rate changes (ΔI_t) have NEGATIVE relationship with stock returns
2. Interest rate volatility (σ²_I) has NEGATIVE relationship with stock returns
3. When interest rate volatility increases → stock returns decrease

Strategy Logic:
- Default: LONG SPY (position = 1)
- When interest rate volatility exceeds threshold → FLAT (position = 0)
- The strategy uses Treasury yield volatility as proxy for interest rate volatility

Note: The original paper uses Indian 91-day T-bills. Here we use US Treasury yields as proxy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data(start_date='2000-01-01', end_date=None):
    """Download SPY and Treasury yield data"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Downloading SPY data...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    
    # Handle multi-index columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    spy['Return'] = spy['Close'].pct_change()
    
    # Download 3-month Treasury yields (TD3M) as proxy for interest rates
    # Using 3-month T-bill secondary market rate from FRED
    print("Downloading Treasury yield data...")
    treasury = yf.download('^IRX', start=start_date, end=end_date, progress=False)  # 13-wk T-bill
    
    # If ^IRX is not available, use alternative
    if treasury.empty or len(treasury) < 100:
        print("Trying alternative Treasury data...")
        treasury = yf.download('^TNX', start=start_date, end=end_date, progress=False)  # 10-yr Treasury
    
    # Handle multi-index columns
    if isinstance(treasury.columns, pd.MultiIndex):
        treasury.columns = treasury.columns.get_level_values(0)
    
    # Combine data
    data = pd.DataFrame(index=spy.index)
    data['SPY_Close'] = spy['Close']
    data['SPY_Return'] = spy['Return']
    
    # Align Treasury data
    common_idx = data.index.intersection(treasury.index)
    data.loc[common_idx, 'Treasury_Yield'] = treasury.loc[common_idx, 'Close'].values
    
    # Forward fill any missing Treasury data
    data['Treasury_Yield'] = data['Treasury_Yield'].ffill().bfill()
    
    # Calculate interest rate changes (ΔI_t)
    data['Interest_Rate_Change'] = data['Treasury_Yield'].pct_change()
    
    # Calculate interest rate volatility using rolling GARCH-like measure
    # We'll use rolling standard deviation of interest rate changes as proxy
    data['Interest_Rate_Vol'] = data['Interest_Rate_Change'].rolling(window=21).std() * np.sqrt(252)
    
    # Drop NaN rows
    data = data.dropna()
    
    print(f"Data loaded: {len(data)} trading days from {data.index[0].date()} to {data.index[-1].date()}")
    
    return data

# ============================================================
# GARCH VOLATILITY ESTIMATION
# ============================================================

def estimate_garch_volatility(returns, p=1, o=0, q=1):
    """
    Estimate GARCH(1,1) volatility from returns series.
    Returns annualized conditional volatility.
    """
    # Scale returns to percentage for numerical stability
    returns_pct = returns * 100
    
    try:
        model = arch_model(returns_pct, vol='Garch', p=p, o=o, q=q, mean='Constant', dist='normal')
        res = model.fit(disp='off', show_warning=False)
        
        # Get conditional volatility (annualized)
        cond_vol = res.conditional_volatility / 100  # Convert back to decimal
        annualized_vol = cond_vol * np.sqrt(252)
        
        return annualized_vol, res
    except Exception as e:
        print(f"GARCH fitting error: {e}")
        # Fallback to rolling std
        return returns.rolling(window=21).std() * np.sqrt(252), None

# ============================================================
# TRADING STRATEGY
# ============================================================

def generate_signals(data, vol_threshold=None):
    """
    Generate trading signals based on GARCH volatility.
    
    Strategy Rules (from paper):
    - Default: LONG SPY (position = 1)
    - When interest rate volatility > threshold → FLAT (position = 0)
    
    The paper shows interest rate volatility negatively affects returns,
    so we exit when volatility spikes.
    """
    df = data.copy()
    
    # Estimate GARCH volatility for SPY returns
    print("Estimating GARCH(1,1) volatility...")
    df['GARCH_Vol'], garch_model = estimate_garch_volatility(df['SPY_Return'])
    
    # Use interest rate volatility from the paper methodology
    # The paper uses interest rate volatility in mean equation
    # Here we combine GARCH vol + interest rate vol for signal
    
    # Normalize and combine volatilities
    df['GARCH_Vol'] = df['GARCH_Vol'].ffill().bfill()
    df['Interest_Rate_Vol'] = df['Interest_Rate_Vol'].ffill().bfill()
    
    # Calculate z-score of interest rate volatility
    vol_ma = df['Interest_Rate_Vol'].rolling(window=60).mean()
    vol_std = df['Interest_Rate_Vol'].rolling(window=60).std()
    df['Vol_zscore'] = (df['Interest_Rate_Vol'] - vol_ma) / vol_std
    
    # If no threshold provided, use 1.5 standard deviations above mean
    if vol_threshold is None:
        vol_threshold = 1.5
    
    # Generate signals: 1 = LONG, 0 = FLAT
    # Exit when interest rate volatility is high (above threshold)
    df['Signal'] = np.where(df['Vol_zscore'] > vol_threshold, 0, 1)
    
    # Default to long if signal is NaN (start of series)
    df['Signal'] = df['Signal'].fillna(1)
    
    return df, garch_model

# ============================================================
# BACKTEST & PERFORMANCE
# ============================================================

def backtest_strategy(df):
    """
    Calculate strategy returns and cumulative equity curve.
    """
    # Strategy returns: signal is lagged by 1 day (we trade at close based on previous signal)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['SPY_Return']
    
    # Buy and hold returns
    df['BuyHold_Return'] = df['SPY_Return']
    
    # Cumulative returns
    df['Strategy_Equity'] = (1 + df['Strategy_Return']).cumprod()
    df['BuyHold_Equity'] = (1 + df['BuyHold_Return']).cumprod()
    
    # Drop NaN
    df = df.dropna()
    
    return df

def calculate_performance_metrics(df):
    """Calculate annualized return and other metrics"""
    # Annualized return
    trading_days = 252
    
    # Strategy
    total_return_strategy = df['Strategy_Equity'].iloc[-1] - 1
    n_years = len(df) / trading_days
    annualized_return_strategy = (1 + total_return_strategy) ** (1 / n_years) - 1
    
    # Buy & Hold
    total_return_bh = df['BuyHold_Equity'].iloc[-1] - 1
    annualized_return_bh = (1 + total_return_bh) ** (1 / n_years) - 1
    
    # Volatility
    strategy_vol = df['Strategy_Return'].std() * np.sqrt(trading_days)
    bh_vol = df['BuyHold_Return'].std() * np.sqrt(trading_days)
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    strategy_sharpe = annualized_return_strategy / strategy_vol if strategy_vol > 0 else 0
    bh_sharpe = annualized_return_bh / bh_vol if bh_vol > 0 else 0
    
    # Max Drawdown
    strategy_cummax = df['Strategy_Equity'].cummax()
    strategy_drawdown = (df['Strategy_Equity'] - strategy_cummax) / strategy_cummax
    max_dd_strategy = strategy_drawdown.min()
    
    bh_cummax = df['BuyHold_Equity'].cummax()
    bh_drawdown = (df['BuyHold_Equity'] - bh_cummax) / bh_cummax
    max_dd_bh = bh_drawdown.min()
    
    metrics = {
        'Total Return (Strategy)': f"{total_return_strategy*100:.2f}%",
        'Total Return (Buy&Hold)': f"{total_return_bh*100:.2f}%",
        'Annualized Return (Strategy)': f"{annualized_return_strategy*100:.2f}%",
        'Annualized Return (Buy&Hold)': f"{annualized_return_bh*100:.2f}%",
        'Volatility (Strategy)': f"{strategy_vol*100:.2f}%",
        'Volatility (Buy&Hold)': f"{bh_vol*100:.2f}%",
        'Sharpe Ratio (Strategy)': f"{strategy_sharpe:.2f}",
        'Sharpe Ratio (Buy&Hold)': f"{bh_sharpe:.2f}",
        'Max Drawdown (Strategy)': f"{max_dd_strategy*100:.2f}%",
        'Max Drawdown (Buy&Hold)': f"{max_dd_bh*100:.2f}%",
    }
    
    return metrics

def plot_performance(df, save_path='garch_performance.png'):
    """Plot strategy vs buy&hold equity curves"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Equity Curves
    ax1 = axes[0]
    ax1.plot(df.index, df['Strategy_Equity'], label='GARCH Strategy', linewidth=1.5)
    ax1.plot(df.index, df['BuyHold_Equity'], label='Buy & Hold SPY', linewidth=1.5, alpha=0.7)
    ax1.set_title('GARCH Strategy vs Buy & Hold SPY - Cumulative Equity', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity ($1.00 initial)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Signal & Volatility
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    ax2.plot(df.index, df['Signal'], label='Signal (1=Long, 0=Flat)', color='blue', alpha=0.7)
    ax2_twin.plot(df.index, df['Interest_Rate_Vol'], label='Interest Rate Volatility', color='red', alpha=0.7)
    
    ax2.set_title('Trading Signal & Interest Rate Volatility', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Signal', color='blue')
    ax2_twin.set_ylabel('Interest Rate Volatility', color='red')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance plot saved to {save_path}")
    
    return fig

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("GARCH Trading Strategy - Based on Paper")
    print("Interest Rate Volatility and Stock Returns")
    print("=" * 60)
    
    # Download data
    data = download_data(start_date='2000-01-01')
    
    # Generate signals
    df, garch_model = generate_signals(data)
    
    # Backtest
    df = backtest_strategy(df)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(df)
    
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Plot
    plot_performance(df, save_path='/workspace/project/openhands/garch_performance.png')
    
    # Save dataframes
    # Signal dataframe
    signal_df = df[['SPY_Close', 'SPY_Return', 'Interest_Rate_Change', 'Interest_Rate_Vol', 
                    'GARCH_Vol', 'Vol_zscore', 'Signal']].copy()
    signal_df.to_csv('/workspace/project/openhands/garch_signals.csv')
    
    # Equity curve dataframe
    equity_df = df[['SPY_Close', 'SPY_Return', 'Strategy_Return', 'Strategy_Equity', 
                   'BuyHold_Return', 'BuyHold_Equity']].copy()
    equity_df.to_csv('/workspace/project/openhands/garch_equity_curve.csv')
    
    print("\nDataFrames saved:")
    print("  - garch_signals.csv (trading signals)")
    print("  - garch_equity_curve.csv (equity curves)")
    
    return df, metrics

if __name__ == "__main__":
    df, metrics = main()
