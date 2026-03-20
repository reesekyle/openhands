"""
IRV Strategy - Based on the paper "Equity Return and Short-Term Interest Rate Volatility: 
Level Effects and Asymmetric Dynamics" by Henry, Olekalns, and Suardi

This strategy implements findings from the paper:
1. Level Effect: Interest rate volatility peaks with the level of short rates
2. Asymmetric Response: Negative equity shocks lead to higher volatility than positive ones
3. The paper found that when short-term interest rates are high, equity volatility is high
4. Negative news about equities increases volatility more than positive news

Trading Rules (Flat or Long, no shorting):
- Default: LONG (1)
- FLAT (0) when:
  a) Interest rate level is high (above threshold) - level effect suggests high volatility
  b) Negative equity shock (previous day's return is negative) - asymmetric volatility effect
  
The strategy is long by default and only goes flat when conditions suggest 
elevated risk based on the paper's findings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Parameters
TICKER = 'SPY'
START_DATE = '1993-01-01'  # SPY inception
END_DATE = '2024-12-31'
TBILL_TICKER = '^IRX'  # 13-week T-Bill rate (closest to 3-month)

def get_data():
    """Fetch SPY and T-Bill data"""
    print("Fetching data...")
    
    # Get SPY data
    spy = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    # Handle multi-level columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.droplevel(1, axis=1)
    
    # Handle different column name formats from yfinance
    if 'Adj Close' in spy.columns:
        price_col = 'Adj Close'
    elif 'Close' in spy.columns:
        price_col = 'Close'
    else:
        price_col = spy.columns[0]
    
    spy_returns = spy[price_col].pct_change().dropna()
    spy_returns.name = 'spy_return'
    
    # Get T-Bill data (13-week rate)
    tbill = yf.download(TBILL_TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    if isinstance(tbill.columns, pd.MultiIndex):
        tbill = tbill.droplevel(1, axis=1)
    
    if 'Adj Close' in tbill.columns:
        tbill_col = 'Adj Close'
    elif 'Close' in tbill.columns:
        tbill_col = 'Close'
    else:
        tbill_col = tbill.columns[0]
    
    tbill_rate = tbill[tbill_col].dropna()
    tbill_rate.name = 'tbill_rate'
    
    # Align data
    combined = pd.DataFrame({
        'spy_return': spy_returns,
        'tbill_rate': tbill_rate
    }).dropna()
    
    print(f"Data range: {combined.index[0]} to {combined.index[-1]}")
    print(f"Total observations: {len(combined)}")
    
    return combined

def calculate_conditional_volatility(returns, p=5, q=1):
    """
    Calculate conditional volatility using GARCH(1,1) model
    This follows the methodology in the paper for estimating time-varying volatility
    """
    # Scale returns to percentage for numerical stability
    returns_pct = returns * 100
    
    try:
        model = arch_model(returns_pct, vol='Garch', p=p, q=q, mean='Constant', dist='normal')
        result = model.fit(disp='off')
        cond_vol = result.conditional_volatility / 100  # Scale back
        return cond_vol, result
    except Exception as e:
        print(f"GARCH fitting error: {e}")
        # Fallback to rolling standard deviation
        return pd.Series(rolling_std(returns, 22), index=returns.index), None

def rolling_std(returns, window):
    """Calculate rolling standard deviation"""
    return returns.rolling(window=window).std()

def calculate_volatility_threshold(combined, lookback=252):
    """
    Calculate threshold for high interest rate level
    Based on the paper's findings that volatility depends on interest rate level
    """
    # Use rolling median of T-Bill rate as threshold
    rolling_median = combined['tbill_rate'].rolling(lookback).median()
    return rolling_median

def generate_signals(combined):
    """
    Generate trading signals based on the IRV paper methodology
    
    Signal = 1 (Long) by default
    Signal = 0 (Flat) when:
    1. Interest rate level is above its rolling median (high rate = high volatility per level effect)
    2. Previous day return was negative (asymmetric volatility effect)
    """
    print("\nGenerating trading signals...")
    
    # Calculate returns
    returns = combined['spy_return'].copy()
    
    # Previous day's return (for asymmetric effect detection)
    prev_return = returns.shift(1)
    
    # Interest rate level threshold (rolling median)
    rate_threshold = calculate_volatility_threshold(combined)
    
    # Signal conditions based on paper:
    # 1. Level effect: High interest rate -> high volatility -> go flat
    high_rate_signal = combined['tbill_rate'] > rate_threshold
    
    # 2. Asymmetric effect: Negative return yesterday -> higher volatility today -> go flat
    negative_shock_signal = prev_return < 0
    
    # Combined signal: flat if either condition is true
    combined['signal'] = np.where(
        (high_rate_signal.fillna(False)) | (negative_shock_signal.fillna(False)),
        0,  # Flat
        1   # Long
    )
    
    # Store intermediate values for analysis
    combined['prev_return'] = prev_return
    combined['rate_threshold'] = rate_threshold
    combined['high_rate'] = high_rate_signal
    combined['negative_shock'] = negative_shock_signal
    
    # Calculate conditional volatility using GARCH
    print("Estimating GARCH conditional volatility...")
    cond_vol, garch_result = calculate_conditional_volatility(returns)
    combined['conditional_vol'] = cond_vol
    
    if garch_result is not None:
        print(f"GARCH parameters: omega={garch_result.params['omega']:.6f}, "
              f"alpha[1]={garch_result.params['alpha[1]']:.6f}, "
              f"beta[1]={garch_result.params['beta[1]']:.6f}")
    
    return combined, garch_result

def calculate_strategy_returns(combined):
    """
    Calculate strategy returns based on signals
    - Signal = 1: Long SPY (earn the return)
    - Signal = 0: Flat (earn 0, stay in cash)
    """
    # Strategy return = signal * SPY return
    combined['strategy_return'] = combined['signal'].shift(1) * combined['spy_return']
    
    # Buy and hold return
    combined['buyhold_return'] = combined['spy_return']
    
    # Fill NaN with 0 for initial periods
    combined['strategy_return'] = combined['strategy_return'].fillna(0)
    
    return combined

def calculate_equity_curves(combined):
    """
    Calculate cumulative equity curves
    """
    # Cumulative returns (assuming starting at 1.0)
    combined['strategy_equity'] = (1 + combined['strategy_return']).cumprod()
    combined['buyhold_equity'] = (1 + combined['buyhold_return']).cumprod()
    
    return combined

def calculate_performance_stats(combined):
    """
    Calculate annualized performance statistics
    """
    # Annualization factor (252 trading days)
    years = len(combined) / 252
    
    # Strategy stats
    strategy_total_return = combined['strategy_equity'].iloc[-1] - 1
    strategy_annualized = (1 + strategy_total_return) ** (1/years) - 1
    strategy_vol = combined['strategy_return'].std() * np.sqrt(252)
    strategy_sharpe = strategy_annualized / strategy_vol if strategy_vol > 0 else 0
    
    # Buy and hold stats
    buyhold_total_return = combined['buyhold_equity'].iloc[-1] - 1
    buyhold_annualized = (1 + buyhold_total_return) ** (1/years) - 1
    buyhold_vol = combined['buyhold_return'].std() * np.sqrt(252)
    buyhold_sharpe = buyhold_annualized / buyhold_vol if buyhold_vol > 0 else 0
    
    # Max drawdown
    strategy_cummax = combined['strategy_equity'].cummax()
    strategy_drawdown = (combined['strategy_equity'] - strategy_cummax) / strategy_cummax
    strategy_max_dd = strategy_drawdown.min()
    
    buyhold_cummax = combined['buyhold_equity'].cummax()
    buyhold_drawdown = (combined['buyhold_equity'] - buyhold_cummax) / buyhold_cummax
    buyhold_max_dd = buyhold_drawdown.min()
    
    # Time in market
    time_in_market = combined['signal'].mean() * 100
    
    stats = {
        'Strategy': {
            'Total Return': f"{strategy_total_return*100:.2f}%",
            'Annualized Return': f"{strategy_annualized*100:.2f}%",
            'Volatility': f"{strategy_vol*100:.2f}%",
            'Sharpe Ratio': f"{strategy_sharpe:.2f}",
            'Max Drawdown': f"{strategy_max_dd*100:.2f}%",
            'Time in Market': f"{time_in_market:.1f}%"
        },
        'Buy & Hold SPY': {
            'Total Return': f"{buyhold_total_return*100:.2f}%",
            'Annualized Return': f"{buyhold_annualized*100:.2f}%",
            'Volatility': f"{buyhold_vol*100:.2f}%",
            'Sharpe Ratio': f"{buyhold_sharpe:.2f}",
            'Max Drawdown': f"{buyhold_max_dd*100:.2f}%"
        }
    }
    
    return stats, {
        'strategy_annualized': strategy_annualized,
        'buyhold_annualized': buyhold_annualized,
        'strategy_vol': strategy_vol,
        'buyhold_vol': buyhold_vol,
        'strategy_sharpe': strategy_sharpe,
        'buyhold_sharpe': buyhold_sharpe,
        'strategy_max_dd': strategy_max_dd,
        'buyhold_max_dd': buyhold_max_dd,
        'time_in_market': time_in_market,
        'years': years
    }

def plot_performance(combined, stats, save_path='irv_performance.png'):
    """
    Plot strategy performance comparison
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Equity Curves
    ax1 = axes[0]
    ax1.plot(combined.index, combined['strategy_equity'], label='IRV Strategy', linewidth=1.5)
    ax1.plot(combined.index, combined['buyhold_equity'], label='Buy & Hold SPY', linewidth=1.5, alpha=0.7)
    ax1.set_title('Cumulative Equity Curves: IRV Strategy vs Buy & Hold SPY', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Signal Distribution
    ax2 = axes[1]
    ax2.fill_between(combined.index, 0, combined['signal'], alpha=0.5, label='Position (1=Long, 0=Flat)', color='green')
    ax2.set_title('IRV Strategy Position Over Time', fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Position')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Flat (0)', 'Long (1)'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Plot 3: Drawdown
    strategy_cummax = combined['strategy_equity'].cummax()
    strategy_drawdown = (combined['strategy_equity'] - strategy_cummax) / strategy_cummax * 100
    buyhold_cummax = combined['buyhold_equity'].cummax()
    buyhold_drawdown = (combined['buyhold_equity'] - buyhold_cummax) / buyhold_cummax * 100
    
    ax3 = axes[2]
    ax3.fill_between(combined.index, strategy_drawdown, 0, alpha=0.5, label='IRV Strategy', color='blue')
    ax3.fill_between(combined.index, buyhold_drawdown, 0, alpha=0.3, label='Buy & Hold SPY', color='orange')
    ax3.set_title('Drawdown Analysis', fontsize=14)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    # Add stats text box
    stats_text = f"""Performance Summary (Annualized):
    
IRV Strategy:
  Return: {stats['Strategy']['Annualized Return']}
  Sharpe: {stats['Strategy']['Sharpe Ratio']}
  Max DD: {stats['Strategy']['Max Drawdown']}
  Time in Market: {stats['Strategy']['Time in Market']}

Buy & Hold SPY:
  Return: {stats['Buy & Hold SPY']['Annualized Return']}
  Sharpe: {stats['Buy & Hold SPY']['Sharpe Ratio']}
  Max DD: {stats['Buy & Hold SPY']['Max Drawdown']}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {save_path}")
    plt.close()
    
    return fig

def main():
    """Main execution function"""
    print("=" * 60)
    print("IRV Trading Strategy - Based on Henry, Olekalns & Suardi (2005)")
    print("=" * 60)
    
    # Step 1: Get data
    combined = get_data()
    
    # Step 2: Generate signals
    combined, garch_result = generate_signals(combined)
    
    # Step 3: Calculate strategy returns
    combined = calculate_strategy_returns(combined)
    
    # Step 4: Calculate equity curves
    combined = calculate_equity_curves(combined)
    
    # Step 5: Calculate performance stats
    stats, detailed_stats = calculate_performance_stats(combined)
    
    # Step 6: Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    print(f"\nAnalysis Period: {years_to_years(detailed_stats['years']):.1f} years")
    
    print("\n--- IRV Strategy ---")
    for key, value in stats['Strategy'].items():
        print(f"  {key}: {value}")
    
    print("\n--- Buy & Hold SPY ---")
    for key, value in stats['Buy & Hold SPY'].items():
        print(f"  {key}: {value}")
    
    # Step 7: Create performance plot
    plot_performance(combined, stats, 'irv_performance.png')
    
    # Step 8: Show sample of trading signals
    print("\n" + "=" * 60)
    print("SAMPLE TRADING SIGNALS (Last 10 days)")
    print("=" * 60)
    sample_cols = ['spy_return', 'tbill_rate', 'signal', 'strategy_return']
    print(combined[sample_cols].tail(10).to_string())
    
    # Save dataframes
    signal_df = combined[['spy_return', 'tbill_rate', 'signal', 'strategy_return', 
                          'buyhold_return', 'strategy_equity', 'buyhold_equity']].copy()
    signal_df.to_csv('irv_signals.csv')
    print("\nTrading signals saved to: irv_signals.csv")
    
    return combined, stats, detailed_stats

def years_to_years(years):
    """Convert to years"""
    return years

if __name__ == "__main__":
    combined, stats, detailed_stats = main()
