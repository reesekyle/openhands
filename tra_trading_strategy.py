"""
Pre-Refunding Announcement Trading Strategy

This strategy implements the trading approach from the paper:
"Pre-Refunding Announcement Gains in U.S. Treasurys" by Chen Wang and Kevin Zhao

The strategy:
- Holds TLT (20+ Year Treasury Bond ETF) on TRA days (announcement day)
- Holds BIL (1-3 Month Treasury Bill ETF) on all other days

Compared to benchmark: BIL buy-and-hold
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_tra_dates(start_year=2002, end_year=2025):
    """
    Generate Treasury Refunding Announcement dates.
    TRAs occur on the first Wednesday of February, May, August, and November.
    """
    tra_dates = []
    
    for year in range(start_year, end_year + 1):
        for month in [2, 5, 8, 11]:  # Feb, May, Aug, Nov
            first_day = datetime(year, month, 1)
            days_until_wednesday = (2 - first_day.weekday()) % 7
            first_wednesday = first_day + timedelta(days=days_until_wednesday)
            tra_dates.append(first_wednesday)
    
    return pd.DatetimeIndex(tra_dates)

def get_pre_tra_dates(tra_dates):
    """
    Get the trading day before each TRA (pre-TRA day).
    """
    pre_tra_dates = []
    for tra_date in tra_dates:
        pre_tra = tra_date - timedelta(days=1)
        while pre_tra.weekday() >= 5:
            pre_tra -= timedelta(days=1)
        pre_tra_dates.append(pre_tra)
    return pd.DatetimeIndex(pre_tra_dates)

def calculate_max_drawdown(cumulative_returns):
    """
    Calculate the maximum drawdown from cumulative returns.
    """
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    # Calculate drawdown
    drawdown = cumulative_returns - running_max
    # Maximum drawdown
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_annualized_return(daily_returns):
    """
    Calculate annualized return based on calendar time.
    
    For a strategy that's only invested on certain days (like Pre-TRA),
    we annualize based on the full calendar period to get the true
    time-weighted return.
    """
    # Total return including all days (zeros are included for days not invested)
    total_return = (1 + daily_returns).prod() - 1
    
    # Number of years based on all calendar days in the dataset
    n_years = len(daily_returns) / 252
    
    if n_years == 0:
        return 0
    
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    return annualized_return

def calculate_annualized_return_paper(daily_returns):
    """
    Calculate annualized return the way the paper does it.
    
    The paper calculates returns only on pre-TRA days (the active trading days),
    then annualizes assuming 4 trading opportunities per year.
    """
    # Filter to only active returns (non-zero)
    active_returns = daily_returns[daily_returns != 0]
    
    if len(active_returns) == 0:
        return 0
    
    # Total return from active trading days only
    total_return = (1 + active_returns).prod() - 1
    
    # Number of years based on number of pre-TRA events (4 per year)
    n_years = len(active_returns) / 4.0
    
    if n_years == 0:
        return 0
    
    # Annualize: (1 + total_return)^(1/years) - 1
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    return annualized_return

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    """
    Calculate annualized Sharpe ratio.
    """
    # Use all returns (including zeros)
    if daily_returns.std() == 0:
        return 0
    
    excess_returns = daily_returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
    return sharpe

def calculate_sharpe_ratio_paper(daily_returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio the way the paper does it.
    Using only the active returns and annualizing assuming 4 periods per year.
    """
    # Filter to only active returns
    active_returns = daily_returns[daily_returns != 0]
    
    if len(active_returns) == 0 or active_returns.std() == 0:
        return 0
    
    # Assume risk-free rate of ~2% annually
    daily_rf = 0.02 / 252
    
    excess_returns = active_returns - daily_rf
    
    # Annualize: multiply by sqrt(4) since we have 4 trading days per year
    sharpe = np.sqrt(4) * excess_returns.mean() / active_returns.std()
    return sharpe

def main():
    print("=" * 60)
    print("Pre-Refunding Announcement Trading Strategy")
    print("=" * 60)
    
    # Step 1: Get TRA dates and pre-TRA dates
    tra_dates = get_tra_dates(start_year=2002, end_year=2025)
    pre_tra_dates = get_pre_tra_dates(tra_dates)
    
    print(f"\nGenerated {len(tra_dates)} TRA dates from 2002-2025")
    print(f"First TRA: {tra_dates[0].strftime('%Y-%m-%d')}")
    print(f"Last TRA: {tra_dates[-1].strftime('%Y-%m-%d')}")
    print(f"First pre-TRA: {pre_tra_dates[0].strftime('%Y-%m-%d')}")
    print(f"Last pre-TRA: {pre_tra_dates[-1].strftime('%Y-%m-%d')}")
    
    # Step 2: Download TLT and BIL data
    print("\nDownloading TLT and BIL data from Yahoo Finance...")
    
    # TLT: 20+ Year Treasury Bond ETF
    tlt = yf.download("TLT", start="2002-01-01", end="2025-12-31", 
                      progress=False, auto_adjust=True)
    # BIL: 1-3 Month Treasury Bill ETF
    bil = yf.download("BIL", start="2002-01-01", end="2025-12-31", 
                      progress=False, auto_adjust=True)
    
    # Flatten multi-level columns if present
    if isinstance(tlt.columns, pd.MultiIndex):
        tlt.columns = tlt.columns.get_level_values(0)
    if isinstance(bil.columns, pd.MultiIndex):
        bil.columns = bil.columns.get_level_values(0)
    
    # Rename columns to distinguish
    tlt = tlt.rename(columns={'Close': 'TLT_Close'})
    bil = bil.rename(columns={'Close': 'BIL_Close'})
    
    # Combine into single dataframe
    df = pd.DataFrame({
        'TLT_Close': tlt['TLT_Close'],
        'BIL_Close': bil['BIL_Close']
    })
    
    # Calculate daily returns
    df['TLT_Return'] = df['TLT_Close'].pct_change()
    df['BIL_Return'] = df['BIL_Close'].pct_change()
    
    print(f"Downloaded {len(df)} trading days")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Step 3: Generate trading signal
    # Signal = 1 on pre-TRA days (hold TLT), 0 otherwise (hold BIL)
    df['Signal'] = 0
    df.loc[df.index.isin(pre_tra_dates), 'Signal'] = 1
    
    # Step 4: Calculate strategy returns
    # On TRA days: hold TLT (get TLT return)
    # On other days: hold BIL (get BIL return)
    df['Strategy_Return'] = np.where(df['Signal'] == 1, df['TLT_Return'], df['BIL_Return'])
    
    # Drop NaN values
    df = df.dropna()
    
    # Step 5: Calculate cumulative returns
    df['Cumulative_TLT'] = (1 + df['TLT_Return']).cumprod()
    df['Cumulative_BIL'] = (1 + df['BIL_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    
    # Step 6: Calculate performance metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    
    # Filter for dates where we have signal = 1 (TRA days)
    tra_days = df[df['Signal'] == 1]
    other_days = df[df['Signal'] == 0]
    
    print(f"\nNumber of TRA trading days: {len(tra_days)}")
    print(f"Number of other trading days: {len(other_days)}")
    
    # Strategy performance
    strategy_returns = df['Strategy_Return']
    bil_returns = df['BIL_Return']
    tlt_returns = df['TLT_Return']
    
    # Annualized return
    ann_return_strategy = calculate_annualized_return(strategy_returns)
    ann_return_bil = calculate_annualized_return(bil_returns)
    ann_return_tlt = calculate_annualized_return(tlt_returns)
    
    print(f"\n--- Annualized Returns ---")
    print(f"TRA Strategy (TLT on TRA days, BIL otherwise): {ann_return_strategy:.2%}")
    print(f"BIL Buy & Hold: {ann_return_bil:.2%}")
    print(f"TLT Buy & Hold: {ann_return_tlt:.2%}")
    
    # Sharpe Ratio
    sharpe_strategy = calculate_sharpe_ratio(strategy_returns)
    sharpe_bil = calculate_sharpe_ratio(bil_returns)
    sharpe_tlt = calculate_sharpe_ratio(tlt_returns)
    
    print(f"\n--- Sharpe Ratios ---")
    print(f"TRA Strategy: {sharpe_strategy:.2f}")
    print(f"BIL Buy & Hold: {sharpe_bil:.2f}")
    print(f"TLT Buy & Hold: {sharpe_tlt:.2f}")
    
    # Maximum Drawdown
    dd_strategy = calculate_max_drawdown(np.log(df['Cumulative_Strategy']))
    dd_bil = calculate_max_drawdown(np.log(df['Cumulative_BIL']))
    dd_tlt = calculate_max_drawdown(np.log(df['Cumulative_TLT']))
    
    print(f"\n--- Maximum Drawdown ---")
    print(f"TRA Strategy: {dd_strategy:.2%}")
    print(f"BIL Buy & Hold: {dd_bil:.2%}")
    print(f"TLT Buy & Hold: {dd_tlt:.2%}")
    
    # Total Return
    total_return_strategy = (df['Cumulative_Strategy'].iloc[-1] - 1) * 100
    total_return_bil = (df['Cumulative_BIL'].iloc[-1] - 1) * 100
    total_return_tlt = (df['Cumulative_TLT'].iloc[-1] - 1) * 100
    
    print(f"\n--- Total Return (Cumulative) ---")
    print(f"TRA Strategy: {total_return_strategy:.2f}%")
    print(f"BIL Buy & Hold: {total_return_bil:.2f}%")
    print(f"TLT Buy & Hold: {total_return_tlt:.2f}%")
    
    # Average returns on TRA days
    print(f"\n--- Average Daily Returns ---")
    print(f"TRA days (TLT): {tra_days['TLT_Return'].mean()*10000:.2f} bps")
    print(f"Other days (BIL): {other_days['BIL_Return'].mean()*10000:.2f} bps")
    
    # Step 7: Create output dataframe
    output_df = df[['TLT_Close', 'BIL_Close', 'TLT_Return', 'BIL_Return', 'Signal', 
                    'Cumulative_TLT', 'Cumulative_BIL', 'Cumulative_Strategy']].copy()
    
    # Export to CSV
    output_df.to_csv('/workspace/project/openhands/tra_strategy_results.csv')
    print("\n" + "=" * 60)
    print("Exported results to: /workspace/project/openhands/tra_strategy_results.csv")
    print("=" * 60)
    
    # Step 8: Plot the equity curve
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Cumulative Returns
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['Cumulative_BIL'], label='BIL Buy & Hold (Benchmark)', linewidth=1.5, color='orange')
    plt.plot(df.index, df['Cumulative_Strategy'], label='TRA Strategy', linewidth=2, color='green')
    plt.title('Cumulative Returns Comparison', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Growth of $1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Signal Timeline
    plt.subplot(2, 2, 2)
    plt.fill_between(df.index, 0, df['Signal'], alpha=0.7, label='Signal (1=TLT, 0=BIL)', color='green')
    plt.title('Trading Signal (Hold TLT on Pre-TRA Days, BIL Otherwise)', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    df['Strategy_Drawdown'] = (df['Cumulative_Strategy'] - df['Cumulative_Strategy'].cummax()) / df['Cumulative_Strategy'].cummax()
    df['BIL_Drawdown'] = (df['Cumulative_BIL'] - df['Cumulative_BIL'].cummax()) / df['Cumulative_BIL'].cummax()
    
    plt.subplot(2, 2, 3)
    plt.plot(df.index, df['Strategy_Drawdown'] * 100, label='TRA Strategy', linewidth=1, color='green')
    plt.plot(df.index, df['BIL_Drawdown'] * 100, label='BIL', linewidth=1, color='orange', alpha=0.7)
    plt.title('Drawdown', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Returns on pre-TRA days
    plt.subplot(2, 2, 4)
    plt.hist(tra_days['TLT_Return'] * 100, bins=30, alpha=0.7, label='TLT on Pre-TRA Days', color='green')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.title('Distribution of TLT Returns on Pre-TRA Days', fontsize=12)
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/project/openhands/tra_strategy_equity_curve.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: /workspace/project/openhands/tra_strategy_equity_curve.png")
    
    plt.show()
    
    # Print sample of TRA dates and returns
    print("\n" + "=" * 60)
    print("Sample TRA Days")
    print("=" * 60)
    print(df[df['Signal'] == 1][['TLT_Close', 'TLT_Return', 'BIL_Return', 'Signal']].head(20))
    
    return df, output_df

if __name__ == "__main__":
    tlt, output_df = main()
