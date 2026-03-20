"""
Pre-Refunding Announcement Trading Strategy

This strategy implements the trading approach from the paper:
"Pre-Refunding Announcement Gains in U.S. Treasurys" by Chen Wang and Kevin Zhao

The strategy goes long on TLT (iShares 20+ Year Treasury Bond ETF) 
on the day before the Treasury Refunding Announcement (TRA).

TRAs occur on the first Wednesday of February, May, August, and November.
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
            # Find the first Wednesday of the month
            # Start from the 1st of the month
            first_day = datetime(year, month, 1)
            
            # Calculate days until Wednesday
            # Wednesday is weekday 2 (Monday=0, Tuesday=1, Wednesday=2, ...)
            days_until_wednesday = (2 - first_day.weekday()) % 7
            first_wednesday = first_day + timedelta(days=days_until_wednesday)
            
            # TRA is on the first Wednesday
            tra_dates.append(first_wednesday)
    
    return pd.DatetimeIndex(tra_dates)

def get_pre_tra_dates(tra_dates):
    """
    Get the trading day before each TRA (pre-TRA day).
    Since TRAs are on Wednesdays, pre-TRA days are Tuesdays.
    """
    pre_tra_dates = []
    
    for tra_date in tra_dates:
        # Find the previous trading day (usually Tuesday, but check for holidays)
        pre_tra = tra_date - timedelta(days=1)
        
        # If it's a weekend, go back to Friday
        while pre_tra.weekday() >= 5:  # Saturday = 5, Sunday = 6
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
    
    # Step 1: Get TRA dates
    tra_dates = get_tra_dates(start_year=2002, end_year=2025)
    print(f"\nGenerated {len(tra_dates)} TRA dates from 2002-2025")
    print(f"First TRA: {tra_dates[0].strftime('%Y-%m-%d')}")
    print(f"Last TRA: {tra_dates[-1].strftime('%Y-%m-%d')}")
    
    # Step 2: Get pre-TRA dates
    pre_tra_dates = get_pre_tra_dates(tra_dates)
    print(f"\nGenerated {len(pre_tra_dates)} pre-TRA dates")
    print(f"First pre-TRA: {pre_tra_dates[0].strftime('%Y-%m-%d')}")
    print(f"Last pre-TRA: {pre_tra_dates[-1].strftime('%Y-%m-%d')}")
    
    # Step 3: Download TLT data
    print("\nDownloading TLT data from Yahoo Finance...")
    tlt = yf.download("TLT", start="2002-01-01", end="2025-12-31", 
                      progress=False, auto_adjust=True)
    
    if tlt.empty:
        print("Error: Failed to download TLT data")
        return
    
    # Flatten multi-level columns if present
    if isinstance(tlt.columns, pd.MultiIndex):
        tlt.columns = tlt.columns.get_level_values(0)
    
    print(f"Downloaded {len(tlt)} trading days of TLT data")
    print(f"Date range: {tlt.index[0].strftime('%Y-%m-%d')} to {tlt.index[-1].strftime('%Y-%m-%d')}")
    print(f"Columns: {tlt.columns.tolist()}")
    
    # Step 4: Calculate daily returns (use Close for adjusted price with auto_adjust=True)
    tlt['Return'] = tlt['Close'].pct_change()
    
    # Step 5: Generate trading signal
    # Signal = 1 on pre-TRA days, 0 otherwise
    tlt['Signal'] = 0
    tlt.loc[tlt.index.isin(pre_tra_dates), 'Signal'] = 1
    
    # Step 6: Calculate strategy returns
    # Strategy returns = Signal * TLT Return (we earn the TLT return on pre-TRA days)
    # On pre-TRA days (Signal=1): we get the TLT return
    # On other days (Signal=0): we hold cash (0 return)
    tlt['Strategy_Return'] = tlt['Signal'] * tlt['Return']
    
    # Drop NaN values
    tlt = tlt.dropna()
    
    # Step 7: Calculate cumulative returns
    # For strategy: only compound returns on pre-TRA days (other days = 0 return)
    tlt['Cumulative_TLT'] = (1 + tlt['Return']).cumprod()
    # For strategy, we only earn returns on signal days
    tlt['Cumulative_Strategy'] = (1 + tlt['Strategy_Return']).cumprod()
    
    # Step 8: Calculate performance metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    
    # Filter for dates where we have signal = 1
    pre_tra_days = tlt[tlt['Signal'] == 1]
    other_days = tlt[tlt['Signal'] == 0]
    
    print(f"\nNumber of pre-TRA trading days: {len(pre_tra_days)}")
    print(f"Number of other trading days: {len(other_days)}")
    
    # Strategy performance (only on signal days)
    strategy_returns = tlt['Strategy_Return']
    tlt_returns = tlt['Return']
    
    # Annualized return - calendar time vs paper method
    ann_return_strategy = calculate_annualized_return(strategy_returns)
    ann_return_strategy_paper = calculate_annualized_return_paper(strategy_returns)
    ann_return_tlt = calculate_annualized_return(tlt_returns)
    
    print(f"\n--- Annualized Returns ---")
    print(f"Pre-TRA Strategy (calendar time): {ann_return_strategy:.2%}")
    print(f"Pre-TRA Strategy (paper method): {ann_return_strategy_paper:.2%}")
    print(f"Buy & Hold TLT: {ann_return_tlt:.2%}")
    
    # Sharpe Ratio - regular vs paper method
    sharpe_strategy = calculate_sharpe_ratio(strategy_returns)
    sharpe_strategy_paper = calculate_sharpe_ratio_paper(strategy_returns)
    sharpe_tlt = calculate_sharpe_ratio(tlt_returns)
    
    print(f"\n--- Sharpe Ratios ---")
    print(f"Pre-TRA Strategy (regular): {sharpe_strategy:.2f}")
    print(f"Pre-TRA Strategy (paper method): {sharpe_strategy_paper:.2f}")
    print(f"Buy & Hold TLT: {sharpe_tlt:.2f}")
    
    # Maximum Drawdown
    dd_strategy = calculate_max_drawdown(tlt['Cumulative_Strategy'].apply(lambda x: np.log(x)))
    dd_tlt = calculate_max_drawdown(tlt['Cumulative_TLT'].apply(lambda x: np.log(x)))
    
    print(f"\n--- Maximum Drawdown ---")
    print(f"Pre-TRA Strategy: {dd_strategy:.2%}")
    print(f"Buy & Hold TLT: {dd_tlt:.2%}")
    
    # Average return on pre-TRA days vs other days
    avg_return_pre_tra = pre_tra_days['Return'].mean() * 100  # in basis points
    avg_return_other = other_days['Return'].mean() * 100
    
    print(f"\n--- Average Daily Returns ---")
    print(f"Pre-TRA days: {avg_return_pre_tra:.2f} bps")
    print(f"Other days: {avg_return_other:.4f} bps")
    
    # Total return
    total_return_strategy = (tlt['Cumulative_Strategy'].iloc[-1] - 1) * 100
    total_return_tlt = (tlt['Cumulative_TLT'].iloc[-1] - 1) * 100
    
    print(f"\n--- Total Return (Cumulative) ---")
    print(f"Pre-TRA Strategy: {total_return_strategy:.2f}%")
    print(f"Buy & Hold TLT: {total_return_tlt:.2f}%")
    
    # Step 9: Create output dataframe
    output_df = tlt[['Close', 'Return', 'Signal', 'Cumulative_TLT', 'Cumulative_Strategy']].copy()
    output_df.columns = ['TLT_Close', 'TLT_Daily_Return', 'Signal', 'Cumulative_TLT_Return', 'Cumulative_Strategy_Return']
    
    # Export to CSV
    output_df.to_csv('/workspace/project/openhands/tra_strategy_results.csv')
    print("\n" + "=" * 60)
    print("Exported results to: /workspace/project/openhands/tra_strategy_results.csv")
    print("=" * 60)
    
    # Step 10: Plot the equity curve
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Cumulative Returns
    plt.subplot(2, 2, 1)
    plt.plot(tlt.index, tlt['Cumulative_TLT'], label='Buy & Hold TLT', linewidth=1.5)
    plt.plot(tlt.index, tlt['Cumulative_Strategy'], label='Pre-TRA Strategy', linewidth=1.5)
    plt.title('Cumulative Returns: Pre-TRA Strategy vs Buy & Hold TLT', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Growth of $1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Signal Timeline
    plt.subplot(2, 2, 2)
    plt.fill_between(tlt.index, 0, tlt['Signal'], alpha=0.7, label='Pre-TRA Signal', color='green')
    plt.title('Trading Signal (1 = Long TLT)', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    tlt['Strategy_Drawdown'] = (tlt['Cumulative_Strategy'] - tlt['Cumulative_Strategy'].cummax()) / tlt['Cumulative_Strategy'].cummax()
    tlt['TLT_Drawdown'] = (tlt['Cumulative_TLT'] - tlt['Cumulative_TLT'].cummax()) / tlt['Cumulative_TLT'].cummax()
    
    plt.subplot(2, 2, 3)
    plt.plot(tlt.index, tlt['Strategy_Drawdown'] * 100, label='Pre-TRA Strategy', linewidth=1)
    plt.plot(tlt.index, tlt['TLT_Drawdown'] * 100, label='Buy & Hold TLT', linewidth=1, alpha=0.7)
    plt.title('Drawdown', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Returns Distribution
    plt.subplot(2, 2, 4)
    plt.hist(pre_tra_days['Return'] * 100, bins=50, alpha=0.7, label='Pre-TRA Days', color='green')
    plt.hist(other_days['Return'] * 100, bins=50, alpha=0.5, label='Other Days', color='blue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.title('Distribution of Daily Returns', fontsize=12)
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/project/openhands/tra_strategy_equity_curve.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: /workspace/project/openhands/tra_strategy_equity_curve.png")
    
    plt.show()
    
    # Print sample of pre-TRA dates and returns
    print("\n" + "=" * 60)
    print("Sample Pre-TRA Trading Days")
    print("=" * 60)
    sample = tlt[tlt['Signal'] == 1][['Close', 'Return', 'Signal']].head(20)
    print(sample)
    
    return tlt, output_df

if __name__ == "__main__":
    tlt, output_df = main()
