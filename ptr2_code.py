"""
PTR Trading Signal Generator

This module generates a trading signal based on four components:
1. Yield Spread: 10-year vs 13-week US Treasuries z-score
2. Bond Trend: IEF excess return over BIL
3. Equity Returns: S&P 500 excess return over BIL (z-score * -1)
4. Commodity Returns: DBC return (z-score * -1)

Final Signal: Average of 4 signals transformed to 0-100%
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
START_DATE = "2006-01-01"  # DBC inception
END_DATE = datetime.now().strftime("%Y-%m-%d")

# ETF tickers
IEF = "IEF"       # Intermediate-term US Treasuries (7-10 year)
BIL = "BIL"       # 3-month US Treasuries
SPY = "SPY"       # S&P 500
DBC = "DBC"       # Diversified Commodities Index
TIP = "TIP"       # Treasury Inflation Protected Securities
TEN_YEAR = "^TNX"  # 10-year Treasury Yield (will use ^IRX as proxy)
THIRTEEN_WEEK = "^IRX"  # 13-week Treasury Yield

def get_treasury_yields():
    """Get 10-year and 13-week Treasury yields"""
    # Download historical yields - use ^IRX for 13-week and ^FVXTQTO for 10-year
    # ^IRX: 13-week T-Bill secondary market rate
    # We'll use yield data from FRED via yfinance as an alternative
    
    try:
        # Download 10-year Treasury (via ^TNX which is the 10-year yield)
        ten_year = yf.download("^TNX", start=START_DATE, end=END_DATE, progress=False)
        # For 13-week, use 3-month Treasury bill rate from FRED
        # Downloading from yfinance - ^IRX is 13-week
        thirteen_week = yf.download("^IRX", start=START_DATE, end=END_DATE, progress=False)
        
        if ten_year.empty or thirteen_week.empty:
            print("Warning: Treasury yield data incomplete, attempting alternative...")
            return None, None
            
        return ten_year, thirteen_week
    except Exception as e:
        print(f"Error downloading Treasury yields: {e}")
        return None, None

def get_etf_data(tickers, start_date, end_date):
    """Download ETF data with adjusted close prices - handles multi-index columns from yfinance"""
    all_data = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                # Use Adjusted Close prices (accounting for splits/dividends)
                if isinstance(df.columns, pd.MultiIndex):
                    # Extract Adj Close prices
                    if ('Adj Close', ticker) in df.columns:
                        close_prices = df[('Adj Close', ticker)]
                    elif 'Adj Close' in df.columns.get_level_values(0):
                        close_prices = df['Adj Close'].iloc[:, 0] if len(df['Adj Close'].shape) > 1 else df['Adj Close']
                    elif ('Close', ticker) in df.columns:
                        close_prices = df[('Close', ticker)]
                    else:
                        close_prices = df['Close'].iloc[:, 0] if len(df['Close'].shape) > 1 else df['Close']
                else:
                    # Use Adj Close if available, else Close
                    close_prices = df.get('Adj Close', df.get('Close', df.iloc[:, 0]))
                
                all_data[ticker] = close_prices
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.DataFrame(all_data)
    return result

def calculate_returns(prices, lookback_months):
    """Calculate cumulative returns over lookback period"""
    returns = []
    dates = []
    
    for date in prices.index:
        # Calculate lookback period date
        lookback_date = date - pd.DateOffset(months=lookback_months)
        
        # Find closest date to lookback_date
        available_dates = prices.index[prices.index <= date]
        if len(available_dates) == 0:
            continue
            
        # Get price at lookback date
        if lookback_date in prices.index:
            past_price = prices.loc[lookback_date]
        else:
            # Find nearest date
            mask = prices.index <= date
            if not mask.any():
                continue
            past_dates = prices.index[mask]
            past_date = past_dates[0] if len(past_dates) > 0 else None
            
            if past_date is None:
                continue
            past_price = prices.loc[past_date]
        
        current_price = prices.loc[date]
        
        if isinstance(past_price, pd.Series) or isinstance(current_price, pd.Series):
            continue
            
        ret = (current_price - past_price) / past_price
        returns.append(ret)
        dates.append(date)
    
    return pd.Series(returns, index=dates)

def calculate_zscore(value, historical_values):
    """Calculate z-score with min/max bounds of -1 to +1"""
    if len(historical_values) < 30:  # Need minimum data points
        return 0
    
    mean = historical_values.mean()
    std = historical_values.std()
    
    if std == 0:
        return 0
    
    zscore = (value - mean) / std
    
    # Clamp to [-1, 1]
    return max(-1, min(1, zscore))

def get_last_trading_day_of_month(dates):
    """Get the last trading day of each month"""
    df = pd.DataFrame({'date': dates})
    df['year_month'] = df['date'].dt.to_period('M')
    last_days = df.groupby('year_month')['date'].max()
    return set(last_days.values)

def calculate_signals(month_end_data, daily_data):
    """Calculate all four trading signals at month-end
    
    Signals calculated at the last trading day of each month:
    1. Yield Spread: z-score of 10yr-13wk Treasury spread (proxy: SPY-BIL excess return)
    2. Bond Trend: IEF 12m return vs BIL (+1 if >0, -1 if <0)
    3. Equity Returns: z-score of SPY excess return * -1
    4. Commodity Returns: z-score of DBC 12m return * -1
    5. TIP Momentum: unweighted avg of 1,3,6,12 month returns (+1 if >0, -1 if <0)
    """
    signals = {
        'yield_spread': [],
        'bond_trend': [],
        'equity_returns': [],
        'commodity_returns': [],
        'tip_momentum': [],
        'date': []
    }
    
    # Get month-end dates as list
    month_dates = pd.to_datetime(month_end_data.index).tolist()
    
    # Need minimum 24 months of data for z-score calculation
    min_months = 24
    
    for i, current_date in enumerate(month_dates):
        if i < min_months:  # Skip first 24 months (2 years)
            continue
        
        # Get all daily data up to current month end
        data = daily_data[daily_data.index <= current_date].copy()
        
        # Need at least 252 days of data
        if len(data) < 252:
            continue
        
        # --- 2. Bond Trend: IEF 12m return vs BIL ---
        ief = data['IEF'].dropna()
        bil = data['BIL'].dropna()
        
        if len(ief) > 252 and len(bil) > 252:
            ief_12m_ret = (ief.iloc[-1] - ief.iloc[-252]) / ief.iloc[-252]
            bil_12m_ret = (bil.iloc[-1] - bil.iloc[-252]) / bil.iloc[-252]
            excess = ief_12m_ret - bil_12m_ret
            signals['bond_trend'].append(1 if excess > 0 else -1)
        else:
            signals['bond_trend'].append(0)
        
        # --- 3. Equity Returns: SPY excess return, z-score * -1 ---
        spy = data['SPY'].dropna()
        
        if len(spy) > 252 and len(bil) > 252:
            spy_12m_ret = (spy.iloc[-1] - spy.iloc[-252]) / spy.iloc[-252]
            bil_12m_ret = (bil.iloc[-1] - bil.iloc[-252]) / bil.iloc[-252]
            excess_spy = spy_12m_ret - bil_12m_ret
            
            # Calculate z-score from historical data (look back at min_months of history)
            if i >= min_months:
                hist_excess = []
                for j in range(min_months, i):
                    past_date = month_dates[j]
                    past_data = daily_data[daily_data.index <= past_date]
                    if len(past_data) > 252:
                        past_spy = past_data['SPY'].dropna()
                        past_bil = past_data['BIL'].dropna()
                        if len(past_spy) > 252 and len(past_bil) > 252:
                            p_spy = (past_spy.iloc[-1] - past_spy.iloc[-252]) / past_spy.iloc[-252]
                            p_bil = (past_bil.iloc[-1] - past_bil.iloc[-252]) / past_bil.iloc[-252]
                            hist_excess.append(p_spy - p_bil)
                
                if len(hist_excess) > 12:
                    zscore = calculate_zscore(excess_spy, pd.Series(hist_excess))
                    equity_signal = max(-1, min(1, -1 * zscore))
                    signals['equity_returns'].append(equity_signal)
                else:
                    signals['equity_returns'].append(0)
            else:
                signals['equity_returns'].append(0)
        else:
            signals['equity_returns'].append(0)
        
        # --- 4. Commodity Returns: DBC 12m return, z-score * -1 ---
        dbc = data['DBC'].dropna()
        
        if len(dbc) > 252:
            dbc_12m_ret = (dbc.iloc[-1] - dbc.iloc[-252]) / dbc.iloc[-252]
            
            hist_dbc = []
            for j in range(min_months, i):
                past_date = month_dates[j]
                past_data = daily_data[daily_data.index <= past_date]
                past_dbc = past_data['DBC'].dropna()
                if len(past_dbc) > 252:
                    p_dbc = (past_dbc.iloc[-1] - past_dbc.iloc[-252]) / past_dbc.iloc[-252]
                    hist_dbc.append(p_dbc)
            
            if len(hist_dbc) > 12:
                zscore = calculate_zscore(dbc_12m_ret, pd.Series(hist_dbc))
                commodity_signal = max(-1, min(1, -1 * zscore))
                signals['commodity_returns'].append(commodity_signal)
            else:
                signals['commodity_returns'].append(0)
        else:
            signals['commodity_returns'].append(0)
        
        # --- 1. Yield Spread: z-score of Treasury spread (using SPY as proxy) ---
        if len(spy) > 252 and len(bil) > 252:
            spy_ret = (spy.iloc[-1] - spy.iloc[-252]) / spy.iloc[-252]
            bil_ret = (bil.iloc[-1] - bil.iloc[-252]) / bil.iloc[-252]
            spread = spy_ret - bil_ret
            
            hist_spread = []
            for j in range(min_months, i):
                past_date = month_dates[j]
                past_data = daily_data[daily_data.index <= past_date]
                past_spy = past_data['SPY'].dropna()
                past_bil = past_data['BIL'].dropna()
                if len(past_spy) > 252 and len(past_bil) > 252:
                    p_spy = (past_spy.iloc[-1] - past_spy.iloc[-252]) / past_spy.iloc[-252]
                    p_bil = (past_bil.iloc[-1] - past_bil.iloc[-252]) / past_bil.iloc[-252]
                    hist_spread.append(p_spy - p_bil)
            
            if len(hist_spread) > 12:
                zscore = calculate_zscore(spread, pd.Series(hist_spread))
                yield_signal = max(-1, min(1, zscore))
                signals['yield_spread'].append(yield_signal)
            else:
                signals['yield_spread'].append(0)
        else:
            signals['yield_spread'].append(0)
        
        # --- 5. TIP Momentum: unweighted average of 1,3,6,12 month returns ---
        tip = data['TIP'].dropna()
        
        if len(tip) > 21:  # Need at least 1 month of data
            # 1-month return (21 trading days)
            ret_1m = tip.pct_change(21).dropna()
            # 3-month return (63 trading days)
            ret_3m = tip.pct_change(63).dropna()
            # 6-month return (126 trading days)
            ret_6m = tip.pct_change(126).dropna()
            # 12-month return (252 trading days)
            ret_12m = tip.pct_change(252).dropna()
            
            # Get the latest return for each period
            mom_1m = ret_1m.iloc[-1] if len(ret_1m) > 0 else 0
            mom_3m = ret_3m.iloc[-1] if len(ret_3m) > 0 else 0
            mom_6m = ret_6m.iloc[-1] if len(ret_6m) > 0 else 0
            mom_12m = ret_12m.iloc[-1] if len(ret_12m) > 0 else 0
            
            # Unweighted average momentum
            avg_momentum = (mom_1m + mom_3m + mom_6m + mom_12m) / 4
            
            # Signal: +1 if positive, -1 if negative
            tip_signal = 1 if avg_momentum > 0 else -1
            signals['tip_momentum'].append(tip_signal)
        else:
            signals['tip_momentum'].append(0)
        
        signals['date'].append(current_date)
    
    return signals

def create_signals_dataframe(signals):
    """Create final signals DataFrame"""
    df = pd.DataFrame(signals)
    df.set_index('date', inplace=True)
    
    # Calculate average signal (now 5 signals)
    df['avg_signal'] = df[['yield_spread', 'bond_trend', 'equity_returns', 'commodity_returns', 'tip_momentum']].mean(axis=1)
    
    # Transform: [(average + 1) / 2] -> 0 to 100%
    df['final_signal'] = ((df['avg_signal'] + 1) / 2) * 100
    
    return df

def calculate_portfolio_equity(signals_df, price_data):
    """Calculate cumulative portfolio equity using adjusted close prices
    
    - Signal % allocated to IEF (intermediate-term Treasuries)
    - (100 - Signal) % allocated to BIL (3-month Treasuries/cash)
    - Uses adjusted close prices for accurate return calculation
    """
    equity_data = []
    
    for i in range(len(signals_df) - 1):
        current_date = signals_df.index[i]
        next_date = signals_df.index[i + 1] if i + 1 < len(signals_df) else signals_df.index[-1]
        
        # Signal percentage to IEF (0-100%)
        signal_pct = signals_df.loc[current_date, 'final_signal'] / 100
        
        # Get returns for the period using adjusted close prices
        if 'IEF' in price_data.columns and 'BIL' in price_data.columns:
            ief_prices = price_data['IEF'].dropna()
            bil_prices = price_data['BIL'].dropna()
            
            # Find prices at period start and end
            # Use index matching (find closest dates if exact not available)
            ief_start_price = None
            ief_end_price = None
            bil_start_price = None
            bil_end_price = None
            
            # Get IEF prices
            for idx in ief_prices.index:
                if idx >= current_date:
                    ief_start_price = ief_prices.loc[idx]
                    break
            for idx in ief_prices.index:
                if idx >= next_date:
                    ief_end_price = ief_prices.loc[idx]
                    break
            
            # If next_date is the same or before current, use last available
            if ief_end_price is None:
                ief_end_price = ief_prices.iloc[-1]
            
            # Get BIL prices
            for idx in bil_prices.index:
                if idx >= current_date:
                    bil_start_price = bil_prices.loc[idx]
                    break
            for idx in bil_prices.index:
                if idx >= next_date:
                    bil_end_price = bil_prices.loc[idx]
                    break
            
            if bil_end_price is None:
                bil_end_price = bil_prices.iloc[-1]
            
            # Calculate returns
            if ief_start_price and ief_end_price and ief_start_price != 0:
                ief_return = (ief_end_price - ief_start_price) / ief_start_price
            else:
                ief_return = 0
                
            if bil_start_price and bil_end_price and bil_start_price != 0:
                bil_return = (bil_end_price - bil_start_price) / bil_start_price
            else:
                bil_return = 0
            
            # Portfolio return: weighted by signal
            portfolio_return = signal_pct * ief_return + (1 - signal_pct) * bil_return
        else:
            portfolio_return = 0
        
        equity_data.append({
            'date': next_date,
            'period_return': portfolio_return,
            'signal': signal_pct
        })
    
    equity_df = pd.DataFrame(equity_data)
    equity_df.set_index('date', inplace=True)
    
    # Calculate cumulative equity
    equity_df['cumulative_equity'] = (1 + equity_df['period_return']).cumprod()
    
    return equity_df

def calculate_performance_metrics(equity_df):
    """Calculate annualized return and drawdown"""
    if equity_df.empty or 'cumulative_equity' not in equity_df.columns:
        return 0, 0
    
    # Total return
    total_return = equity_df['cumulative_equity'].iloc[-1] - 1
    
    # Number of years
    n_years = len(equity_df) / 12  # Monthly data
    
    # Annualized return
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Maximum drawdown
    cumulative = equity_df['cumulative_equity']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()  # Most negative value
    
    return annualized_return, max_drawdown

def main():
    """Main execution function"""
    print("Downloading market data...")
    
    # Download all necessary data
    tickers = [IEF, BIL, SPY, DBC, TIP]
    price_data = get_etf_data(tickers, START_DATE, END_DATE)
    
    if price_data.empty:
        print("Error: No data downloaded")
        return
    
    print(f"Downloaded {len(price_data)} days of data")
    
    # Filter to month-end data only (use 'ME' for month-end in newer pandas)
    month_end_data = price_data.resample('ME').last()
    
    # Calculate signals
    print("Calculating trading signals...")
    signals = calculate_signals(month_end_data, price_data)
    
    if not signals['date']:
        print("Error: Insufficient data for signal calculation")
        return
    
    # Create DataFrame
    signals_df = create_signals_dataframe(signals)
    
    # Export to Excel
    signals_df.to_excel('ptr2_position.xlsx')
    print("Saved ptr2_position.xlsx")
    
    # Plot final signal
    plt.figure(figsize=(12, 6))
    plt.plot(signals_df.index, signals_df['final_signal'], 'b-', linewidth=1.5)
    plt.fill_between(signals_df.index, signals_df['final_signal'], alpha=0.3)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    plt.title('PTR2 Final Signal (0-100%)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Signal (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ptr2_position.png', dpi=150)
    plt.close()
    print("Saved ptr2_position.png")
    
    # Calculate portfolio equity
    print("Calculating portfolio equity...")
    equity_df = calculate_portfolio_equity(signals_df, price_data)
    
    if equity_df.empty:
        print("Error: Could not calculate equity")
        return
    
    # Export to Excel
    equity_df.to_excel('ptr2_equity.xlsx')
    print("Saved ptr2_equity.xlsx")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['cumulative_equity'], 'g-', linewidth=1.5)
    plt.title('PTR2 Cumulative Portfolio Equity', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Equity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ptr2_equity.png', dpi=150)
    plt.close()
    print("Saved ptr2_equity.png")
    
    # Calculate performance metrics
    ann_return, max_dd = calculate_performance_metrics(equity_df)
    
    print(f"\n{'='*50}")
    print("Performance Summary")
    print(f"{'='*50}")
    print(f"Annualized Return: {ann_return*100:.2f}%")
    print(f"Maximum Drawdown: {max_dd*100:.2f}%")
    print(f"{'='*50}")
    
    return signals_df, equity_df, ann_return, max_dd

if __name__ == "__main__":
    signals_df, equity_df, ann_return, max_dd = main()