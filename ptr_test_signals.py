#!/usr/bin/env python3
"""
Test all signal combinations to find optimal return/drawdown ratio
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configuration
START_DATE = '2006-01-01'
END_DATE = '2026-04-30'
IEF = 'IEF'
BIL = 'BIL'
SPY = 'SPY'
DBC = 'DBC'

def get_etf_data(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                if ('Adj Close', ticker) in df.columns:
                    close_prices = df[('Adj Close', ticker)]
                else:
                    close_prices = df['Close'].iloc[:, 0]
            else:
                close_prices = df.get('Adj Close', df.get('Close', df.iloc[:, 0]))
            all_data[ticker] = close_prices
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def calculate_zscore(value, historical_values):
    if len(historical_values) < 30:
        return 0
    mean = historical_values.mean()
    std = historical_values.std()
    if std == 0:
        return 0
    zscore = (value - mean) / std
    return max(-1, min(1, zscore))

def calculate_signals_v2(month_end_data, daily_data, included_signals):
    signals = {k: [] for k in included_signals}
    signals['date'] = []
    
    month_dates = pd.to_datetime(month_end_data.index).tolist()
    min_months = 24
    
    for i, current_date in enumerate(month_dates):
        if i < min_months:
            continue
        data = daily_data[daily_data.index <= current_date].copy()
        if len(data) < 252:
            continue
        
        ief = data['IEF'].dropna()
        bil = data['BIL'].dropna()
        spy = data['SPY'].dropna()
        dbc = data['DBC'].dropna()
        
        # Bond Trend
        if 'bond_trend' in included_signals:
            if len(ief) > 252 and len(bil) > 252:
                ief_12m_ret = (ief.iloc[-1] - ief.iloc[-252]) / ief.iloc[-252]
                bil_12m_ret = (bil.iloc[-1] - bil.iloc[-252]) / bil.iloc[-252]
                excess = ief_12m_ret - bil_12m_ret
                signals['bond_trend'].append(1 if excess > 0 else -1)
            else:
                signals['bond_trend'].append(0)
        
        # Equity Returns
        if 'equity_returns' in included_signals:
            if len(spy) > 252 and len(bil) > 252:
                spy_12m_ret = (spy.iloc[-1] - spy.iloc[-252]) / spy.iloc[-252]
                bil_12m_ret = (bil.iloc[-1] - bil.iloc[-252]) / bil.iloc[-252]
                excess_spy = spy_12m_ret - bil_12m_ret
                
                hist_excess = []
                for j in range(min_months, i):
                    past_date = month_dates[j]
                    past_data = daily_data[daily_data.index <= past_date]
                    past_spy = past_data['SPY'].dropna()
                    past_bil = past_data['BIL'].dropna()
                    if len(past_spy) > 252 and len(past_bil) > 252:
                        p_spy = (past_spy.iloc[-1] - past_spy.iloc[-252]) / past_spy.iloc[-252]
                        p_bil = (past_bil.iloc[-1] - past_bil.iloc[-252]) / past_bil.iloc[-252]
                        hist_excess.append(p_spy - p_bil)
                
                if len(hist_excess) > 12:
                    zscore = calculate_zscore(excess_spy, pd.Series(hist_excess))
                    signals['equity_returns'].append(max(-1, min(1, -1 * zscore)))
                else:
                    signals['equity_returns'].append(0)
            else:
                signals['equity_returns'].append(0)
        
        # Commodity Returns
        if 'commodity_returns' in included_signals:
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
                    signals['commodity_returns'].append(max(-1, min(1, -1 * zscore)))
                else:
                    signals['commodity_returns'].append(0)
            else:
                signals['commodity_returns'].append(0)
        
        # Yield Spread
        if 'yield_spread' in included_signals:
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
                    signals['yield_spread'].append(max(-1, min(1, zscore)))
                else:
                    signals['yield_spread'].append(0)
            else:
                signals['yield_spread'].append(0)
        
        signals['date'].append(current_date)
    
    return signals

def calculate_performance(signals_df, price_data):
    equity_data = []
    for i in range(len(signals_df) - 1):
        current_date = signals_df.index[i]
        next_date = signals_df.index[i + 1] if i + 1 < len(signals_df) else signals_df.index[-1]
        signal_pct = signals_df.loc[current_date, 'final_signal'] / 100
        
        if 'IEF' in price_data.columns and 'BIL' in price_data.columns:
            ief_prices = price_data['IEF'].dropna()
            bil_prices = price_data['BIL'].dropna()
            
            ief_start = None
            ief_end = None
            bil_start = None
            bil_end = None
            
            for idx in ief_prices.index:
                if idx >= current_date:
                    ief_start = ief_prices.loc[idx]
                    break
            for idx in ief_prices.index:
                if idx >= next_date:
                    ief_end = ief_prices.loc[idx]
                    break
            if ief_end is None:
                ief_end = ief_prices.iloc[-1]
            
            for idx in bil_prices.index:
                if idx >= current_date:
                    bil_start = bil_prices.loc[idx]
                    break
            for idx in bil_prices.index:
                if idx >= next_date:
                    bil_end = bil_prices.loc[idx]
                    break
            if bil_end is None:
                bil_end = bil_prices.iloc[-1]
            
            ief_return = (ief_end - ief_start) / ief_start if ief_start else 0
            bil_return = (bil_end - bil_start) / bil_start if bil_start else 0
            portfolio_return = signal_pct * ief_return + (1 - signal_pct) * bil_return
        else:
            portfolio_return = 0
        
        equity_data.append({'date': next_date, 'period_return': portfolio_return})
    
    equity_df = pd.DataFrame(equity_data)
    equity_df.set_index('date', inplace=True)
    equity_df['cumulative_equity'] = (1 + equity_df['period_return']).cumprod()
    
    total_return = equity_df['cumulative_equity'].iloc[-1] - 1
    n_years = len(equity_df) / 12
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    cumulative = equity_df['cumulative_equity']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    return ann_return, max_dd

if __name__ == '__main__':
    # Download data
    print('Downloading data...')
    price_data = get_etf_data([IEF, BIL, SPY, DBC], START_DATE, END_DATE)
    month_end_data = price_data.resample('ME').last()
    
    # Test all signal combinations
    combos = [
        ['yield_spread', 'bond_trend', 'equity_returns', 'commodity_returns'],  # All 4
        ['bond_trend', 'equity_returns', 'commodity_returns'],  # Without yield
        ['yield_spread', 'equity_returns', 'commodity_returns'],  # Without bond
        ['yield_spread', 'bond_trend', 'commodity_returns'],  # Without equity
        ['yield_spread', 'bond_trend', 'equity_returns'],  # Without commodity
        ['bond_trend', 'equity_returns'],  # Just 2
        ['bond_trend', 'commodity_returns'],
        ['equity_returns', 'commodity_returns'],
        ['yield_spread', 'bond_trend'],
        ['yield_spread', 'equity_returns'],
        ['yield_spread', 'commodity_returns'],
        ['bond_trend'],  # Just 1
        ['equity_returns'],
        ['commodity_returns'],
        ['yield_spread'],
    ]
    
    results = []
    print('Testing combinations...')
    
    for combo in combos:
        signals = calculate_signals_v2(month_end_data, price_data, combo)
        if not signals['date']:
            continue
        
        df = pd.DataFrame(signals)
        df.set_index('date', inplace=True)
        df['avg_signal'] = df[combo].mean(axis=1)
        df['final_signal'] = ((df['avg_signal'] + 1) / 2) * 100
        
        ann_ret, max_dd = calculate_performance(df, price_data)
        ret_dd_ratio = abs(ann_ret / max_dd) if max_dd != 0 else 0
        
        results.append({
            'signals': '+'.join(combo),
            'n_signals': len(combo),
            'ann_ret': ann_ret,
            'max_dd': max_dd,
            'ratio': ret_dd_ratio
        })
    
    # Sort by ratio
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ratio', ascending=False)
    
    print('\n' + '='*80)
    print('SIGNAL COMBINATION RESULTS (sorted by return/drawdown ratio)')
    print('='*80)
    for _, r in results_df.iterrows():
        print(f"{r['n_signals']:1d} signal(s): {r['signals']:45s} | Ann: {r['ann_ret']*100:6.2f}% | DD: {r['max_dd']*100:7.2f}% | Ratio: {r['ratio']:.2f}")
    print('='*80)