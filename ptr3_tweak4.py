#!/usr/bin/env python3
"""
PTR3 - Tweak 4: International Bond ETF Momentum

Add signal based on EMB (iShares J.P. Morgan USD Emerging Markets Bond ETF) momentum
If EMB momentum positive: +1 (indicating global bond strength)
If EMB momentum negative: -1
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

START_DATE = "2006-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

IEF = "IEF"
BIL = "BIL"
SPY = "SPY"
DBC = "DBC"
EMB = "EMB"  # Emerging markets bonds
# Also try BNDX (Total International Bond ETF)

def calculate_zscore(value, historical_values):
    if len(historical_values) < 30:
        return 0
    mean = historical_values.mean()
    std = historical_values.std()
    if std == 0:
        return 0
    zscore = (value - mean) / std
    return max(-1, min(1, zscore))

def get_etf_data(tickers, start_date, end_date):
    all_data = {}
    for ticker in tickers:
        try:
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
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def calculate_signals(month_end_data, daily_data):
    signals = {
        'yield_spread': [],
        'bond_trend': [],
        'equity_returns': [],
        'commodity_returns': [],
        'intl_bond_momentum': [],  # NEW: Tweak 4
        'date': []
    }
    
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
        emb = data['EMB'].dropna() if 'EMB' in data.columns else None
        
        # Bond Trend
        if len(ief) > 252 and len(bil) > 252:
            ief_12m = (ief.iloc[-1] - ief.iloc[-252]) / ief.iloc[-252]
            bil_12m = (bil.iloc[-1] - bil.iloc[-252]) / bil.iloc[-252]
            signals['bond_trend'].append(1 if ief_12m > bil_12m else -1)
        else:
            signals['bond_trend'].append(0)
        
        # Equity Returns
        if len(spy) > 252 and len(bil) > 252:
            spy_12m = (spy.iloc[-1] - spy.iloc[-252]) / spy.iloc[-252]
            bil_12m = (bil.iloc[-1] - bil.iloc[-252]) / bil.iloc[-252]
            excess = spy_12m - bil_12m
            
            hist_excess = []
            for j in range(min_months, i):
                pdata = daily_data[daily_data.index <= month_dates[j]]
                ps = pdata['SPY'].dropna()
                pb = pdata['BIL'].dropna()
                if len(ps) > 252 and len(pb) > 252:
                    hist_excess.append((ps.iloc[-1] - ps.iloc[-252]) / ps.iloc[-252] - 
                                  (pb.iloc[-1] - pb.iloc[-252]) / pb.iloc[-252])
            
            if len(hist_excess) > 12:
                zscore = calculate_zscore(excess, pd.Series(hist_excess))
                signals['equity_returns'].append(max(-1, min(1, -1 * zscore)))
            else:
                signals['equity_returns'].append(0)
        else:
            signals['equity_returns'].append(0)
        
        # Commodity Returns
        if len(dbc) > 252:
            dbc_12m = (dbc.iloc[-1] - dbc.iloc[-252]) / dbc.iloc[-252]
            
            hist_dbc = []
            for j in range(min_months, i):
                pdata = daily_data[daily_data.index <= month_dates[j]]
                pdbc = pdata['DBC'].dropna()
                if len(pdbc) > 252:
                    hist_dbc.append((pdbc.iloc[-1] - pdbc.iloc[-252]) / pdbc.iloc[-252])
            
            if len(hist_dbc) > 12:
                zscore = calculate_zscore(dbc_12m, pd.Series(hist_dbc))
                signals['commodity_returns'].append(max(-1, min(1, -1 * zscore)))
            else:
                signals['commodity_returns'].append(0)
        else:
            signals['commodity_returns'].append(0)
        
        # Yield Spread
        if len(spy) > 252 and len(bil) > 252:
            spy_ret = (spy.iloc[-1] - spy.iloc[-252]) / spy.iloc[-252]
            bil_ret = (bil.iloc[-1] - bil.iloc[-252]) / bil.iloc[-252]
            spread = spy_ret - bil_ret
            
            hist_spread = []
            for j in range(min_months, i):
                pdata = daily_data[daily_data.index <= month_dates[j]]
                ps = pdata['SPY'].dropna()
                pb = pdata['BIL'].dropna()
                if len(ps) > 252 and len(pb) > 252:
                    hist_spread.append((ps.iloc[-1] - ps.iloc[-252]) / ps.iloc[-252] - 
                                  (pb.iloc[-1] - pb.iloc[-252]) / pb.iloc[-252])
            
            if len(hist_spread) > 12:
                zscore = calculate_zscore(spread, pd.Series(hist_spread))
                signals['yield_spread'].append(max(-1, min(1, zscore)))
            else:
                signals['yield_spread'].append(0)
        else:
            signals['yield_spread'].append(0)
        
        # TWEAK 4: International bond momentum (EMB)
        if emb is not None and len(emb) > 252:
            emb_12m = (emb.iloc[-1] - emb.iloc[-252]) / emb.iloc[-252]
            
            # Get z-score of EMB returns
            hist_emb = []
            for j in range(min_months, i):
                pdata = daily_data[daily_data.index <= month_dates[j]]
                pemb = pdata['EMB'].dropna() if 'EMB' in pdata.columns else None
                if pemb is not None and len(pemb) > 252:
                    hist_emb.append((pemb.iloc[-1] - pemb.iloc[-252]) / pemb.iloc[-252])
            
            if len(hist_emb) > 12:
                zscore = calculate_zscore(emb_12m, pd.Series(hist_emb))
                signals['intl_bond_momentum'].append(zscore)
            else:
                signals['intl_bond_momentum'].append(0)
        else:
            signals['intl_bond_momentum'].append(0)
        
        signals['date'].append(current_date)
    
    return signals

def create_signals_dataframe(signals):
    df = pd.DataFrame(signals)
    df.set_index('date', inplace=True)
    df['avg_signal'] = df[['yield_spread', 'bond_trend', 'equity_returns', 'commodity_returns', 'intl_bond_momentum']].mean(axis=1)
    df['final_signal'] = ((df['avg_signal'] + 1) / 2) * 100
    return df

def calculate_portfolio_equity(signals_df, price_data):
    equity_data = []
    for i in range(len(signals_df) - 1):
        current_date = signals_df.index[i]
        next_date = signals_df.index[i + 1] if i + 1 < len(signals_df) else signals_df.index[-1]
        signal_pct = signals_df.loc[current_date, 'final_signal'] / 100
        
        if 'IEF' in price_data.columns and 'BIL' in price_data.columns:
            ief_prices = price_data['IEF'].dropna()
            bil_prices = price_data['BIL'].dropna()
            
            ief_start, ief_end, bil_start, bil_end = None, None, None, None
            
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
    return equity_df

def calculate_performance(equity_df):
    total_return = equity_df['cumulative_equity'].iloc[-1] - 1
    n_years = len(equity_df) / 12
    ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
    cumulative = equity_df['cumulative_equity']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    return ann_return, max_dd

if __name__ == '__main__':
    print("Tweak 4: International Bond ETF Momentum (EMB)")
    print("=" * 50)
    
    price_data = get_etf_data([IEF, BIL, SPY, DBC, EMB], START_DATE, END_DATE)
    month_end_data = price_data.resample('ME').last()
    
    signals = calculate_signals(month_end_data, price_data)
    signals_df = create_signals_dataframe(signals)
    signals_df.to_excel('ptr3_tweak4_position.xlsx')
    print("Saved ptr3_tweak4_position.xlsx")
    
    plt.figure(figsize=(12, 6))
    plt.plot(signals_df.index, signals_df['final_signal'], 'b-', linewidth=1.5)
    plt.fill_between(signals_df.index, signals_df['final_signal'], alpha=0.3)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    plt.title('PTR3 Tweak 4: Intl Bond Momentum (0-100%)')
    plt.xlabel('Date')
    plt.ylabel('Signal (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ptr3_tweak4_position.png', dpi=150)
    plt.close()
    print("Saved ptr3_tweak4_position.png")
    
    equity_df = calculate_portfolio_equity(signals_df, price_data)
    equity_df.to_excel('ptr3_tweak4_equity.xlsx')
    print("Saved ptr3_tweak4_equity.xlsx")
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['cumulative_equity'], 'g-', linewidth=1.5)
    plt.title('PTR3 Tweak 4: Cumulative Portfolio Equity')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Equity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ptr3_tweak4_equity.png', dpi=150)
    plt.close()
    print("Saved ptr3_tweak4_equity.png")
    
    ann_return, max_dd = calculate_performance(equity_df)
    print(f"\n{'='*50}")
    print("PTR3 Tweak 4 Performance")
    print(f"{'='*50}")
    print(f"Annualized Return: {ann_return*100:.2f}%")
    print(f"Maximum Drawdown: {max_dd*100:.2f}%")
    print(f"Return/DD Ratio: {abs(ann_return/max_dd):.2f}")
    print(f"{'='*50}")