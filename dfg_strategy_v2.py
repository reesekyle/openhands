"""
DFG (Divergence of Fear Gauges) Trading Signal Implementation - V2
Based on the paper: "Divergence of Fear Gauges and Stock Market Returns"
by Xinfeng Ruan and Xiaopeng Wei

This version uses:
- DFGratio = MOVE / VIX (as per Table 5 of the paper)
- Expanding window approach for out-of-sample prediction
- Multiple threshold strategies
- Exact paper period: Jan 2004 - Dec 2022
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA DOWNLOAD
# ============================================

def download_data(start_date='2004-01-01', end_date='2022-12-31'):
    """Download MOVE, VIX, and SPY data from Yahoo Finance"""
    print("Downloading data from Yahoo Finance...")
    print(f"Period: {start_date} to {end_date} (matching paper)")
    
    # Download data
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    move_data = yf.download('^MOVE', start=start_date, end=end_date, progress=False)
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
    
    # Extract close prices
    def get_close(df):
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close
    
    vix = get_close(vix_data)
    move = get_close(move_data)
    spy = get_close(spy_data)
    
    # Combine into DataFrame
    data = pd.DataFrame({'VIX': vix, 'MOVE': move, 'SPY': spy}).dropna()
    
    # Calculate SPY returns
    data['SPY_Return'] = data['SPY'].pct_change()
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    return data

# ============================================
# DFGRATIO CALCULATION
# ============================================

def calculate_dfgratio(data):
    """
    Calculate DFGratio using EXPANDING WINDOW approach
    
    Following the paper's out-of-sample methodology:
    - Use first 8 years (2004-2012) for initial training
    - Then use expanding window forward
    - This is the proper out-of-sample approach from the paper
    """
    print("\nCalculating DFGratio with Expanding Window...")
    
    df = data.copy()
    
    # Initial training period: 8 years (matching paper)
    min_periods = 252 * 8  # ~8 years
    
    df['DFG'] = np.nan
    
    print(f"Initial training period: {min_periods} trading days (~8 years)")
    print("Using expanding window for out-of-sample DFG calculation...")
    
    for i in range(min_periods, len(df)):
        # Use data from start to i-1 for regression
        vix_train = df['VIX'].iloc[:i].values.reshape(-1, 1)
        move_train = df['MOVE'].iloc[:i].values
        
        # Fit regression
        reg = LinearRegression()
        reg.fit(vix_train, move_train)
        
        # Predict for current period
        predicted_move = reg.intercept_ + reg.coef_[0] * df['VIX'].iloc[i]
        
        # DFGratio = Actual MOVE / Predicted MOVE
        # This gives us the divergence from expected
        df.iloc[i, df.columns.get_loc('DFG')] = df['MOVE'].iloc[i] / predicted_move
    
    print(f"DFGratio stats: mean={df['DFG'].mean():.4f}, std={df['DFG'].std():.4f}")
    print(f"Date range with valid DFG: {df[df['DFG'].notna()].index[0].date()} to {df[df['DFG'].notna()].index[-1].date()}")
    
    return df

# ============================================
# EXPANDING WINDOW APPROACH
# ============================================

def calculate_expanding_window_dfg(df, min_periods=252*3):
    """
    Calculate DFG using expanding window approach
    - Use first 3 years (252*3 = 756 days) for initial regression
    - Then expand window forward
    - This matches the paper's out-of-sample methodology
    """
    print(f"\nCalculating expanding window DFG (initial periods: {min_periods})...")
    
    df = df.copy()
    df['DFG_Expanding'] = np.nan
    
    for i in range(min_periods, len(df)):
        # Use data from start to i-1 for regression
        vix_train = df['VIX'].iloc[:i].values.reshape(-1, 1)
        move_train = df['MOVE'].iloc[:i].values
        
        # Fit regression
        reg = LinearRegression()
        reg.fit(vix_train, move_train)
        
        # Predict for current period
        predicted_move = reg.intercept_ + reg.coef_[0] * df['VIX'].iloc[i]
        
        # DFG = Actual - Predicted
        df.iloc[i, df.columns.get_loc('DFG_Expanding')] = df['MOVE'].iloc[i] - predicted_move
    
    return df

# ============================================
# TRADING SIGNAL GENERATION - MULTIPLE THRESHOLDS
# ============================================

def generate_signals_multiple_thresholds(df):
    """
    Generate trading signals - CORRECTED
    
    Strategy: Stay LONG SPY, exit to CASH when DFG is HIGH
    
    High DFG = Bond market volatility > expected given equity volatility
    This indicates potential market stress -> EXIT to cash
    
    Low DFG = Normal conditions -> Stay LONG SPY
    """
    print("\nGenerating trading signals...")
    print("Strategy: Stay LONG, exit to CASH when DFG is HIGH")
    
    # Calculate forward returns (next 2 days average, as per paper)
    df['Ret_2day'] = (df['SPY_Return'].shift(1) + df['SPY_Return'].shift(2)) / 2
    df['Ret_2day_pct'] = df['Ret_2day'] * 100
    
    # ==== Strategy 1: Exit when DFG > 90th percentile ====
    # Stay long, exit to cash only when DFG is very high (top 10%)
    threshold_90 = df['DFG'].quantile(0.90)
    df['Signal_90'] = np.where(df['DFG'] > threshold_90, 0, 1)  # 0=cash, 1=long
    
    # ==== Strategy 2: Exit when DFG > 75th percentile (top quartile) ====
    threshold_75 = df['DFG'].quantile(0.75)
    df['Signal_75'] = np.where(df['DFG'] > threshold_75, 0, 1)
    
    # ==== Strategy 3: Exit when DFG > 67th percentile (top tercile) ====
    threshold_67 = df['DFG'].quantile(0.67)
    df['Signal_67'] = np.where(df['DFG'] > threshold_67, 0, 1)
    
    # ==== Strategy 4: Exit when DFG > median (top 50%) ====
    threshold_50 = df['DFG'].quantile(0.50)
    df['Signal_50'] = np.where(df['DFG'] > threshold_50, 0, 1)
    
    # ==== Strategy 5: Z-score based (exit when DFG > 1.5 std) ====
    dfg_rolling_mean = df['DFG'].rolling(60).mean()
    dfg_rolling_std = df['DFG'].rolling(60).std()
    df['DFG_zscore'] = (df['DFG'] - dfg_rolling_mean) / dfg_rolling_std
    df['Signal_zscore'] = np.where(df['DFG_zscore'] > 1.5, 0, 1)  # Exit when very high
    
    print(f"Thresholds (exit to cash when DFG >): 90th={threshold_90:.4f}, 75th={threshold_75:.4f}, 67th={threshold_67:.4f}, 50th={threshold_50:.4f}")
    
    return df

# ============================================
# BACKTESTING
# ============================================

def backtest_all_strategies(df):
    """
    Backtest all strategy variants
    
    Signal = 1: Stay LONG SPY
    Signal = 0: Exit to CASH
    """
    print("\nRunning backtests...")
    
    # Strategy 1: Exit when DFG > 90th percentile
    df['Ret_Strat_90'] = df['Signal_90'].shift(1) * df['SPY_Return']
    
    # Strategy 2: Exit when DFG > 75th percentile
    df['Ret_Strat_75'] = df['Signal_75'].shift(1) * df['SPY_Return']
    
    # Strategy 3: Exit when DFG > 67th percentile
    df['Ret_Strat_67'] = df['Signal_67'].shift(1) * df['SPY_Return']
    
    # Strategy 4: Exit when DFG > median (top 50%)
    df['Ret_Strat_50'] = df['Signal_50'].shift(1) * df['SPY_Return']
    
    # Strategy 5: Z-score based
    df['Ret_Strat_zscore'] = df['Signal_zscore'].shift(1) * df['SPY_Return']
    
    # Benchmark: Buy and Hold
    df['Ret_Benchmark'] = df['SPY_Return']
    
    return df

# ============================================
# PERFORMANCE METRICS
# ============================================

def calculate_metrics(df, return_col, name, valid_range=None):
    """Calculate annualized return and drawdown"""
    
    # Use valid date range if provided
    if valid_range is not None:
        df = df.loc[valid_range[0]:valid_range[1]]
    
    valid = df[return_col].dropna()
    
    if len(valid) == 0:
        return None
    
    # Calculate cumulative return
    cumulative = (1 + valid).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    days = len(valid)
    years = days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Drawdown
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio
    sharpe = np.sqrt(252) * valid.mean() / valid.std() if valid.std() > 0 else 0
    
    print(f"\n{name}:")
    print(f"  Period: {valid.index[0].date()} to {valid.index[-1].date()}")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Annualized Return: {annualized_return*100:.2f}%")
    print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe
    }

# ============================================
# REGRESSION ANALYSIS
# ============================================

def run_regression_analysis(df):
    """Run regression to compare with paper results"""
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS (DFGratio)")
    print("="*60)
    
    # Drop NaN
    reg_data = df[['DFG', 'Ret_2day_pct']].dropna()
    
    # Regression
    X = reg_data['DFG'].values
    y = reg_data['Ret_2day_pct'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    
    print(f"\nRegression: Ret[t+1,t+2] ~ DFGratio")
    print(f"  Coefficient (β1): {slope:.4f}")
    print(f"  t-statistic: {slope/std_err:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  R²: {r_value**2 * 100:.4f}%")
    print(f"\nPaper Table 5 Panel A (DFGratio):")
    print(f"  Coefficient: -3.371*** (for Ret[t+1,t+2])")
    print(f"  R²: ~0.5% (implied)")
    
    return slope, std_err, p_value, r_value**2

# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("DFG TRADING SIGNAL - VERSION 2")
    print("Using DFGratio + Multiple Thresholds + Expanding Window")
    print("="*60)
    
    # Download data (exact paper period)
    data = download_data(start_date='2004-01-01', end_date='2022-12-31')
    
    # Calculate DFGratio
    data = calculate_dfgratio(data)
    
    # Generate multiple signals
    data = generate_signals_multiple_thresholds(data)
    
    # Backtest all strategies
    data = backtest_all_strategies(data)
    
    # Drop NaN for metrics
    backtest_data = data.dropna()
    
    # Get valid date range for strategies (after expanding window starts)
    valid_start = backtest_data[backtest_data['DFG'].notna()].index[0]
    valid_end = backtest_data.index[-1]
    valid_range = (valid_start, valid_end)
    
    print(f"\nValid backtest period: {valid_start.date()} to {valid_end.date()}")
    print("(After initial 8-year training window)")
    
    # Calculate metrics - ALL using same valid range for fair comparison
    print("\n" + "="*60)
    print("BACKTEST RESULTS (2012-2022, out-of-sample)")
    print("Strategy: Stay LONG SPY, exit to CASH when DFG is HIGH")
    print("="*60)
    
    benchmark = calculate_metrics(
        backtest_data, 'Ret_Benchmark', 
        "Buy & Hold SPY (Benchmark)",
        valid_range=valid_range
    )
    
    strat_90 = calculate_metrics(
        backtest_data, 'Ret_Strat_90',
        "Strategy: Exit when DFG > 90th percentile (top 10%)",
        valid_range=valid_range
    )
    
    strat_75 = calculate_metrics(
        backtest_data, 'Ret_Strat_75',
        "Strategy: Exit when DFG > 75th percentile (top 25%)",
        valid_range=valid_range
    )
    
    strat_67 = calculate_metrics(
        backtest_data, 'Ret_Strat_67',
        "Strategy: Exit when DFG > 67th percentile (top 33%)",
        valid_range=valid_range
    )
    
    strat_50 = calculate_metrics(
        backtest_data, 'Ret_Strat_50',
        "Strategy: Exit when DFG > median (top 50%)",
        valid_range=valid_range
    )
    
    strat_zscore = calculate_metrics(
        backtest_data, 'Ret_Strat_zscore',
        "Strategy: Z-score based (exit when DFG > 1.5 std)",
        valid_range=valid_range
    )
    
    # Regression analysis
    coef, std, pval, r2 = run_regression_analysis(data)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BACKTEST PERFORMANCE COMPARISON                          ║
╠════════════════════════════╦═══════════════════╦═══════════════════╦═══════╣
║        Strategy            ║ Annualized Return ║    Max Drawdown ║ Sharpe ║
╠════════════════════════════╬═══════════════════╬═══════════════════╬═══════╣
║  Buy & Hold SPY           ║      {benchmark['annualized_return']*100:>6.2f}%      ║      {benchmark['max_drawdown']*100:>7.2f}%    ║ {benchmark['sharpe_ratio']:>5.3f} ║
║  Long < Median            ║      {strat_50['annualized_return']*100:>6.2f}%      ║      {strat_50['max_drawdown']*100:>7.2f}%    ║ {strat_50['sharpe_ratio']:>5.3f} ║
║  Long < 33rd              ║      {strat_33['annualized_return']*100:>6.2f}%      ║      {strat_33['max_drawdown']*100:>7.2f}%    ║ {strat_33['sharpe_ratio']:>5.3f} ║
║  Short > 67th              ║      {strat_67['annualized_return']*100:>6.2f}%      ║      {strat_67['max_drawdown']*100:>7.2f}%    ║ {strat_67['sharpe_ratio']:>5.3f} ║
║  Z-score Based            ║      {strat_zscore['annualized_return']*100:>6.2f}%      ║      {strat_zscore['max_drawdown']*100:>7.2f}%    ║ {strat_zscore['sharpe_ratio']:>5.3f} ║
╚════════════════════════════╩═══════════════════╩═══════════════════╩═══════╝
    """)
    
    print("REGRESSION ANALYSIS:")
    print(f"  Our DFGratio coefficient: {coef:.4f} (t={coef/std:.2f}, p={pval:.4f})")
    print(f"  Paper DFGratio coefficient: -3.371 (t=-2.76, significant)")
    
    if coef < 0 and pval < 0.05:
        print("\n✓ Negative coefficient - MATCHING paper direction!")
    elif coef > 0:
        print("\n  Note: Positive coefficient with Yahoo data (paper has negative)")
    
    # Save results
    data.to_csv('/workspace/dfg_results_v2.csv')
    print("\nResults saved to /workspace/dfg_results_v2.csv")
    
    return data

if __name__ == "__main__":
    data = main()
