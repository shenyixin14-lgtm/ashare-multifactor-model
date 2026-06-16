#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shenyixin
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak

SYMBOL = "000300"          # CSI 300 index code
MOMENTUM_WINDOW = 20       # Momentum look-back window
REVERSAL = 5               # Reversal look-back window (trading days)
Q = 0.2                    # Long/short quantile (top & bottom 20%)
COST_RATE = 0.001          # transaction cost of portfolio adjustment
RETRIES = 2                # Number of retries
SLEEP = 0.5                # Delay between stocks
COST_GRID = [0, 0.0005, 0.001, 0.0015, 0.002]   # Cost sensitivity sweep

def get_one(code, momentum_window=MOMENTUM_WINDOW, reversal=REVERSAL,retries=RETRIES):
    for attempt in range(retries):          
        try:
            if code.startswith('6'):
                name = 'sh' + code
            else:
                name = 'sz' + code
            raw = ak.stock_zh_a_hist_tx(symbol=name)
            result = raw[['date','close']].copy()
            result['code'] = code
            result['momentum'] = result['close'].pct_change(momentum_window)
            result['reversal'] = result['close'].pct_change(reversal)
            result['today_chg'] = result['close'].pct_change()
            result['next_ret'] = result['close'].pct_change().shift(-1)
            return result                    
        except Exception as e:
            print(f"Attempt{attempt+1}failed: {type(e).__name__},retrying...")
            time.sleep(3)                   
    return None

def get_more(codes, momentum_window=MOMENTUM_WINDOW, reversal=REVERSAL, f=get_one):
    frame = []
    total = len(codes)
    success = 0
    for i, code in enumerate(codes, 1):
        one = f(code,momentum_window,reversal)
        if one is not None:
            frame.append(one)
            success += 1
        print(f"progress {i}/{total} | success {success} | current {code}")
        time.sleep(SLEEP)
    panel = pd.concat(frame, ignore_index=True)
    panel = panel.dropna(subset=['reversal','momentum','next_ret','today_chg'])
    return panel

def align_direction(panel):
    panel = panel.copy()
    panel['momentum_aligned'] = -panel['momentum']
    panel['reversal_aligned'] = -panel['reversal']
    return panel

def standardize(panel):
    panel = panel.copy()
    for col in ['momentum_aligned', 'reversal_aligned']:
        panel[col + '_z'] = panel.groupby('date')[col].transform(lambda x:(x - x.mean()) / x.std())
    return panel

def filter_tradable(panel):
    panel = panel.copy()
    tradable = panel['today_chg'].abs() < 0.098
    panel = panel[tradable]
    return panel 
    
def cross_ic(x, factor_col, min_stocks=5):
    x = x.dropna(subset=[factor_col, "next_ret"])
    if len(x) < min_stocks:                    
        return np.nan
    return x[factor_col].corr(x["next_ret"], method="spearman")

def compute_daily_ic(panel, factor_col, f=cross_ic):
    daily_ic = panel.groupby("date").apply(lambda x:f(x, factor_col)).dropna()
    return daily_ic

def summarize_ic(daily_ic):
    ic_mean = daily_ic.mean()
    icir    = ic_mean / daily_ic.std() * np.sqrt(252)
    ic_win  = (daily_ic > 0).mean()
    return {'ic_mean': ic_mean, 'icir': icir, 'ic_win': ic_win}

def plot_cumulative_ic(daily_ic, momentum_window=20, reversal=5, name=None):
    daily_ic.cumsum().plot(figsize=(8, 5), title=f'Cumulative IC ({name})')
    plt.show()

def strategy(x, q=Q, min_stocks=10):
    if len(x) < min_stocks:
        return np.nan
    is_short = x['composite_icw'] < x['composite_icw'].quantile(q)
    is_long = x['composite_icw'] > x['composite_icw'].quantile(1-q)
    long_ret  = x.loc[is_long, 'next_ret'].mean()
    short_ret = x.loc[is_short, 'next_ret'].mean()
    return long_ret - short_ret

def calc_sharpe_max_dd(panel, cost_rate=COST_RATE, trading_days=252):
    daily_ret = panel.groupby('date').apply(strategy).dropna()
    daily_net_ret = daily_ret - cost_rate
    sharpe = daily_net_ret.mean() / daily_net_ret.std() * np.sqrt(trading_days)
    acc = (1 + daily_net_ret).cumprod()
    max_dd = (acc / acc.cummax() - 1).min()
    return {'sharpe': sharpe, 'max_dd': max_dd}

if __name__ == "__main__":
    if os.path.exists('multifactor_raw.csv'):
        panel = pd.read_csv('multifactor_raw.csv')
        panel['date'] = pd.to_datetime(panel['date'])
    else:
        cons = ak.index_stock_cons(symbol=SYMBOL)
        codes = cons['品种代码'].astype(str).str.zfill(6).tolist()
        panel = get_more(codes, momentum_window=MOMENTUM_WINDOW, reversal=REVERSAL)
        panel.to_csv('multifactor_raw.csv', index=False)
    print(panel.shape, panel['code'].nunique())
    p = align_direction(panel)
    p_s = standardize(p)
    panel = filter_tradable(p_s)
    train = panel[panel['date'] < '2019-01-01'].copy()
    test  = panel[panel['date'] >= '2019-01-01'].copy()
    ic_mom = summarize_ic(compute_daily_ic(train, 'momentum_aligned_z'))['ic_mean']
    ic_rev = summarize_ic(compute_daily_ic(train, 'reversal_aligned_z'))['ic_mean']
    w_mom = ic_mom / (ic_mom + ic_rev)
    w_rev = 1 - w_mom
    
    for df in [train, test]:
        df['composite'] = df['momentum_aligned_z'] + df['reversal_aligned_z']   
        df['composite_icw'] = w_mom*df['momentum_aligned_z'] + w_rev*df['reversal_aligned_z']    
    for col in ['momentum_aligned_z', 'reversal_aligned_z', 'composite', 'composite_icw']:
        stats_train = summarize_ic(compute_daily_ic(train, col))
        stats_test  = summarize_ic(compute_daily_ic(test, col))
        print(f"{col} train: IC={stats_train['ic_mean']:.4f}, ICIR={stats_train['icir']:.4f}, IC_WIN={stats_train['ic_win']:.2%}")
        print(f"{col} test:  IC={stats_test['ic_mean']:.4f}, ICIR={stats_test['icir']:.4f}, IC_WIN={stats_test['ic_win']:.2%}")
    for cost_rate in COST_GRID:
        r_train = calc_sharpe_max_dd(train, cost_rate)
        r_test  = calc_sharpe_max_dd(test, cost_rate)
        print(f"cost={cost_rate} train: sharpe={r_train['sharpe']:.4f}, max_dd={r_train['max_dd']:.2%}")
        print(f"cost={cost_rate} test:  sharpe={r_test['sharpe']:.4f}, max_dd={r_test['max_dd']:.2%}")