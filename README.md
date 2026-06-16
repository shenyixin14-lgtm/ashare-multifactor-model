# A-Share Multi-Factor Model: Momentum + Reversal with Out-of-Sample Validation

A cross-sectional multi-factor study on CSI 300 constituents, combining a 20-day
momentum factor and a 5-day reversal factor. The project covers the full research
loop: factor construction, direction alignment, cross-sectional standardization,
IC-weighted combination, tradability filtering, out-of-sample validation, and
transaction-cost sensitivity analysis.

## Method

- **Data**: CSI 300 constituents (~288 stocks after cleaning), daily prices via akshare (Tencent source). Fetching uses retries and failure handling.
- **Factors**: 20-day momentum and 5-day reversal, both computed from close prices.
- **Look-ahead control**: factors use only past windows; target is next-day return (`shift(-1)`).
- **Direction alignment**: both factors negated to a common "higher = bullish" form (A-shares show short-term reversal, so raw values are negatively related to future return).
- **Standardization**: cross-sectional Z-score per day, so factors of different scales carry equal weight when combined.
- **Combination**: equal-weight vs IC-weight (weights proportional to each factor's IC).
- **Tradability filter**: stocks hitting the daily price limit (|same-day return| ≥ 9.8%) are excluded, using the *same-day* return (known at trade time, no look-ahead).
- **Out-of-sample design**: IC weights fitted on 2011–2018 (train), evaluated on 2019–2026 (test). All ICIR figures are annualized.

## Key Results

Cross-sectional Rank IC (annualized ICIR), train vs out-of-sample test:

| Factor | Train IC | Test IC | Train ICIR | Test ICIR |
|---|---|---|---|---|
| Momentum (20d) | 0.0248 | 0.0172 | 1.53 | 1.26 |
| Reversal (5d) | 0.0553 | 0.0293 | 3.56 | 2.33 |
| Composite (equal) | 0.0462 | 0.0265 | 2.87 | 2.00 |
| Composite (IC-weighted) | 0.0525 | 0.0287 | 3.31 | 2.21 |

Long-short cost sensitivity (out-of-sample Sharpe, annualized):

| Cost (one-way) | Test Sharpe |
|---|---|
| 0 | 0.58 |
| 0.05% | -0.01 |
| 0.10% | -0.60 |

## Key Findings

- **Reversal dominates in A-shares.** The reversal factor is far stronger than momentum, consistent with the well-documented short-term reversal effect in China's retail-driven market — the opposite of the momentum effect seen in US equities.
- **IC-weighting is robust, not overfit.** The IC-weighted composite keeps a higher test ICIR (2.21) than the equal-weighted version (2.00), even though weights were fitted only on the train period. This confirms the weighting adds real, out-of-sample value.
- **Combination gain is limited.** The two factors are positively correlated (~0.47, since the 5-day window is nested in the 20-day one), so the composite barely beats the strongest single factor. A more complementary, low-correlation factor (e.g. volatility, turnover) would help more.
- **No factor collapses out-of-sample.** All ICs stay positive and ICIRs above 1 in the test period — the predictive power is genuine and persistent, not a sample-internal artifact.

## Lessons on Backtest Pitfalls

This project doubled as an exercise in spotting and fixing common backtest traps:

- **Look-ahead in the tradability filter.** An early version filtered stocks by *next-day* return, which inflated out-of-sample Sharpe from ~0.58 to ~1.08 by retroactively dropping volatile names. Switching to the same-day return (known at trade time) removed the bias — a near-halving of Sharpe.
- **Look-ahead in factor weights.** Using full-sample IC to set weights peeks into the future; the project uses train-period IC only, validated out-of-sample.
- **Unrealistic compounded returns.** Daily-rebalanced long-short cumulative return explodes under naive compounding (implying full reinvestment, no leverage limit, no risk control). Long-short factor portfolios are better judged by Sharpe, IC, and drawdown than by absolute return.

## Limitations

- **No neutralization.** With price-only data, the factors are not neutralized against size or industry, so part of the signal may be incidental style exposure.
- **Survivorship bias.** Current index constituents are used over the full history, which overstates performance.
- **Execution assumption.** Returns are close-to-close; real trading executes at next-day open and cannot transact names that gap to the price limit.

## Usage

```bash
pip install akshare pandas numpy matplotlib
python multifactor_ic.py
```

On first run the script fetches data and caches it to `multifactor_raw.csv`;
subsequent runs read the cache.
