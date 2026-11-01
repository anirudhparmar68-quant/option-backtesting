# Session summary: NIFTY / SENSEX options backtesting (Upstox data, Sep 2024 - Oct 2026)

Each strategy has `<name>.html` (QuantStats tearsheet vs NIFTY), `<name>_metrics.csv` and
`<name>_daily_pnl.csv`. `summary.csv` has all numbers. Capital = rungs x Rs 1.5 lakh.
Holdout = Jul-Oct 2026 (never used to choose anything).

| Strategy | Capital | Net (2 yrs) | Max DD | Ret/DD | CAGR | Sharpe | Sortino | Loss months | Holdout Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| A Single 09:20 straddle (N+S) | 0.03 cr | 2.85 L | 0.50 L | 5.7 | 54% | 1.83 | 2.91 | 4/26 | 2.62 |
| B Staggered straddle 09:20-13:00, SL price, 0 tick | 6.63 cr | 5.55 cr | 56.0 L | 9.9 | 49% | 3.58 | 6.09 | 2/26 | 6.02 |
| C B + skip gap > 1% | 6.63 cr | 5.49 cr | 24.1 L | 22.8 | 49% | 3.94 | 7.09 | 1/26 | 5.96 |
| D CCS: close-confirmed, next-minute, 1 tick, gap filter | 6.63 cr | 3.61 cr | 39.2 L | 9.2 | 31% | 2.44 | 3.77 | 5/26 | 3.34 |
| E CCS + vol targeting | 6.63 cr | 3.31 cr | 29.3 L | 11.3 | 29% | 2.53 | 3.93 | 4/26 | 3.25 |
| F Staggered straddle, 0.2% index SL | 6.63 cr | 3.19 cr | 69.8 L | 4.6 | 25% | 1.81 | 2.64 | 5/26 | 2.66 |
| G NIFTY Rs100 strangle, SL price, 0 tick | 3.315 cr | 2.98 cr | 26.1 L | 11.4 | 54% | 4.13 | 7.37 | 1/26 | 6.01 |
| H NIFTY Rs100 strangle, realistic, 1 tick | 3.315 cr | 2.20 cr | 38.3 L | 5.7 | 37% | 2.90 | 4.63 | 4/26 | 3.42 |
| I S1 iron fly (user rules), 3 ticks | 3.315 cr | 44.5 L | 57.2 L | 0.8 | 6% | 0.49 | 0.71 | 9/26 | 1.13 |
| J S1 iron fly, 7-delta wings, no re-entry, 3 ticks | 3.315 cr | 85.0 L | 33.9 L | 2.5 | 13% | 1.04 | 1.63 | 9/26 | 2.72 |
| K S2 iron condor (user rules), 3 ticks | 3.315 cr | -1.03 cr | 1.04 cr | -1.0 | -14% | -4.82 | -5.68 | 23/26 | -5.09 |
| L Call-calendar ladder (user idea), 1 tick | 0.21 cr* | 2.17 L | 0.78 L | 2.8 | 5%* | 0.96 | 1.29 | 9/26 | 1.24 |
| M Calendar ladder tuned (exit 09:30 expiry, last 3 days) | 0.15 cr* | 1.89 L | 0.33 L | 5.8 | 6%* | 1.97 | 3.78 | 9/26 | 2.53 |

*Calendars are hedged and need far less than Rs 1.5 lakh per rung, so their capital is
overstated and CAGR understated.

## Key lessons
1. The SL fill assumption decides everything for stop-based short premium: SL-price fills (B, G)
   look great; filling at the minute close turns B into a loss. Close-confirmed stops executed
   the next minute (D, H) are realistic and still profitable.
2. Skipping days whose open gaps > 1% halves drawdowns (B -> C).
3. Rs100-premium strangle beats the ATM straddle on risk (H vs D, NIFTY).
4. On expiry day, trading the next expiry instead of 0-DTE helps.
5. Calendars are long vega: they earn in volatility spikes (Apr 2025) and have ~0 correlation
   with the short-straddle book - a natural hedge.
6. Ideas that failed honestly (research_log.md): gamma-neutral calendar, OI pinning, no-stop
   fly, NIFTY-SENSEX pair trade, opening-range breakout (options and synthetic futures),
   overnight straddle (the SENSEX "Sharpe 5.7" was a data artifact), 99-million-combination
   search (overfit; failed walk-forward and holdout), account profit locks.
